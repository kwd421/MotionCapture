"""Real stage/decoder/array tests with explicit synthetic, non-benchmark models."""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest
from test_recording import tiny_vfr as tiny_vfr
from test_wholebody_handoff import Session
from test_wholebody_optimization import boxes

from motioncapture import wholebody_optimize_bench as bench
from motioncapture.recording import RecordedFrame, RecordedIdentity, inspect_recording
from motioncapture.wholebody_catalog import BenchmarkError
from motioncapture.wholebody_replay import Release, ReplayAges, SourcePacer
from motioncapture.wholebody_stage_comparison import StageDifference, StageReference, update_digest
from motioncapture.wholebody_stages import StagePipeline


class FingerprintModel(Session):
    """Input-dependent synthetic heads, intentionally reused per private session."""
    def __init__(self, kind="pose", pattern=(2,)):
        super().__init__(kind)
        self.pattern = pattern
        self.inputs = []
        self.x = np.zeros((1, 133, 384), np.float32)
        self.y = np.zeros((1, 133, 512), np.float32)

    def run(self, tensor):
        self.called_on.append(threading.get_ident())
        self.calls += 1
        if self.kind == "detector":
            return boxes(self.pattern[(self.calls - 1) % len(self.pattern)])
        signature = hashlib.sha256(tensor.tobytes()).digest()
        self.inputs.append(signature)
        self.x.fill(0)
        self.y.fill(0)
        self.x[:, :, int.from_bytes(signature[:2], "little") % 384] = .8
        self.y[:, :, int.from_bytes(signature[2:4], "little") % 512] = .6
        return [self.x, self.y]


def source(n):
    image = np.random.default_rng(128).integers(0, 256, (160, 224, 3), np.uint8)
    for i in range(n):
        yield RecordedFrame(RecordedIdentity("fixture", "stream", i, i*1500+(i//2)*80,
                                             Fraction(1, 90000)), image.copy(), .1)


def factory_set(pattern=(2,), pose_class=FingerprintModel):
    made = []
    def make(kind):
        model = FingerprintModel(kind, pattern) if kind == "detector" else pose_class()
        made.append(model)
        return model
    return (lambda: make("detector"), lambda: make("pose")), made


def digest_packets(packets):
    digest = hashlib.sha256()
    for packet in packets:
        update_digest(digest, packet.detected.frame, packet.people)
    return digest.hexdigest()


@pytest.mark.parametrize("people", [0, 1, 2, 3, 8])
def test_exact_inputs_all_slots_order_and_source_immutability(people):
    original = list(source(3))
    before = [frame.image_bgr.tobytes() for frame in original]
    completed, inputs = [], []
    for lanes in (1, 2):
        factories, made = factory_set((people,))
        with StagePipeline(*factories, normalization_kernel="opencv", pose_lanes=lanes) as pipe:
            packets = list(pipe.packets(original, 3, overlap=True, fast=True, advance_pose=True))
        completed.append(packets)
        assert [len(p.people) for p in packets] == [people]*3
        assert [p.detected.frame.identity.sequence for p in packets] == [0, 1, 2]
        pose_models = [m for m in made if m.kind == "pose"]
        assert sum(m.calls for m in pose_models) == people*3
        if lanes == 2:
            assert pose_models[1].calls == (people//2)*3
        inputs.append(sorted(item for model in pose_models for item in model.inputs))
        assert pipe.snapshot()["model_stages"] == lanes + 1
        assert pipe.snapshot()["unemitted_pose_requests"] == 0
        assert all(m.closed_on == m.created_on for m in made)
        assert len({m.created_on for m in made}) == lanes + 1
        assert all(v == "owner_released" for v in pipe.snapshot()["cleanup"].values())
        for packet in packets:
            assert sum(packet.per_person_ms) == pytest.approx(packet.times["pose_inference_ms"])
            assert all(not p.xy.flags.writeable for p in packet.people)
    assert inputs[0] == inputs[1]
    assert digest_packets(completed[0]) == digest_packets(completed[1])
    assert [f.image_bgr.tobytes() for f in original] == before


def test_both_lanes_enter_native_work_before_either_finishes_without_sleep():
    barrier = threading.Barrier(2)
    class Overlap(FingerprintModel):
        def run(self, tensor):
            if self.calls == 0:
                barrier.wait(timeout=3)  # Serial execution cannot pass this barrier.
            return super().run(tensor)
    factories, made = factory_set((2,), Overlap)
    with StagePipeline(*factories, pose_lanes=2, normalization_kernel="opencv") as pipe:
        output = list(pipe.packets(source(1), 1, overlap=True, fast=True, advance_pose=True))
    assert len(output[0].people) == 2
    assert all(m.calls == 1 for m in made)
    assert output[0].times["pose_auxiliary_join_ms"] >= 0


@pytest.mark.parametrize("bad_lane", ["primary", "auxiliary", "both"])
def test_native_failure_drains_both_lanes_without_retry_or_partial_people(bad_lane):
    barrier = threading.Barrier(2)
    made = []
    next_lane = [0]
    def detector_factory():
        model = FingerprintModel("detector", (2,))
        made.append(model)
        return model
    class Failing(FingerprintModel):
        def __init__(self, lane):
            super().__init__()
            self.lane = lane
        def run(self, tensor):
            super().run(tensor)
            barrier.wait(timeout=3)
            if bad_lane in (self.lane, "both"):
                raise BenchmarkError(f"injected_{self.lane}")
            return [self.x, self.y]
    def pose_factory():
        lane = "primary" if next_lane[0] == 0 else "auxiliary"
        next_lane[0] += 1
        model = Failing(lane)
        made.append(model)
        return model
    expected = "primary" if bad_lane == "both" else bad_lane
    pipe = StagePipeline(detector_factory, pose_factory, pose_lanes=2)
    with pytest.raises(BenchmarkError, match=f"injected_{expected}"):
        with pipe:
            list(pipe.packets(source(1), 1, overlap=True, fast=False))
    assert pipe.snapshot()["emitted_frames"] == 0
    assert all(m.calls == 1 for m in made)
    assert all(m.closed_on == m.created_on for m in made)


def test_auxiliary_constructor_failure_closes_primary_and_never_reads_source():
    made = []
    def detector():
        model = FingerprintModel("detector")
        made.append(model)
        return model
    def pose():
        if len(made) == 2:
            raise BenchmarkError("auxiliary_constructor_failed")
        model = FingerprintModel()
        made.append(model)
        return model
    pipe = StagePipeline(detector, pose, pose_lanes=2)
    with pytest.raises(BenchmarkError, match="auxiliary_constructor_failed"):
        with pipe:
            pytest.fail("must not enter")
    assert all(m.closed_on == m.created_on and m.calls == 0 for m in made)
    assert pipe.snapshot()["read_frames"] == 0
    assert pipe.snapshot()["cleanup"]["auxiliary_pose"] == "unknown_after_constructor_failure"


def test_auxiliary_close_failure_is_not_lost_or_made_success():
    count = [0]
    class BadClose(FingerprintModel):
        def close(self):
            super().close()
            raise BenchmarkError("injected_auxiliary_close")
    def pose_factory():
        count[0] += 1
        return BadClose() if count[0] == 2 else FingerprintModel()
    pipe = StagePipeline(lambda: FingerprintModel("detector"), pose_factory, pose_lanes=2)
    with pytest.raises(BenchmarkError, match="injected_auxiliary_close"):
        with pipe:
            list(pipe.packets(source(1), 1, overlap=True, fast=False))
    assert pipe.snapshot()["cleanup"]["auxiliary_pose"] == "failed"


def test_early_stop_cancels_future_source_wait_and_joins_all_model_owners():
    waiting = threading.Event()
    pacer = SourcePacer(2)
    def wait(seconds):
        waiting.set()
        return pacer._stop.wait(seconds)
    pacer._wait = wait
    class WaitForNext(FingerprintModel):
        def run(self, tensor):
            assert waiting.wait(3), "detector must reach next-source wait"
            return super().run(tensor)
    def far_frames():
        for f in source(3):
            yield replace(f, identity=replace(f.identity, pts=f.identity.sequence*900000))
    factories, made = factory_set((2,), WaitForNext)
    pipe = StagePipeline(*factories, pose_lanes=2, pacer=pacer)
    with pipe:
        stream = pipe.packets(far_frames(), 2, overlap=True, fast=False, advance_pose=True)
        first = next(stream)
        assert len(first.people) == 2 and waiting.is_set()
        stream.close()
    assert pacer.cancelled and pacer.released == 1
    assert pipe.snapshot()["read_frames"] == 2
    assert all(m.closed_on == m.created_on for m in made)


def test_multibox_hash_changes_are_visible_even_when_coordinate_matching_is_ambiguous():
    frame = next(source(1))
    factories, _ = factory_set((2,))
    with StagePipeline(*factories) as pipe:
        people = next(pipe.packets([frame], 1, overlap=False, fast=False)).people
    reference = StageReference(1, frame_hashes=True)
    reference.store(frame, people)
    changed_scores = people[1].scores.copy()
    changed_scores[0] += .01
    changed = [people[0], replace(people[1], scores=changed_scores)]
    comparison = StageDifference()
    comparison.add(reference, frame, changed)
    row = comparison.summary()
    assert row["ambiguous_multi_person_frames"] == 1
    assert row["parts"]["body"]["max_pixels"] is None
    assert row["per_frame_predictions"]["multi_person_frames_checked"] == 1
    assert row["per_frame_predictions"]["multi_person_frames_changed"] == 1
    assert row["per_frame_predictions"]["first_changes"][0]["sequence"] == 0


def test_workload_age_records_remain_bounded_and_preserve_absent_workload():
    observer = ReplayAges(20)
    assert observer.summary()["source_age_by_ending_person_count"] == {}
    for i in range(20):
        ident = RecordedIdentity("s", "v", i, i*1500, Fraction(1, 90000))
        due = i*20_000_000
        observer.add(ident, Release(due, due+1000, 0), due+5_000_000+i*1_000_000,
                     people=i % 3, pose_stage_ms=1.0)
    row = observer.summary()
    assert row["source_age_ms"]["mean_ms"] == 14.5
    assert sum(r["samples"] for r in row["source_age_by_ending_person_count"].values()) == 20
    assert len(row["worst_source_ages"]) == 16
    assert row["worst_source_ages"][0]["sequence"] == 19
    assert row["worst_source_ages"][0]["source_age_ms"] == 24
    assert row["thresholds_are_diagnostics_not_acceptance_criteria"] is True


def test_invalid_workload_does_not_mutate_replay_prefix():
    observer = ReplayAges(1)
    before = observer.summary()
    ident = RecordedIdentity("s", "v", 0, 0, Fraction(1, 90000))
    with pytest.raises(BenchmarkError, match="workload"):
        observer.add(ident, Release(0, 0, 0), 1, people=9)
    assert observer.summary() == before


def test_real_vfr_paced_abba_hashes_all_people_and_no_clobber(tiny_vfr, tmp_path, monkeypatch):
    args = bench.parser().parse_args([str(tiny_vfr), "--research-only", "--max-frames", "0",
                                     "--suite", "pose-parallel",
                                     "--output", str(tmp_path/"a.json")])
    monkeypatch.setattr(bench, "verify_asset", lambda root, key: (tmp_path/key, {"fixture": True}))
    observed = []
    def run(args, probe, mode, factories, reference):
        factories, made = factory_set((1, 2, 0, 3, 8, 1))
        row, bank = bench.run_pass(args, probe, mode, factories, reference)
        observed.append(made)
        return row, bank
    assert bench.execute(args, pass_runner=run) == 0
    data = json.loads(args.output.read_text())
    assert [r["pose_lanes"] for r in data["runs"]] == [1, 2, 2, 1]
    assert data["all_pass_prediction_hashes_equal"]
    assert data["all_pass_detector_hashes_equal"] and data["all_pass_pixel_hashes_equal"]
    n = len(inspect_recording(tiny_vfr).pts)
    for i, row in enumerate(data["runs"]):
        assert row["all_frames"]["frames"] == n
        assert row["all_frames"]["person_observations"] == 15
        assert row["paced_loop_fps"] is not None and row["unpaced_loop_fps"] is None
        assert row["pipeline"]["source_pacing"]["released_frames"] == n
        assert row["pipeline"]["unemitted_pose_requests"] == 0
        assert row["pipeline"]["source_pacing"]["clock_rebases"] == 0
        assert row["live_60fps_verified"] is False
        if i:
            h = row["provider_disagreement"]["per_frame_predictions"]
            assert h["checked_frames"] == n and h["changed_frames"] == 0
            assert h["multi_person_frames_checked"] == 3
        assert sum(m.calls for m in observed[i] if m.kind == "pose") == 15
    assert len(list(tmp_path.glob("a.arm-??.json"))) == 4
    with pytest.raises(FileExistsError):
        bench.execute(args, pass_runner=run)


def test_candidate_failure_preserves_prefix_and_stops_without_serial_fallback(
        tiny_vfr, tmp_path, monkeypatch):
    args = bench.parser().parse_args([str(tiny_vfr), "--research-only", "--suite", "pose-parallel",
                                     "--max-frames", "0", "--output", str(tmp_path/"failed.json")])
    monkeypatch.setattr(bench, "verify_asset", lambda root, key: (tmp_path/key, {}))
    class Failing(FingerprintModel):
        def run(self, tensor):
            raise BenchmarkError("injected_candidate")
    def run(args, probe, mode, factories, reference):
        model_class = Failing if mode == "source-pts-dualpose-cvlut" else FingerprintModel
        factories, _ = factory_set((1, 2, 0), model_class)
        return bench.run_pass(args, probe, mode, factories, reference)
    assert bench.execute(args, pass_runner=run) == 2
    data = json.loads(args.output.read_text())
    assert len(data["runs"]) == 2 and data["status"] == "failed"
    assert data["runs"][0]["status"] == "completed"
    assert data["runs"][1]["paced_loop_fps"] is None
    assert data["runs"][1]["error"]["code"] == "injected_candidate"
    assert data["runs"][1]["hash_scope"] == "completed_prefix"
    assert all(v == "owner_released" for v in data["runs"][1]["pipeline"]["cleanup"].values())


@pytest.mark.parametrize("lanes", [0, 3, True])
def test_invalid_lane_count_does_not_construct_native_owners(lanes):
    with pytest.raises(BenchmarkError, match="lane_count"):
        StagePipeline(lambda: pytest.fail("must not construct"),
                      lambda: pytest.fail("must not construct"), pose_lanes=lanes)


def test_runner_retains_preflights_before_auxiliary_constructor_failure(tiny_vfr, tmp_path):
    args = bench.parser().parse_args([str(tiny_vfr), "--research-only", "--suite", "pose-parallel",
                                     "--output", str(tmp_path/"never-written.json")])
    calls = [0]
    def pose_factory():
        calls[0] += 1
        if calls[0] == 2:
            raise BenchmarkError("auxiliary_open_failed")
        return FingerprintModel()
    row, bank = bench.run_pass(args, inspect_recording(tiny_vfr), "source-pts-dualpose-cvlut",
                              (lambda: FingerprintModel("detector"), pose_factory))
    assert bank is None and row["status"] == "failed"
    assert row["all_frames"]["frames"] == 0 and row["paced_loop_fps"] is None
    assert set(row["backend"]) == {"detector", "pose"}
    assert row["error"]["code"] == "auxiliary_open_failed"
    assert row["pipeline"]["cleanup"]["auxiliary_pose"] == "unknown_after_constructor_failure"
