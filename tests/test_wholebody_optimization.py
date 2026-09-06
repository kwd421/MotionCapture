"""Real OpenCV/FFmpeg, explicit synthetic model doubles. No native ML FPS claim."""
from __future__ import annotations

import json
import threading
from fractions import Fraction

import numpy as np
import pytest

from motioncapture import wholebody_boundary_probe as diagnostics
from motioncapture import wholebody_fast_input as fast
from motioncapture import wholebody_optimize_bench as bench
from motioncapture.recording import RecordedFrame, RecordedIdentity, inspect_recording
from motioncapture.wholebody_catalog import BenchmarkError
from motioncapture.wholebody_onnx import pose_tensor
from motioncapture.wholebody_stages import StageOwner, StagePipeline
from test_recording import tiny_vfr  # noqa: F401


def heads(index=192):
    x, y = np.zeros((1, 133, 384), np.float32), np.zeros((1, 133, 512), np.float32)
    x[:, :, index], y[:, :, 256] = 1., 1.
    return [x, y]


def boxes(count=1):
    result = np.zeros((1, 3549, 85), np.float32)
    for i in range(count):
        result[0, i*40, :6] = [2, 2, 0, 0, .99, .99]
    return [result]


class Model:
    def __init__(self, kind, pattern=(1,), sync=None, fail_at=None, close_fail=False):
        self.kind, self.pattern, self.sync = kind, pattern, sync
        self.fail_at, self.close_fail = fail_at, close_fail
        self.calls = 0
        self.ids = [threading.get_ident()]
        self.closed = False
        self.metadata = {"fixture": True, "kind": kind}

    def run(self, image):
        self.calls += 1
        self.ids.append(threading.get_ident())
        if self.calls == self.fail_at:
            raise BenchmarkError("injected_model_failure")
        if self.sync is not None:
            if self.kind == "detector" and self.calls == 2:
                self.sync.set()
            if self.kind == "pose" and self.calls == 1:
                assert self.sync.wait(3), "next detector did not overlap current pose"
        if self.kind == "detector":
            return boxes(self.pattern[(self.calls-1) % len(self.pattern)])
        return heads()

    def close(self):
        self.ids.append(threading.get_ident())
        self.closed = True
        if self.close_fail:
            raise BenchmarkError("injected_close_failure")


def source(n):
    for i in range(n):
        yield RecordedFrame(RecordedIdentity("s", "stream", i, i*1517, Fraction(1, 90000)),
                            np.full((48, 64, 3), i % 256, np.uint8), .1)


def make_pipeline(pattern=(1,), sync=None, **kwargs):
    instances = []
    def factory(kind):
        model = Model(kind, pattern, sync, **kwargs)
        instances.append(model)
        return model
    return StagePipeline(lambda: factory("detector"), lambda: factory("pose")), instances


def test_normalization_every_uint8_channel_is_bitwise_equal():
    crop = np.repeat(np.arange(256, dtype=np.uint8)[None, :, None], 3, axis=2)
    expected = ((crop-fast.MEAN)/fast.STD).transpose(2, 0, 1)[None].astype(np.float32)
    output = fast.normalize(crop)
    assert output.tobytes() == expected.tobytes()
    assert output.flags.c_contiguous and fast.TABLE.flags.writeable is False


@pytest.mark.parametrize("image_shape", [(48, 64), (720, 1280), (1080, 1920)])
@pytest.mark.parametrize("box", [[.01, -.0001, 63.991, 47.99], [10, 20, 30, 100],
                                 [-300.1, 7.25, 500.32, 1700.15]])
def test_real_affine_lookup_matches_original_including_ulp_boxes(image_shape, box):
    image = np.random.default_rng(46).integers(0, 256, (*image_shape, 3), np.uint8)
    before = fast.array_hash(image)
    for dtype in (np.float32, np.float64):
        bbox = np.array(box, dtype=dtype)
        for b in (bbox, np.nextafter(bbox, np.full(4, np.inf, dtype=dtype))):
            expected = pose_tensor(image, b, (192, 256))
            actual = fast.fast_pose_tensor(image, b, (192, 256))
            assert all(a.tobytes() == v.tobytes() for a, v in zip(expected, actual, strict=True))
            fast.check_recipe(image, b, (192, 256))
    assert fast.array_hash(image) == before


def test_previous_tensors_are_not_reused_and_missing_is_not_zero_filled():
    crop = np.full((20, 20, 3), 100, np.uint8)
    first = fast.normalize(crop)
    saved = first.copy()
    fast.normalize(np.zeros_like(crop))
    assert first.tobytes() == saved.tobytes() and not np.shares_memory(first, crop)
    with pytest.raises(BenchmarkError):
        fast.fast_pose_tensor(crop, np.array([0, 0, 0, 0]), (192, 256))
    with pytest.raises(BenchmarkError):
        fast.normalize(crop.astype(np.float32))


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("lookup", [False, True])
def test_stage_boundaries_exact_n_ownership_multiple_and_empty(overlap, lookup):
    pipeline, instances = make_pipeline(pattern=(1, 2, 0))
    frames = list(source(6))
    before = [fast.array_hash(f.image_bgr) for f in frames]
    with pipeline:
        outputs = list(pipeline.packets(frames, 6, overlap=overlap, fast=lookup, verify_eof=True))
    assert [len(p.people) for p in outputs] == [1, 2, 0, 1, 2, 0]
    assert [p.detected.frame.identity.sequence for p in outputs] == list(range(6))
    assert [fast.array_hash(f.image_bgr) for f in frames] == before
    assert instances[0].calls == instances[1].calls == 6
    assert pipeline.snapshot()["read_frames"] == 6
    assert all(model.closed and len(set(model.ids)) == 1 for model in instances)
    assert instances[0].ids[0] != instances[1].ids[0] != threading.get_ident()
    for packet in outputs:
        assert len(packet.per_person_ms) == len(packet.people)
        assert sum(packet.per_person_ms) == packet.times["pose_inference_ms"]
    assert outputs[2].times["pose_inference_ms"] == 0


def test_overlap_is_proven_with_events_not_sleep():
    pipeline, instances = make_pipeline(sync=threading.Event())
    with pipeline:
        assert len(list(pipeline.packets(source(2), 2, overlap=True, fast=False))) == 2
    assert all(m.closed for m in instances)


def test_each_stage_refuses_a_second_request_even_if_the_first_finished():
    owner = StageOwner(lambda: Model("detector"), "limit")
    try:
        owner.open()
        owner.submit(lambda session, submitted: 3)
        with pytest.raises(BenchmarkError, match="outstanding_limit"):
            owner.submit(lambda *args: 4)
        assert owner.receive() == 3
    finally:
        owner.close()


@pytest.mark.parametrize("overlap", [False, True])
def test_no_speculative_frame_n_plus_one_on_prefix(overlap):
    pipeline, instances = make_pipeline()
    advanced = []
    def frames():
        for frame in source(5):
            advanced.append(frame.identity.sequence)
            yield frame
    with pipeline:
        assert len(list(pipeline.packets(frames(), 2, overlap=overlap, fast=True))) == 2
    assert advanced == [0, 1] and all(m.calls == 2 for m in instances)


def test_incomplete_or_extra_sequence_does_not_pass():
    for frames in ([*source(1)], [*source(3)]):
        pipeline, _ = make_pipeline()
        with pytest.raises((BenchmarkError, RuntimeError)):
            with pipeline:
                list(pipeline.packets(frames, 2, overlap=True, fast=False, verify_eof=True))


def test_model_failure_preserves_primary_during_cleanup():
    pipeline, instances = make_pipeline(fail_at=2, close_fail=True)
    with pytest.raises(BenchmarkError, match="injected_model_failure"):
        with pipeline:
            list(pipeline.packets(source(5), 5, overlap=True, fast=False))
    assert all(m.closed for m in instances)
    assert pipeline.snapshot()["cleanup"] == {"detector": "failed", "pose": "failed"}


def test_factory_failure_never_opens_other_stage():
    def failure():
        raise BenchmarkError("constructor_test")
    pipeline = StagePipeline(failure, lambda: pytest.fail("must not create pose"))
    with pytest.raises(BenchmarkError, match="constructor_test"):
        with pipeline:
            pass
    assert pipeline.snapshot()["cleanup"]["detector"] == "unknown_after_constructor_failure"


@pytest.mark.parametrize("mode", list(bench.MODES))
def test_runner_real_vfr_all_frames_pts_hashes_and_cleanup(tiny_vfr, tmp_path, mode):
    args = bench.parser().parse_args([str(tiny_vfr), "--research-only", "--max-frames", "0",
                                     "--output", str(tmp_path/"test.json")])
    probe = inspect_recording(tiny_vfr)
    factories = (lambda: Model("detector", (1, 2, 0)), lambda: Model("pose"))
    row, bank = bench.run_pass(args, probe, mode, factories)
    assert row["status"] == "completed"
    assert row["all_frames"]["frames"] == len(probe.pts)
    assert tuple(bank.pts) == probe.pts
    assert row["pipeline"]["read_frames"] == row["pipeline"]["emitted_frames"] == 6
    assert row["pipeline"]["cleanup"] == {"detector": "owner_released", "pose": "owner_released"}
    assert row["last_completed"]["pts"] == probe.pts[-1]
    assert str(tiny_vfr) not in json.dumps(row)
    assert row["ground_truth_accuracy_verified"] is False


def test_runner_same_images_and_predictions_all_modes(tiny_vfr, tmp_path):
    args = bench.parser().parse_args(
        [str(tiny_vfr), "--research-only", "--output", str(tmp_path/"x")])
    probe = inspect_recording(tiny_vfr)
    reference, rows = None, []
    for mode in bench.MODES:
        row, new = bench.run_pass(args, probe, mode,
                                  (lambda: Model("detector"), lambda: Model("pose")), reference)
        reference = new or reference
        rows.append(row)
        assert row["status"] == "completed"
    for field in ("pixels_sha256", "detector_predictions_sha256", "predictions_sha256"):
        assert len({r[field] for r in rows}) == 1
    assert rows[1]["reference_hash_equal"] is rows[2]["reference_hash_equal"] is True


def test_failure_retains_completed_prefix_and_no_successful_fps(tiny_vfr, tmp_path):
    args = bench.parser().parse_args(
        [str(tiny_vfr), "--research-only", "--output", str(tmp_path/"x")])
    row, bank = bench.run_pass(args, inspect_recording(tiny_vfr), "sequential-lut",
                              (lambda: Model("detector", fail_at=3), lambda: Model("pose")))
    assert row["status"] == "failed" and bank is None
    assert row["unpaced_loop_fps"] is None and row["all_frames"]["frames"] == 2
    assert row["last_completed"]["sequence"] == 1


def report_for(probe, indexes):
    return {"source": {"sha256": probe.sha256}, "runs": [{"provider_disagreement": {
        "worst_frames": [{"sequence": i, "pts": probe.pts[i], "max_pixel_disagreement": 20.}
                          for i in indexes]}}]}


def test_diagnostic_frame_selection_bound_and_source_guard(tiny_vfr, tmp_path):
    probe = inspect_recording(tiny_vfr)
    path = tmp_path/"log.json"
    data = report_for(probe, [3])
    path.write_text(json.dumps(data))
    selected, _ = diagnostics.select_frames(path, probe, 6)
    assert selected == [2, 3, 4]
    data["source"]["sha256"] = "bad"
    path.write_text(json.dumps(data))
    with pytest.raises(BenchmarkError, match="source_hash"):
        diagnostics.select_frames(path, probe, 6)


def test_boundary_probe_same_tensor_control_and_no_raw_data():
    frame = next(source(1))
    row = diagnostics.observe_frame(frame, Model("detector"), Model("detector"),
                                    Model("pose"), (192, 256), .3)
    assert row["status"] == "compared"
    assert row["crop_pixels"]["byte_equal"] is row["pose_input"]["byte_equal"] is True
    assert row["same_tensor_a_repeat"]["x"]["byte_equal"] is True
    assert row["full_downstream"]["left_hand"]["max_source_pixels"] == 0
    assert "coordinates" not in json.dumps(row)


def test_probe_detects_same_input_nondeterministic_outputs():
    class Alternating(Model):
        def run(self, image):
            self.calls += 1
            return heads(192 if self.calls < 3 else 300)
    row = diagnostics.observe_frame(next(source(1)), Model("detector"), Model("detector"),
                                    Alternating("pose"), (192, 256), .3)
    assert row["pose_input"]["byte_equal"] is True
    assert row["same_tensor_a_repeat"]["x"]["argmax_changed_joints"] == 133


def test_probe_ambiguous_multiple_people_is_not_fake_zero_error():
    row = diagnostics.observe_frame(next(source(1)), Model("detector", (2,)),
                                    Model("detector", (2,)), Model("pose"), (192, 256), .3)
    assert row["status"] == "unmatched_zero_or_multiple_people"
    assert "full_downstream" not in row


def test_diagnostics_real_vfr_sequential_selection_and_preserved_failure(tiny_vfr, tmp_path):
    probe = inspect_recording(tiny_vfr)
    path = tmp_path/"report.json"
    path.write_text(json.dumps(report_for(probe, [3])))
    row = diagnostics.run_diagnostics(tiny_vfr, probe, path, 6,
        lambda: Model("detector"), lambda: Model("detector"), lambda: Model("pose"))
    assert row["status"] == "completed"
    assert [r["sequence"] for r in row["frames"]] == [2, 3, 4]
    assert all(r["status"] == "compared" for r in row["frames"])
    assert str(tiny_vfr) not in json.dumps(row)


def test_exact_recipe_guard_rejects_incompatible_local_owner(monkeypatch):
    def incompatible(*args):
        data = list(pose_tensor(*args))
        data[0][:] += .1
        return data
    monkeypatch.setattr(fast, "pose_tensor", incompatible)
    with pytest.raises(BenchmarkError, match="recipe_mismatch"):
        fast.check_recipe(np.zeros((48, 64, 3), np.uint8), np.array([0, 0, 64, 48.]), (192, 256))


def test_suite_serializes_six_arms_and_does_not_overwrite(tiny_vfr, tmp_path, monkeypatch):
    args = bench.parser().parse_args([str(tiny_vfr), "--research-only", "--max-frames", "0",
                                     "--output", str(tmp_path/"run.json")])
    monkeypatch.setattr(bench, "verify_asset", lambda root, key: (tmp_path/key, {"fixture": True}))
    seen = []
    def fake_native(args, probe, mode, factories, reference):
        seen.append(mode)
        return bench.run_pass(args, probe, mode,
                             (lambda: Model("detector"), lambda: Model("pose")), reference)
    assert bench.execute(args, pass_runner=fake_native) == 0
    data = json.loads(args.output.read_text())
    assert seen == [*bench.MODES, *reversed(bench.MODES)]
    assert data["all_pass_prediction_hashes_equal"] is True
    assert len(list(tmp_path.glob("*.arm-??.json"))) == 6
    with pytest.raises(FileExistsError):
        bench.execute(args, pass_runner=fake_native)


def test_stage_comparison_rejects_wrong_pts_and_keeps_missing_distinct():
    from motioncapture.wholebody_onnx import decode_pose
    from motioncapture.wholebody_stage_comparison import StageDifference, StageReference

    frame = next(source(1))
    person = decode_pose(heads(), (192, 256), np.array([50., 50.]),
                         np.array([100., 100.]), .3)
    reference = StageReference(1)
    reference.store(frame, [person])
    comparison = StageDifference()
    comparison.add(reference, frame, [])
    row = comparison.summary()
    assert row["parts"]["left_hand"]["max_pixels"] is None
    assert row["reference_only_point_observations"]["left_hand"] == 21
    assert row["coordinate_delta_lower_is_better"] is True
    invalid = RecordedFrame(RecordedIdentity("s", "stream", 0, 1, Fraction(1, 90000)),
                            frame.image_bgr, .1)
    with pytest.raises(BenchmarkError, match="identity_mismatch"):
        comparison.add(reference, invalid, [person])


def test_pipeline_does_not_hide_multi_person_comparison_as_perfect_match():
    from motioncapture.wholebody_onnx import decode_pose
    from motioncapture.wholebody_stage_comparison import StageDifference, StageReference

    frame = next(source(1))
    person = decode_pose(heads(), (192, 256), np.array([50., 50.]),
                         np.array([100., 100.]), .3)
    reference = StageReference(1)
    reference.store(frame, [person, person])
    comparison = StageDifference()
    comparison.add(reference, frame, [person, person])
    row = comparison.summary()
    assert row["ambiguous_multi_person_frames"] == 1
    assert row["parts"]["left_hand"]["matched_points"] == 0
    assert row["parts"]["left_hand"]["max_pixels"] is None
