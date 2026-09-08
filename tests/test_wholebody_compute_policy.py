"""Actual file runner/PTS/owners; explicit synthetic sessions, never neural speed tests."""
from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest
from test_recording import tiny_vfr as tiny_vfr
from test_wholebody_handoff import Session, frames

from motioncapture import wholebody_optimize_bench as bench
from motioncapture.recording import inspect_recording
from motioncapture.wholebody_catalog import BenchmarkError
from motioncapture.wholebody_onnx import Person2D, provider_plan
from motioncapture.wholebody_stage_comparison import StageDifference, StageReference


def arguments(video, directory, *extra):
    return bench.parser().parse_args([
        str(video), "--suite", "compute-policy", "--research-only",
        "--allow-cpu-partitions", "--max-frames", "0", "--decode-threads", "1",
        "--output", str(directory / "result.json"), *extra,
    ])


def fixture_runtime(monkeypatch, directory, *, fail_provider=None, change=False):
    instances = []
    class FixtureSession(Session):
        def __init__(self, path, shape, provider, *, allow_cpu, threads):
            kind = "detector" if shape == (1, 3, 416, 416) else "pose"
            if provider == fail_provider:
                raise BenchmarkError("injected_policy_unavailable")
            super().__init__(kind)
            self.provider = provider
            self.metadata = {"explicit_synthetic_fixture": True, "requested": provider,
                             "allow_cpu": allow_cpu, "threads": threads}
            instances.append(self)
        def run(self, tensor):
            output = super().run(tensor)
            if change and self.kind == "pose" and self.provider != "coreml-all":
                output[0] = np.roll(output[0], 1, axis=2)
            return output
    monkeypatch.setattr(bench, "OrtModel", FixtureSession)
    monkeypatch.setattr(bench, "verify_asset", lambda _, key: (directory/key, {"fixture": True}))
    return instances


def test_immutable_fixed_plan_and_real_provider_option_mapping(tmp_path, monkeypatch):
    args = arguments(tmp_path/"input", tmp_path)
    arms = bench.experiment_arms(args)
    pairs = [(a.detector_provider, a.pose_provider) for a in arms]
    assert pairs == [
        ("coreml-all", "coreml-all"), ("coreml-gpu", "coreml-ane"),
        ("coreml-ane", "coreml-gpu"), ("coreml-ane", "coreml-gpu"),
        ("coreml-gpu", "coreml-ane"), ("coreml-all", "coreml-all"),
    ]
    assert all(a.mode == "source-pts-ready-cvlut" for a in arms)
    with pytest.raises(FrozenInstanceError):
        arms[0].pose_provider = "cpu"
    monkeypatch.setattr("motioncapture.wholebody_onnx.platform.system", lambda: "Darwin")
    monkeypatch.setattr("motioncapture.wholebody_onnx.platform.machine", lambda: "arm64")
    for name, units in [("coreml-all", "ALL"), ("coreml-gpu", "CPUAndGPU"),
                        ("coreml-ane", "CPUAndNeuralEngine")]:
        providers, _ = provider_plan(name, ["CoreMLExecutionProvider"], allow_cpu=True)
        assert providers[0][1]["MLComputeUnits"] == units
        assert providers[-1] == "CPUExecutionProvider"


@pytest.mark.parametrize("override", [
    ["--detector-provider", "cpu"], ["--pose-provider", "coreml-gpu"],
    ["--diagnose-from", "unused.json"],
])
def test_conflicting_overrides_fail_before_model_creation(
    tiny_vfr, tmp_path, monkeypatch, override,
):
    args = arguments(tiny_vfr, tmp_path, *override)
    instances = fixture_runtime(monkeypatch, tmp_path)
    assert bench.execute(args) == 2
    result = json.loads(args.output.read_text())
    assert result["error"]["code"] == "compute_policy_requires_fixed_plan_without_diagnostics"
    assert result["error"]["phase"] == "configuration"
    assert not instances and not result["runs"]


def test_real_vfr_six_arms_factories_manifest_cleanup_and_hashes(tiny_vfr, tmp_path, monkeypatch):
    args = arguments(tiny_vfr, tmp_path)
    instances = fixture_runtime(monkeypatch, tmp_path)
    assert bench.execute(args) == 0
    result = json.loads(args.output.read_text())
    rows = result["runs"]
    n = len(inspect_recording(tiny_vfr).pts)
    assert len(rows) == 6 and len(instances) == 12  # no auxiliary session
    assert args.pose_provider == args.detector_provider == "coreml-all"  # no mutation
    assert result["configuration"]["detector_provider"] == "per_arm_plan"
    assert result["all_pass_prediction_hashes_equal"] is True
    assert result["all_pass_detector_hashes_equal"] is True
    assert result["all_pass_pixel_hashes_equal"] is True
    assert result["live_60fps_verified"] is False
    assert result["accuracy_verified"] is False
    for index, row in enumerate(rows):
        arm = result["configuration"]["execution_arm_plan"][index]
        started = json.loads(
            args.output.with_name(f"result.arm-{index+1:02d}.started.json").read_text())
        assert all(started[k] == arm[k] for k in arm)
        assert row["execution_arm"] == arm
        assert row["backend"]["detector"]["requested"] == arm["detector_provider"]
        assert row["backend"]["pose"]["requested"] == arm["pose_provider"]
        assert row["same_provider_in_all_arms"] is False
        assert row["pose_lanes"] == 1 and row["source_pacing"] == "original_pts"
        assert row["all_frames"]["frames"] == row["replay_ages"]["frames"] == n
        assert row["pipeline"]["source_pacing"]["frames_skipped"] == 0
        assert row["pipeline"]["source_pacing"]["clock_rebases"] == 0
        assert row["unpaced_loop_fps"] is None
        assert row["host_process_cost"]["process_cpu_s"] >= 0
        assert row["host_process_cost"]["gpu_ane_time_measured"] is False
        assert set(row["pipeline"]["cleanup"].values()) == {"owner_released"}
        if index:
            comparison = row["provider_disagreement"]
            assert "cross-policy" in comparison["reference"]
            assert comparison["per_frame_predictions"]["checked_frames"] == n
            assert comparison["per_frame_detector_boxes"]["checked_frames"] == n
            assert comparison["per_frame_detector_boxes"]["changed_frames"] == 0
    assert all(s.calls == n and s.closed_on == s.created_on for s in instances)
    with pytest.raises(FileExistsError):
        bench.execute(args)


def test_failed_requested_policy_stops_without_retry_or_other_arm(tiny_vfr, tmp_path, monkeypatch):
    args = arguments(tiny_vfr, tmp_path)
    instances = fixture_runtime(monkeypatch, tmp_path, fail_provider="coreml-ane")
    assert bench.execute(args) == 2
    d = json.loads(args.output.read_text())
    assert len(d["runs"]) == 2
    assert d["runs"][0]["status"] == "completed"
    failed = d["runs"][1]
    assert failed["error"]["code"] == "injected_policy_unavailable"
    assert failed["error"]["phase"] == "session_setup"
    assert failed["paced_loop_fps"] is None
    assert failed["host_process_cost"]["process_cpu_s"] is None
    assert [s.provider for s in instances] == ["coreml-all", "coreml-all", "coreml-gpu"]
    assert all(s.closed_on == s.created_on for s in instances)
    assert not args.output.with_name("result.arm-03.started.json").exists()


def test_changed_numeric_outputs_are_reported_not_promoted(tiny_vfr, tmp_path, monkeypatch):
    args = arguments(tiny_vfr, tmp_path)
    fixture_runtime(monkeypatch, tmp_path, change=True)
    assert bench.execute(args) == 0  # execution completion is NOT quality acceptance
    d = json.loads(args.output.read_text())
    assert d["all_pass_prediction_hashes_equal"] is False
    assert d["all_pass_detector_hashes_equal"] is True
    assert d["accuracy_verified"] is False
    for row in d["runs"][1:5]:
        p = row["provider_disagreement"]
        assert p["per_frame_predictions"]["changed_frames"] > 0
        assert p["per_frame_detector_boxes"]["changed_frames"] == 0
        assert p["parts"]["left_hand"]["max_pixels"] > 0


def person(x=0.):
    return Person2D(np.full((133, 2), x), np.ones(133), np.ones(133, bool))


def test_sum_and_elapsed_stratification_remain_distinct():
    people = [person(), person()]
    packet = SimpleNamespace(
        people=people, detected=SimpleNamespace(boxes=np.zeros((2, 4), np.float32)),
        per_person_ms=(10., 12.), times={"pose_inference_ms": 22., "pose_stage_ms": 13.,
                                       "detector_stage_ms": 8.},
    )
    stats = bench.Stats()
    stats.add(packet)
    group = stats.summary()["pose_by_person_count"]["2"]
    assert group["frame_pose_ms"]["mean_ms"] == 22
    assert group["elapsed_pose_stage_ms"]["mean_ms"] == 13
    assert group["elapsed_detector_stage_ms"]["mean_ms"] == 8
    assert group["per_person_ms"]["samples"] == 2
    assert group["per_person_ms"]["mean_ms"] == 11


def test_all_slot_box_hashes_detect_order_and_count_changes():
    incoming = list(frames(3))
    bank = StageReference(3, frame_hashes=True)
    observations = [[], [person(), person(1)], [person()]]
    boxes = [np.empty((0, 4), np.float32), np.array([[0,0,3,3],[5,5,8,8]], np.float32),
             np.array([[0,0,3,3]], np.float32)]
    for frame, people, box in zip(incoming, observations, boxes, strict=True):
        bank.store(frame, people, boxes=box)
    cmp = StageDifference()
    cmp.add(bank, incoming[0], [], boxes=boxes[0])
    cmp.add(bank, incoming[1], observations[1], boxes=boxes[1][::-1])
    cmp.add(bank, incoming[2], [], boxes=np.empty((0, 4), np.float32))
    result = cmp.summary()
    b = result["per_frame_detector_boxes"]
    assert (b["checked_frames"], b["changed_frames"], b["count_mismatch_frames"]) == (3, 2, 1)
    assert b["first_changes"][0]["sequence"] == 1
    assert result["ambiguous_multi_person_frames"] == 1
    assert result["per_frame_predictions"]["multi_person_frames_changed"] == 0
    assert result["per_frame_predictions"]["changed_frames"] == 1


def test_old_reference_without_boxes_does_not_claim_box_agreement():
    frame = next(iter(frames(1)))
    bank = StageReference(1, frame_hashes=True)
    bank.store(frame, [person()])
    cmp = StageDifference()
    cmp.add(bank, frame, [person()], boxes=np.zeros((1, 4), np.float32))
    assert cmp.summary()["per_frame_detector_boxes"]["checked_frames"] == 0


@pytest.mark.parametrize("bad", [np.zeros((1, 4), np.float64), np.zeros((1, 3), np.float32),
                                 np.full((1, 4), np.nan, np.float32)])
def test_invalid_box_contract_fails(bad):
    bank = StageReference(1, frame_hashes=True)
    with pytest.raises(BenchmarkError, match="invalid_detector_box_comparison"):
        bank.store(next(iter(frames(1))), [person()], boxes=bad)


def test_process_cpu_clock_is_not_loop_wall_or_gpu(tiny_vfr, tmp_path, monkeypatch):
    args = arguments(tiny_vfr, tmp_path)
    values = iter((2_000_000_000, 5_000_000_000))
    monkeypatch.setattr(bench.time, "process_time_ns", lambda: next(values))
    def create(kind):
        session = Session(kind)
        session.metadata = {"requested": "coreml-all", "explicit_synthetic_fixture": True}
        return session
    row, _ = bench.run_pass(args, inspect_recording(tiny_vfr), "source-pts-ready-cvlut",
                            (lambda: create("detector"), lambda: create("pose")))
    assert row["status"] == "completed"
    metric = row["host_process_cost"]
    assert metric["process_cpu_s"] == 3
    assert metric["mean_cpu_cores_during_loop"] == pytest.approx(3/row["loop_s"])
    assert metric["gpu_ane_time_measured"] is False


def test_original_dualpose_suite_still_hashes_every_frame(tiny_vfr, tmp_path, monkeypatch):
    args = arguments(tiny_vfr, tmp_path)
    args.suite = "pose-parallel"
    fixture_runtime(monkeypatch, tmp_path)
    assert bench.execute(args) == 0
    d = json.loads(args.output.read_text())
    assert [r["pose_lanes"] for r in d["runs"]] == [1, 2, 2, 1]
    assert d["all_pass_prediction_hashes_equal"] is True
    assert all(r["same_provider_in_all_arms"] for r in d["runs"])


def test_wrong_session_policy_is_rejected_before_measured_frames(tiny_vfr, tmp_path):
    args = arguments(tiny_vfr, tmp_path)
    sessions = []
    def create(kind):
        s = Session(kind)
        s.metadata = {"requested": "cpu", "explicit_synthetic_fixture": True}
        sessions.append(s)
        return s
    row, bank = bench.run_pass(args, inspect_recording(tiny_vfr), "source-pts-ready-cvlut",
                               (lambda: create("detector"), lambda: create("pose")))
    assert row["status"] == "failed" and bank is None
    assert row["error"]["code"] == "compute_policy_session_metadata_mismatch"
    assert row["all_frames"]["frames"] == 0
    assert all(s.calls == 0 and s.closed_on == s.created_on for s in sessions)
