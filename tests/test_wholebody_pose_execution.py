"""Real VFR/worker path with explicit model fixtures; not native acceleration tests."""
from __future__ import annotations

import copy
import json
from dataclasses import FrozenInstanceError
from fractions import Fraction

import numpy as np
import pytest
from test_recording import tiny_vfr as tiny_vfr
from test_wholebody_handoff import Session

from motioncapture import wholebody_optimize_bench as bench
from motioncapture.recording import RecordedIdentity, inspect_recording
from motioncapture.wholebody_catalog import BenchmarkError
from motioncapture.wholebody_replay import Release, ReplayAges


def arguments(video, directory, *extra):
    return bench.parser().parse_args([
        str(video), "--suite", "pose-execution", "--research-only",
        "--allow-cpu-partitions", "--max-frames", "0", "--decode-threads", "1",
        "--output", str(directory / "result.json"), *extra,
    ])


def install(monkeypatch, directory, *, fault=None, multi=False):
    instances = []

    class Fixture(Session):
        def __init__(self, path, shape, provider, *, allow_cpu, threads):
            kind = "detector" if shape == (1, 3, 416, 416) else "pose"
            if fault == "constructor" and kind == "pose" and threads == 1:
                raise BenchmarkError("injected_small_pool_failure")
            super().__init__(kind)
            self.provider, self.threads = provider, threads
            self.metadata = {"explicit_neural_fixture": True, "requested": provider,
                             "intra_op_threads": threads}
            if kind == "pose" and threads == 1:
                if fault == "thread_metadata":
                    self.metadata["intra_op_threads"] = 4
                if fault == "provider_metadata":
                    self.metadata["requested"] = "coreml-all"
            instances.append(self)

        def run(self, tensor):
            result = super().run(tensor)
            if multi and self.kind == "detector":
                if self.calls % 3 == 1:
                    result[0].fill(0)
                elif self.calls % 3 == 2:
                    result[0][0, 3539, :6] = [10, 10, 0, 0, .98, .98]
            if fault == "changed_output" and self.kind == "pose" and self.threads == 1:
                result[0] = np.roll(result[0], 1, axis=2)
            return result

    monkeypatch.setattr(bench, "OrtModel", Fixture)
    monkeypatch.setattr(bench, "verify_asset", lambda _, key: (directory / key, {"fixture": True}))
    return instances


def test_plan_is_orthogonal_immutable_and_inherits_legacy_budget(tmp_path):
    args = arguments(tmp_path / "video", tmp_path)
    arms = bench.experiment_arms(args)
    actual = [(a.detector_provider, a.pose_provider, a.pose_threads(args.ort_threads))
              for a in arms]
    a, b, c = ("coreml-all", "coreml-all", 4), ("coreml-all", "coreml-ane", 4), (
        "coreml-all", "coreml-ane", 1)
    assert actual == [a, b, c, c, b, a]
    assert all(a.mode == "source-pts-ready-cvlut" for a in arms)
    with pytest.raises(FrozenInstanceError):
        arms[0].pose_intra_op_threads = 2
    legacy = bench.ExecutionArm("overlap-lut", "cpu", "cpu")
    assert legacy.pose_threads(3) == 3
    assert "pose_intra_op_threads" not in legacy.record()
    assert arms[2].record()["pose_intra_op_threads"] == 1


@pytest.mark.parametrize("value", [True, 0, 65])
def test_invalid_pose_budget_is_not_replaced(value):
    with pytest.raises(BenchmarkError, match="pose_thread_budget"):
        bench.ExecutionArm(
            "source-pts-ready-cvlut", "coreml-all", "coreml-ane", value).pose_threads(4)


@pytest.mark.parametrize("extra", [
    ["--detector-provider", "cpu"], ["--pose-provider", "coreml-ane"],
    ["--ort-threads", "2"], ["--diagnose-from", "unused.json"],
])
def test_conflicting_overrides_fail_before_assets_and_sessions(
    tiny_vfr, tmp_path, monkeypatch, extra,
):
    args = arguments(tiny_vfr, tmp_path, *extra)
    instances = install(monkeypatch, tmp_path)
    assert bench.execute(args) == 2
    result = json.loads(args.output.read_text())
    assert result["error"]["code"] == "pose_execution_requires_fixed_plan_without_diagnostics"
    assert result["error"]["phase"] == "configuration"
    assert not instances and not result["runs"]


def test_real_vfr_six_arms_session_options_counts_hashes_and_trace(tiny_vfr, tmp_path, monkeypatch):
    args = arguments(tiny_vfr, tmp_path)
    instances = install(monkeypatch, tmp_path, multi=True)
    assert bench.execute(args) == 0
    result = json.loads(args.output.read_text())
    n = len(inspect_recording(tiny_vfr).pts)
    assert len(instances) == 12  # one detector and one pose model in EACH fresh arm
    assert [s.threads for s in instances[::2]] == [4] * 6
    assert [s.threads for s in instances[1::2]] == [4, 4, 1, 1, 4, 4]
    assert [s.provider for s in instances[::2]] == ["coreml-all"] * 6
    assert [s.provider for s in instances[1::2]] == [
        "coreml-all", "coreml-ane", "coreml-ane", "coreml-ane", "coreml-ane", "coreml-all"]
    assert args.ort_threads == 4 and not hasattr(args, "pose_intra_op_threads")
    assert result["configuration"]["stage_trace_source_window_seconds"] == 1
    assert result["all_pass_prediction_hashes_equal"]
    assert result["all_pass_pixel_hashes_equal"] and result["all_pass_detector_hashes_equal"]
    for i, row in enumerate(result["runs"]):
        arm = result["configuration"]["execution_arm_plan"][i]
        start = json.loads(args.output.with_name(f"result.arm-{i+1:02d}.started.json").read_text())
        assert row["execution_arm"] == arm
        assert all(start[key] == value for key, value in arm.items())
        assert row["backend"]["pose"]["intra_op_threads"] == arm["pose_intra_op_threads"]
        assert row["all_frames"]["frames"] == n
        assert row["all_frames"]["person_count_distribution"]["2"] > 0
        assert row["pipeline"]["pose_lanes"] == 1
        assert row["pipeline"]["dependency_pose_handoff_enabled"] is False
        assert row["pipeline"]["source_pacing"]["frames_skipped"] == 0
        assert row["pipeline"]["source_pacing"]["clock_rebases"] == 0
        assert set(row["pipeline"]["cleanup"].values()) == {"owner_released"}
        trace = row["replay_ages"]["stage_trace"]
        assert trace["coverage"] == "complete" and trace["frames"] == n
        assert trace["scalar_sample_count"] == trace["maximum_scalar_samples"]
        assert sum(w["frames"] for w in trace["source_windows"]) == n
        assert row["all_frames"]["stages"]["replay_observer_ms"]["samples"] == n
        assert "stage_times" in row["replay_ages"]["worst_source_ages"][0]
        assert row["live_60fps_verified"] is False
        if i:
            assert row["provider_disagreement"]["per_frame_predictions"]["changed_frames"] == 0
            assert row["provider_disagreement"]["per_frame_predictions"][
                "multi_person_frames_checked"] > 0
    assert all(s.closed_on == s.created_on for s in instances)
    with pytest.raises(FileExistsError):
        bench.execute(args)


@pytest.mark.parametrize("fault,code", [
    ("constructor", "injected_small_pool_failure"),
    ("thread_metadata", "pose_execution_thread_metadata_mismatch"),
    ("provider_metadata", "compute_policy_session_metadata_mismatch"),
])
def test_selected_candidate_failure_is_terminal_no_retry(
    tiny_vfr, tmp_path, monkeypatch, fault, code,
):
    args = arguments(tiny_vfr, tmp_path)
    instances = install(monkeypatch, tmp_path, fault=fault)
    assert bench.execute(args) == 2
    result = json.loads(args.output.read_text())
    assert len(result["runs"]) == 3
    failed = result["runs"][-1]
    assert failed["error"]["code"] == code
    assert failed["error"]["phase"] == "session_setup"
    assert failed["all_frames"]["frames"] == 0 and failed["paced_loop_fps"] is None
    assert failed["execution_arm"]["pose_intra_op_threads"] == 1
    assert all(r["status"] == "completed" for r in result["runs"][:2])
    assert not args.output.with_name("result.arm-04.started.json").exists()
    assert all(s.closed_on == s.created_on for s in instances)


def test_output_change_stays_visible_without_changing_detector(tiny_vfr, tmp_path, monkeypatch):
    args = arguments(tiny_vfr, tmp_path)
    install(monkeypatch, tmp_path, fault="changed_output", multi=True)
    assert bench.execute(args) == 0
    result = json.loads(args.output.read_text())
    assert not result["all_pass_prediction_hashes_equal"]
    assert result["all_pass_detector_hashes_equal"]
    assert not result["accuracy_verified"] and not result["live_60fps_verified"]
    for row in result["runs"][2:4]:
        cmp = row["provider_disagreement"]
        assert cmp["per_frame_predictions"]["changed_frames"] > 0
        assert cmp["per_frame_predictions"]["multi_person_frames_changed"] > 0
        assert cmp["per_frame_detector_boxes"]["changed_frames"] == 0


def identity(i, pts):
    return RecordedIdentity("fixture", "stream", i, pts, Fraction(1, 1000))


def stage_values():
    return dict(zip(ReplayAges.TRACE_FIELDS, [10., 3., 2., 4., 1., 2., 2., 3., .5], strict=True))


def observe(ages, i, pts, *, people=1):
    due = 100_000_000 + pts * 1_000_000
    ages.add(identity(i, pts), Release(due, due + 10_000_000, 0), due + 25_000_000,
             people=people, pose_stage_ms=4., stage_times=stage_values())


def test_source_second_bins_and_correlated_path_not_nominal_fps():
    ages = ReplayAges(4, trace_stages=True)
    for i, pts in enumerate([7, 1006, 1007, 1024]):
        observe(ages, i, pts, people=i % 3)
    summary = ages.summary()
    trace = summary["stage_trace"]
    assert trace["frames"] == 4 and trace["source_window_seconds"] == 1
    a, b = trace["source_windows"]
    assert (a["start_s"], a["frames"], b["start_s"], b["frames"]) == (0, 2, 1, 2)
    assert a["metrics"]["source_age_ms"]["mean_ms"] == 25
    assert b["metrics"]["unattributed_observer_gap_ms"]["mean_ms"] == 3
    assert a["person_count_distribution"] == {"0": 1, "1": 1}
    assert summary["source_age_ms"]["mean_ms"] == 25
    assert summary["source_windows"][0]["observed_frames"] == 4  # old 10s bins unchanged
    assert all(r["stage_times"]["pose_inference_ms"] == 3 for r in summary["worst_source_ages"])


@pytest.mark.parametrize("defect", ["missing", "nan", "negative", "path", "pose_mismatch"])
def test_invalid_trace_preserves_previous_good_statistics(defect):
    ages = ReplayAges(3, trace_stages=True)
    observe(ages, 0, 0)
    before = copy.deepcopy(ages.summary())
    times = stage_values()
    if defect == "missing":
        del times["decode_read_ms"]
    elif defect == "nan":
        times["pose_stage_ms"] = float("nan")
    elif defect == "negative":
        times["verification_ms"] = -1
    elif defect == "path":
        times["detector_to_pose_wait_ms"] = 100
    else:
        times["pose_stage_ms"] = 5
    with pytest.raises(BenchmarkError):
        ages.add(identity(1, 17), Release(117_000_000, 127_000_000, 0), 142_000_000,
                 people=1, pose_stage_ms=4., stage_times=times)
    assert ages.summary() == before


def test_trace_is_bounded_partial_and_never_silently_enabled():
    selected = ReplayAges(25, trace_stages=True)
    assert selected.summary()["stage_trace"]["status"] == "no_samples"
    for i in range(20):
        observe(selected, i, 17 * i)
    s = selected.summary()
    assert s["stage_trace"]["coverage"] == "partial"
    assert s["stage_trace"]["scalar_sample_count"] < s["stage_trace"]["maximum_scalar_samples"]
    assert len(s["worst_source_ages"]) == 16
    legacy = ReplayAges(1)
    assert legacy.summary()["stage_trace"]["status"] == "not_selected"
    with pytest.raises(BenchmarkError, match="not_selected"):
        observe(legacy, 0, 0)
    assert legacy.frames == 0
