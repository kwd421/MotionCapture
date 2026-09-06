"""Owner/callback integration with encoded VFR and explicit model test doubles."""
from types import SimpleNamespace as NS

import pytest

from motioncapture import hand_acceleration_bench as experiment
from motioncapture import recording_bench as bench
from motioncapture.recording import inspect_recording
from test_recording import tiny_vfr  # noqa: F401
from test_recorded_tracker_clock import tracker  # noqa: F401


def test_explicit_prefix_stops_at_exact_requested_count(tiny_vfr, tmp_path):
    args = bench._parser().parse_args([str(tiny_vfr), "--mode", "decode", "--preview", "none",
                                      "--output", str(tmp_path/"unused")])
    probe = inspect_recording(tiny_vfr)
    seen = []
    result = bench.run_pass(args, probe, result_observer=lambda f, r: seen.append(f.identity.pts),
                            frame_limit=3)
    assert seen == list(probe.pts[:3])
    assert result["scope"] == "explicit_prefix" and result["requested_frames"] == 3
    assert result["all_frames"]["frames"] == 3 and result["decoder_cleanup_complete"]
    assert result["observer_overhead_in_loop_fps"]
    assert result["all_frames"]["stages"]["comparison_observer_ms"]["samples"] == 3


def test_observer_failure_is_not_a_completed_run(tiny_vfr, tmp_path):
    args = bench._parser().parse_args([str(tiny_vfr), "--mode", "decode", "--preview", "none",
                                      "--output", str(tmp_path/"unused")])
    def fail(*_):
        raise ValueError("reference_failed")
    with pytest.raises(ValueError, match="reference_failed"):
        bench.run_pass(args, inspect_recording(tiny_vfr), result_observer=fail)


def test_model_cleanup_continues_after_external_hand_error(tracker):
    t, _ = tracker
    calls = []
    def close_hand():
        calls.append("hand")
        raise RuntimeError("browser_disconnected")
    t._pose = NS(close=lambda: calls.append("pose"))
    t._face = NS(close=lambda: calls.append("face"))
    t._hands = NS(close=close_hand)
    with pytest.raises(RuntimeError, match="browser_disconnected"):
        t.close()
    assert calls == ["face", "hand", "pose"]
    assert t._pose is t._hands is t._face is None


def test_native_unavailable_is_explicit_before_server_or_file_access(tmp_path, monkeypatch):
    import json
    args = experiment._parser().parse_args([str(tmp_path/"not-opened.mp4"),
                                            "--output", str(tmp_path/"failed.json")])
    monkeypatch.setattr(bench, "_installed", lambda _: None)
    assert experiment.run(args) == 2
    data = json.loads(args.output.read_text())
    assert data["status"] == "failed" and data["stage"] == "preflight"
    assert data["runs"] == [] and data["source"] is None
    assert str(tmp_path) not in args.output.read_text()


def test_existing_report_is_not_overwritten(tmp_path):
    args = experiment._parser().parse_args(["unused.mp4", "--output", str(tmp_path/"keep.json")])
    args.output.write_text("keep")
    with pytest.raises(FileExistsError):
        experiment.run(args)
    assert args.output.read_text() == "keep"


@pytest.mark.parametrize("limit", [0, 3])
def test_six_pass_runner_uses_fresh_models_and_reports_disagreement(
    tiny_vfr, tmp_path, monkeypatch, limit,
):
    """Real VFR decode, explicitly synthetic inference/browser orchestration."""
    import json
    import sys
    import types

    from motioncapture.contracts import Landmark, LandmarkResult, LandmarkTimings

    created, closed, source_times = [], [], []
    model = tmp_path / "synthetic.task"
    model.write_bytes(b"explicit test-only asset")
    assets_module = types.ModuleType("motioncapture.model_assets")
    assets_module.MODEL_ASSETS = [NS(key="hands", sha256="test-only")]
    assets_module.require_models = lambda _: {"hands": model}
    monkeypatch.setitem(sys.modules, "motioncapture.model_assets", assets_module)
    fake_mp = types.ModuleType("mediapipe")
    fake_mp.ImageFormat = NS(SRGB=1)
    fake_mp.Image = lambda **kwargs: kwargs["data"]
    monkeypatch.setitem(sys.modules, "mediapipe", fake_mp)

    class Lab:
        url, hidden_events, visible = "synthetic://test-only", 0, True
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
        def wait_ready(self):
            pass

    class Task:
        def __init__(self, lab, delegate):
            self.delegate = delegate
        def detect_for_video(self, image, timestamp):
            assert timestamp == 0  # The single outside-pass preflight.
        def close(self):
            closed.append(self.delegate)
        def summary(self, budget):
            return {"synthetic_test": True, "delegate": self.delegate}

    class Tracker:
        def __init__(self, model_dir, task_scheduling, hand_task_factory=None):
            assert task_scheduling == "parallel"
            self.hand = hand_task_factory(model) if hand_task_factory else None
            self.times = []
            created.append(self)
        def __enter__(self):
            return self
        def __exit__(self, *_):
            if self.hand:
                self.hand.close()
        def process_recorded(self, frame):
            self.times.append(frame.identity.pts)
            source_times.append(frame.identity.pts)
            x = .3 if self.hand and self.hand.delegate == "GPU" else .2
            points = tuple(Landmark(x, .2, .0) for _ in range(21))
            result = LandmarkResult((), (), points, points, (), (), (), ())
            return NS(identity=frame.identity, result=result,
                      timings=LandmarkTimings(0, 1, 1, 1, 1, 0, 1))

    module = types.ModuleType("motioncapture.landmarkers")
    module.MediaPipeLandmarkTracker = Tracker
    monkeypatch.setitem(sys.modules, "motioncapture.landmarkers", module)
    monkeypatch.setattr(bench, "_installed", lambda _: "0.10.31")
    monkeypatch.setattr(experiment, "BrowserHandLab", Lab)
    monkeypatch.setattr(experiment, "BrowserHandTask", Task)
    monkeypatch.setattr(experiment, "collect_assets", lambda *_: ({}, {"test_only": True}))
    args = experiment._parser().parse_args([
        str(tiny_vfr), "--output", str(tmp_path/"result.json"), "--preview", "none",
        "--include-web-cpu", "--max-frames", str(limit),
    ])
    assert experiment.run(args) == 0
    report = json.loads(args.output.read_text())
    assert report["status"] == "completed" and report["native_reference_repeat_equal"]
    assert [r["backend"] for r in report["runs"]] == [
        "native", "web_cpu", "web_gpu", "web_gpu", "web_cpu", "native",
    ]
    pts = inspect_recording(tiny_vfr).pts
    expected = list(pts[:limit] if limit else pts)
    assert len(created) == 6 and all(t.times == expected for t in created)
    assert len(source_times) == len(expected) * 6
    assert closed == ["GPU", "CPU", "GPU", "GPU", "CPU"]
    for index in (2, 3):
        delta = report["runs"][index]["hand_comparison"]
        assert delta["image_xy_displacement_pixels"]["mean"] > 0
        assert delta["quality_verdict"] == "requires_review"
    assert report["accuracy_verified"] is False
    assert str(tmp_path) not in args.output.read_text()
