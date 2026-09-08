from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from test_recording import tiny_vfr as tiny_vfr

from motioncapture import recording_bench as bench
from motioncapture.recording import inspect_recording


def test_summary_missing_and_zero_are_distinct():
    stats = bench.Samples()
    assert stats.summary(16.67)["mean_ms"] is None
    stats.add(0)
    assert stats.summary(16.67)["mean_ms"] == 0
    with pytest.raises(ValueError):
        stats.add(float("nan"))
    assert stats.summary(16.67)["samples"] == 1


def test_atomic_report_does_not_overwrite(tmp_path):
    out = tmp_path / "report.json"
    bench.write_report(out, {"status": "completed"})
    with pytest.raises(FileExistsError):
        bench.write_report(out, {"status": "different"})
    assert json.loads(out.read_text()) == {"status": "completed"}
    assert len(list(tmp_path.iterdir())) == 1


def test_full_decode_only_run_no_inference_or_extra_frames(tiny_vfr, tmp_path, monkeypatch):
    out = tmp_path / "decode.json"
    monkeypatch.setattr("sys.argv", ["bench", str(tiny_vfr), "--mode", "decode",
                        "--output", str(out), "--repeats", "2", "--verify-pixels"])
    assert bench.main() == 0
    result = json.loads(out.read_text())
    assert result["status"] == "completed"
    assert result["source"]["frames"] == 6
    assert len(result["runs"]) == 2
    assert result["runs"][0]["pixels_sha256"] == result["runs"][1]["pixels_sha256"]
    assert str(tiny_vfr) not in out.read_text()
    for run in result["runs"]:
        assert run["inference_executed"] is False
        assert run["resolved_detection_workloads"] is None
        assert run["all_frames"]["frames"] == 6
        assert run["all_frames"]["detections"] is None


def test_missing_mediapipe_does_not_fall_back_to_decode(tmp_path, monkeypatch):
    out = tmp_path / "failed.json"
    monkeypatch.setattr(bench, "_installed", lambda name: None)
    monkeypatch.setattr("sys.argv", ["bench", str(tmp_path / "not-read.mp4"),
                                    "--output", str(out)])
    assert bench.main() == 2
    data = json.loads(out.read_text())
    assert data["status"] == "failed" and data["runs"] == []
    assert data["error"]["reason"] == "pinned_mediapipe_unavailable"


def test_recording_bench_runs_all_tasks_path_with_explicit_test_tracker(
    tiny_vfr, tmp_path, monkeypatch,
):
    # This test exercises the benchmark orchestration, NOT native MediaPipe.
    @dataclass
    class Timings:
        total_ms: float = 2
        hands_ms: float = 1

    calls = []

    class Tracker:
        def __enter__(self):
            calls.append("open")
            return self

        def __exit__(self, *_):
            calls.append("close")

        def process_recorded(self, frame):
            calls.append(frame.identity.pts)
            result = SimpleNamespace(left_hand_landmarks=[1], right_hand_landmarks=[2],
                                     face_landmarks=[3], pose_landmarks=[4])
            return SimpleNamespace(identity=frame.identity, timings=Timings(), result=result)

    monkeypatch.setattr(bench, "_make_tracker", lambda *args: Tracker())
    args = bench._parser().parse_args([str(tiny_vfr), "--output", str(tmp_path / "unused"),
                                      "--preview", "none", "--warmup-frames", "2"])
    probe = inspect_recording(tiny_vfr)
    data = bench.run_pass(args, probe)
    assert calls[0] == "open" and calls[-1] == "close"
    assert calls[1:-1] == list(probe.pts)
    assert data["all_frames"]["frames"] == 6
    assert data["steady_after_initial_frames"]["frames"] == 4
    assert data["all_frames"]["detections"]["face"] == 6
    assert data["resolved_detection_workloads"]["resolved_hands=2,face=1"]["frames"] == 6
