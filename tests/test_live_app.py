"""Exercise the real optimized application with explicit test-only sources."""
from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest
from test_live_pipeline import FakeCapture, FakeTracker

from motioncapture import live_app, preview
from motioncapture.errors import InferenceError


@pytest.fixture
def harness(monkeypatch, tmp_path):
    capture, tracker = FakeCapture(), FakeTracker()
    monkeypatch.setattr(live_app, "CaptureRuntime", lambda *args, **kwargs: capture)
    monkeypatch.setattr(live_app, "_new_tracker", lambda *args: tracker)
    monkeypatch.setattr(preview, "_load_connections", lambda: (
        ((0, 1),), ((0, 1),), ((0, 1),),
    ))
    composer = preview.DisplayPreview()
    monkeypatch.setattr(
        live_app, "_preview_tools", lambda mode: (composer.compose, SimpleNamespace),
    )
    return capture, tracker, tmp_path


@pytest.mark.parametrize("count", [1, 2, 7])
@pytest.mark.parametrize("schedule", ["overlap", "sequential"])
def test_headless_runs_exactly_n_no_speculative_n_plus_one(harness, count, schedule):
    capture, tracker, tmp_path = harness
    args = live_app._parser().parse_args([
        "--headless-frames", str(count), "--session-dir", str(tmp_path),
        "--pipeline-scheduling", schedule,
    ])
    assert live_app.run(args) == 0
    assert len(tracker.processed) == count and capture.frames == count
    files = list(tmp_path.iterdir())
    assert len(files) == 1 and files[0].suffix == ".json"
    payload = json.loads(files[0].read_text())
    assert payload["frames"] == count and payload["terminal_status"] == "completed"
    state = payload["performance"]["pipeline"]
    assert state["delivered"] == count and state["inferences_completed"] == count
    assert state["pending"] == state["discarded"] == 0
    assert state["cleanup_complete"] is True
    assert payload["performance"]["additional_stages"]["event_pump"]["samples"] == 0
    assert payload["privacy"]["raw_frames_recorded"] is False


def test_gui_is_main_thread_and_next_inference_overlaps_display(harness, monkeypatch):
    _, tracker, tmp_path = harness
    main_id = threading.get_ident()
    calls = []

    def imshow(*args):
        assert threading.get_ident() == main_id
        calls.append("show")
        if len(calls) == 1:
            assert tracker.second_started.wait(1), "next inference was held behind GUI work"

    for name in ("namedWindow", "resizeWindow", "destroyAllWindows"):
        monkeypatch.setattr(live_app.cv2, name, lambda *args: None)
    monkeypatch.setattr(live_app.cv2, "imshow", imshow)
    monkeypatch.setattr(live_app.cv2, "waitKey", lambda *args: -1)
    args = live_app._parser().parse_args(["--max-frames", "2", "--session-dir", str(tmp_path)])
    assert live_app.run(args) == 0
    assert calls == ["show", "show"]
    data = json.loads(next(tmp_path.glob("*.json")).read_text())
    timings = data["performance"]["additional_stages"]
    assert timings["presentation_submit"]["samples"] == 2
    assert timings["event_pump"]["samples"] == 2
    assert data["performance"]["sensor_to_photon_measured"] is False


def test_failed_tracker_still_writes_failure_manifest(harness):
    capture, tracker, tmp_path = harness
    tracker.failure = RuntimeError("expected test failure")
    args = live_app._parser().parse_args(["--headless-frames", "2", "--session-dir", str(tmp_path)])
    with pytest.raises(InferenceError, match="expected test failure"):
        live_app.run(args)
    data = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert data["terminal_status"] == "failed"
    assert data["performance"]["pipeline"]["state"] == "failed"
    assert data["frames"] == 0 and capture.cleaned


def test_q_can_abandon_only_one_in_flight_result(harness, monkeypatch):
    _, tracker, tmp_path = harness
    for name in ("namedWindow", "resizeWindow", "destroyAllWindows"):
        monkeypatch.setattr(live_app.cv2, name, lambda *args: None)

    def imshow(*args):
        assert tracker.second_started.wait(1)

    monkeypatch.setattr(live_app.cv2, "imshow", imshow)
    monkeypatch.setattr(live_app.cv2, "waitKey", lambda *args: ord("q"))
    args = live_app._parser().parse_args(["--session-dir", str(tmp_path)])
    assert live_app.run(args) == 0
    data = json.loads(next(tmp_path.glob("*.json")).read_text())
    state = data["performance"]["pipeline"]
    assert data["frames"] == 1 and state["delivered"] == 1
    assert state["inferences_completed"] == 2 and state["discarded"] == 1
    assert state["pending"] == 0 and state["cleanup_complete"] is True


@pytest.mark.parametrize("args", [["--headless-frames", "-1"], ["--fps", "nan"],
                                   ["--headless-frames", "1", "--max-frames", "1"]])
def test_invalid_cli_never_opens_camera(harness, args):
    capture, _, _ = harness
    with pytest.raises(ValueError):
        live_app.run(live_app._parser().parse_args(args))
    assert not capture.opened
