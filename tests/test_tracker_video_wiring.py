"""Test only the real adapter's wiring with an explicit synthetic native API."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from motioncapture.contracts import CapturedFrame, FrameIdentity
from motioncapture.errors import InferenceError


@pytest.fixture
def module(monkeypatch):
    # This module is never used by production imports or throughput measurements.
    fake_api = SimpleNamespace(
        Image=lambda *, image_format, data: SimpleNamespace(data=data),
        ImageFormat=SimpleNamespace(SRGB="test-only"),
    )
    monkeypatch.setitem(sys.modules, "mediapipe", fake_api)
    path = Path(__file__).parents[1] / "src/motioncapture/landmarkers.py"
    spec = importlib.util.spec_from_file_location("_test_tracker_wiring", path)
    loaded = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "_test_tracker_wiring", loaded)
    spec.loader.exec_module(loaded)
    monkeypatch.setattr(loaded, "require_models", lambda path: {})
    return loaded


def source(index, value):
    return CapturedFrame(
        FrameIdentity("file:test", "s", index, 100 + index,
                      presentation_timestamp_ns=10_000_000_000 + index * 33_333_333),
        np.full((12, 16, 3), value, np.uint8),
    )


@pytest.mark.parametrize("mode", ["allocated", "reuse"])
def test_real_tracker_uses_file_pts_and_preserves_all_three_tasks(module, mode):
    tracker = module.MediaPipeLandmarkTracker(Path("unused-in-test"), rgb_mode=mode)
    calls = []
    empty = SimpleNamespace(
        pose_landmarks=[], pose_world_landmarks=[], hand_landmarks=[],
        hand_world_landmarks=[], face_landmarks=[], face_blendshapes=[], handedness=[],
    )

    class Detector:
        def __init__(self, name):
            self.name = name
        def detect_for_video(self, image, timestamp):
            calls.append((self.name, timestamp, int(image.data[0, 0, 0])))
            return empty
        def close(self):
            pass
    tracker._pose, tracker._hands, tracker._face = [Detector(n) for n in ("body", "hands", "face")]
    for index, value in enumerate([15, 31, 52]):
        frame = source(index, value)
        output = tracker.process(frame)
        assert output.identity == frame.identity
        assert output.model_timestamp_ms == index * 33
    assert len(calls) == 9
    assert calls[-3:] == [(n, 66, 52) for n in ("body", "hands", "face")]
    tracker.close()
    assert tracker._rgb._buffer is None


def test_failed_process_cannot_reuse_buffer_before_explicit_cleanup(module):
    tracker = module.MediaPipeLandmarkTracker(Path("unused"), rgb_mode="reuse")
    class BadDetector:
        def detect_for_video(self, *args):
            raise RuntimeError("test failure")
        def close(self):
            pass
    tracker._pose = tracker._hands = tracker._face = BadDetector()
    with pytest.raises(InferenceError, match="test failure"):
        tracker.process(source(0, 0))
    with pytest.raises(InferenceError, match="Failed tracker"):
        tracker.process(source(1, 1))
    tracker.close()
