"""Production clock/core with explicit native-task substitutes, not an AI benchmark."""
from __future__ import annotations

import importlib.util
import sys
from fractions import Fraction
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from motioncapture.contracts import CapturedFrame, FrameIdentity
from motioncapture.errors import InferenceError
from motioncapture.recording import RecordedFrame, RecordedIdentity


@pytest.fixture
def tracker(monkeypatch):
    mp = ModuleType("mediapipe")
    mp.Image = lambda *, image_format, data: data
    mp.ImageFormat = SimpleNamespace(SRGB=1)
    assets = ModuleType("motioncapture.model_assets")
    assets.require_models = lambda *_: {}
    monkeypatch.setitem(sys.modules, "mediapipe", mp)
    monkeypatch.setitem(sys.modules, "motioncapture.model_assets", assets)
    # Load an isolated name; do not contaminate the real landmarker import cache.
    source = Path(__file__).parents[1] / "src/motioncapture/landmarkers.py"
    name = "motioncapture._recording_clock_test"
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    t = module.MediaPipeLandmarkTracker(Path("unused"))
    clocks = []

    class Task:
        def __init__(self, result):
            self.result = result

        def detect_for_video(self, image, timestamp):
            clocks.append(timestamp)
            return self.result

    t._pose = Task(SimpleNamespace(pose_landmarks=[], pose_world_landmarks=[]))
    t._hands = Task(SimpleNamespace(hand_landmarks=[], hand_world_landmarks=[], handedness=[]))
    t._face = Task(SimpleNamespace(face_landmarks=[], face_blendshapes=[]))
    return t, clocks


def recorded(pts, sequence=0, stream="s"):
    return RecordedFrame(RecordedIdentity("source", stream, sequence, pts, Fraction(1, 90000)),
                         np.zeros((8, 8, 3), np.uint8), .1)


def test_source_pts_reach_all_three_models_without_cfr_retiming(tracker):
    t, clocks = tracker
    for i, pts in enumerate([90000, 91517, 93033, 96000]):
        f = recorded(pts, i)
        out = t.process_recorded(f)
        assert out.identity is f.identity
        assert not hasattr(out.identity, "received_ns")
    assert clocks == [0] * 3 + [16] * 3 + [33] * 3 + [66] * 3


@pytest.mark.parametrize("pts", [90000, 89999, 90001])
def test_invalid_model_time_fails_not_nudged(tracker, pts):
    t, clocks = tracker
    t.process_recorded(recorded(90000))
    with pytest.raises(InferenceError, match="strictly increase"):
        t.process_recorded(recorded(pts, 1))
    assert clocks == [0, 0, 0]


def test_cannot_change_recording_lifecycle(tracker):
    t, _ = tracker
    t.process_recorded(recorded(0))
    with pytest.raises(InferenceError, match="new tracker lifecycle"):
        t.process_recorded(recorded(1517, 1, "different"))


def test_cannot_mix_live_and_recorded_time_even_with_same_identifiers(tracker):
    t, _ = tracker
    t.process_recorded(recorded(0))
    live = CapturedFrame(FrameIdentity("source", "s", 1, 17_000_000),
                         np.zeros((8, 8, 3), np.uint8))
    with pytest.raises(InferenceError, match="new tracker lifecycle"):
        t.process(live)


def test_live_host_time_path_is_retained(tracker):
    t, clocks = tracker
    for i, ns in enumerate([1_000_000_000, 1_033_300_000]):
        f = CapturedFrame(FrameIdentity("camera", "live", i, ns),
                          np.zeros((8, 8, 3), np.uint8))
        out = t.process(f)
        assert out.identity is f.identity
        assert out.identity.timestamp_provenance == "host_receive_monotonic"
    assert clocks == [0, 0, 0, 33, 33, 33]
