from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from motioncapture.contracts import (
    CapturedFrame,
    FrameIdentity,
    LandmarkResult,
    copy_blendshapes,
    copy_landmarks,
)


def test_numeric_copy_is_owned_and_retains_unknown_confidence() -> None:
    source = SimpleNamespace(x=-0.2, y=1.3, z=-0.6, presence=None, visibility=0.7)
    points = copy_landmarks([source])
    source.x = 100
    assert (points[0].x, points[0].y, points[0].z) == (-0.2, 1.3, -0.6)
    assert points[0].presence is None
    assert points[0].visibility == 0.7
    with pytest.raises(FrozenInstanceError):
        points[0].x = 5


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), "0.5", True])
def test_invalid_coordinate_is_not_an_empty_or_zero_result(bad: object) -> None:
    with pytest.raises((ValueError, TypeError)):
        copy_landmarks([SimpleNamespace(x=bad, y=0.0, z=0.0)])


def test_blendshape_zero_is_valid_but_missing_and_duplicate_are_errors() -> None:
    result = copy_blendshapes([SimpleNamespace(category_name="jawOpen", score=0.0)])
    assert result[0].score == 0.0
    assert result[0].index is None
    with pytest.raises(ValueError, match="Missing"):
        copy_blendshapes([SimpleNamespace(category_name="jawOpen", score=None)])
    with pytest.raises(ValueError, match="unique"):
        copy_blendshapes([result[0], result[0]])


def test_missing_detection_is_explicit_and_wrong_topology_is_invalid() -> None:
    empty = LandmarkResult((), (), (), (), (), (), (), ())
    assert set(empty.detection_state.values()) == {"missing"}
    points = copy_landmarks([SimpleNamespace(x=0.2, y=0.5, z=0.0)])
    with pytest.raises(ValueError, match="topology"):
        LandmarkResult(points, (), (), (), (), (), (), ())
    body = LandmarkResult(points * 33, (), (), (), (), (), (), ())
    assert body.detection_state["pose"] == "valid"
    assert body.detection_state["face"] == "missing"


def test_frame_identity_preserves_host_time_without_printing_camera_pixels() -> None:
    identity = FrameIdentity("local:test:0", "stream-1", 9, 123_456_789)
    frame = CapturedFrame(identity, np.zeros((2, 2, 3), dtype=np.uint8))
    assert frame.identity.received_ns == 123_456_789
    assert frame.identity.timestamp_provenance == "host_receive_monotonic"
    assert "array(" not in repr(frame)
    with pytest.raises(ValueError):
        FrameIdentity("", "stream", 1, 1)
    with pytest.raises(ValueError):
        FrameIdentity("camera", "stream", -1, 1)


def test_invalid_frame_buffer_is_not_a_valid_capture() -> None:
    identity = FrameIdentity("local:test:0", "stream", 0, 1)
    with pytest.raises(ValueError, match="BGR"):
        CapturedFrame(identity, np.zeros((2, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match="integers"):
        FrameIdentity("local:test:0", "stream", 0, float("nan"))
