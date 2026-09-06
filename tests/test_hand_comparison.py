from types import SimpleNamespace as NS

import pytest

from motioncapture.contracts import Landmark, LandmarkResult
from motioncapture.hand_comparison import HandReference


def frame(i, pts):
    return NS(identity=NS(sequence=i, pts=pts))


def result(left=True, right=True, dx=0):
    p = (Landmark(.2 + dx, .3, 0),) * 21
    w = (Landmark(.01, .02, .03),) * 21
    return LandmarkResult((), (), p if left else (), w if left else (),
                          p if right else (), w if right else (), (), ())


def test_numeric_displacement_and_presence_are_independent():
    ref = HandReference([10, 20], 1920, 1080)
    ref.record(frame(0, 10), result())
    ref.record(frame(1, 20), result(False, False))
    compare = ref.comparator()
    compare.observe(frame(0, 10), result(True, False, dx=1/1920))
    compare.observe(frame(1, 20), result(False, True))
    summary = compare.summary()
    assert summary["reference_only_frames"] == [0, 1]
    assert summary["candidate_only_frames"] == [0, 1]
    assert summary["image_xy_displacement_pixels"]["mean"] == pytest.approx(1)
    assert summary["hand_relative_world_displacement_mm"]["max"] == 0
    assert not summary["accuracy_verified"]


def test_no_matching_hands_is_unknown_not_zero_error():
    ref = HandReference([0], 1280, 720)
    ref.record(frame(0, 0), result())
    compare = ref.comparator()
    compare.observe(frame(0, 0), result(False, False))
    assert compare.summary()["image_xy_displacement_pixels"]["mean"] is None


def test_reference_requires_exact_order_pts_and_completeness():
    ref = HandReference([10, 20], 1280, 720)
    with pytest.raises(ValueError):
        ref.record(frame(1, 20), result())
    ref.record(frame(0, 10), result())
    with pytest.raises(ValueError):
        ref.comparator()
    with pytest.raises(ValueError):
        ref.record(frame(1, 21), result())
