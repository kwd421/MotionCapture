from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from motioncapture import preview
from motioncapture.contracts import Blendshape, Landmark, LandmarkResult


@pytest.fixture
def composer(monkeypatch):
    # Topology fixture, not a real face model or a production fallback.
    monkeypatch.setattr(preview, "_load_connections", lambda: (
        ((0, 1), (1, 2)), ((0, 1), (1, 2)), ((0, 1), (1, 2)),
    ))
    return preview.DisplayPreview()


def metrics():
    return SimpleNamespace(camera_backend="test", camera_index=0, frame_width=1280,
                           frame_height=720, processing_fps=30, inference_ms=15,
                           pose_ms=12, hands_ms=14, face_ms=6)


def result():
    def points(count):
        return tuple(Landmark(0.25 + (i % 3) / 8, 0.4 + (i % 2) / 9, 0) for i in range(count))
    return LandmarkResult(points(33), points(33), points(21), points(21), points(21),
                          points(21), points(478), (Blendshape("jawOpen", 0.7),))


def test_source_observations_and_previous_output_are_not_mutated(composer):
    rng = np.random.default_rng(9)
    source = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    original = source.copy()
    observation = result()
    first = composer.compose(source, observation, metrics(), mirror=False)
    saved = first.copy()
    composer.compose(source, observation, metrics(), mirror=True)
    np.testing.assert_array_equal(source, original)
    np.testing.assert_array_equal(first, saved)
    assert not np.shares_memory(source, first)
    assert observation.face_blendshapes[0].score == 0.7


def test_mirroring_reflects_only_camera_overlay(composer):
    source = np.random.default_rng(2).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    direct = composer.compose(source, result(), metrics(), mirror=False)
    mirrored = composer.compose(source, result(), metrics(), mirror=True)
    np.testing.assert_array_equal(direct[110:480, :960, :][:, ::-1],
                                  mirrored[110:480, :960])
    np.testing.assert_array_equal(direct[:, 960:], mirrored[:, 960:])


def test_missing_detection_has_no_retained_pose_or_blendshape(composer):
    source = np.zeros((720, 1280, 3), dtype=np.uint8)
    composer.compose(source, result(), metrics(), mirror=False)
    empty = LandmarkResult((), (), (), (), (), (), (), ())
    blank = composer.compose(source, empty, metrics(), mirror=False)
    assert not blank[110:480, :960].any()
    # Template is fixed, not a mutable cache of the previous pose or score bars.
    assert composer._panel.flags.writeable is False


def test_display_background_matches_opencv_resize(composer):
    source = np.random.default_rng(3).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    empty = LandmarkResult((), (), (), (), (), (), (), ())
    image = composer.compose(source, empty, metrics(), mirror=False)
    expected = cv2.resize(source, (960, 540), interpolation=cv2.INTER_AREA)
    np.testing.assert_array_equal(image[110:480, :960], expected[110:480])


@pytest.mark.parametrize("draw_points", [True, False])
def test_batched_segments_match_separate_line_drawing(draw_points):
    rng = np.random.default_rng(5)
    points = tuple(Landmark(float(x), float(y), 0, visibility=float(v))
                   for x, y, v in rng.uniform(-0.2, 1.2, (30, 3)))
    edges = tuple((i, i + 1) for i in range(29))
    topology = preview._Topology.create(edges, 30, draw_points)
    actual = np.zeros((140, 180, 3), dtype=np.uint8)
    expected = actual.copy()
    color = (68, 231, 255)
    preview._draw(actual, points, topology, color, 3)
    for start, end in edges:
        if points[start].visibility >= 0.2 and points[end].visibility >= 0.2:
            a = (round(points[start].x * 180), round(points[start].y * 140))
            b = (round(points[end].x * 180), round(points[end].y * 140))
            cv2.line(expected, a, b, color, 3)
    if draw_points:
        for point in points:
            if point.visibility >= 0.2:
                cv2.circle(expected, (round(point.x * 180), round(point.y * 140)),
                           2, color, -1, cv2.LINE_AA)
    np.testing.assert_array_equal(actual, expected)


def test_bad_geometry_is_not_zero_filled(composer):
    points = (Landmark(float("nan"), 0.3, 0),) * 33
    bad = LandmarkResult(points, (), (), (), (), (), (), ())
    with pytest.raises(ValueError, match="Unrenderable"):
        composer.compose(np.zeros((16, 16, 3), np.uint8), bad, metrics(), mirror=False)


def test_face_contours_only_project_referenced_vertices(composer):
    assert tuple(composer._face.indices) == (0, 1, 2)
    assert len(composer._pose.indices) == 33


def test_invalid_topology_is_error():
    with pytest.raises(ValueError):
        preview._Topology.create(((0, 5),), 2, False)
