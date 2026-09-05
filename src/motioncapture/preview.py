"""Display-resolution overlay; source pixels and observations remain untouched."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from motioncapture.contracts import Landmark, LandmarkResult

if TYPE_CHECKING:
    from motioncapture.render import RenderMetrics

WIDTH, HEIGHT, PANEL_WIDTH = 960, 540, 400


def _load_connections() -> tuple[tuple[tuple[int, int], ...], ...]:
    # Missing MediaPipe remains an error. Tests inject topology, never live data.
    import mediapipe as mp

    groups = (
        mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS,
        mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS,
        mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
    )
    return tuple(tuple((edge.start, edge.end) for edge in group) for group in groups)


def _text(
    image: np.ndarray, text: str, xy: tuple[int, int], *,
    scale: float = 0.45, color: tuple[int, int, int] = (225, 231, 239), thickness: int = 1,
) -> None:
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


@dataclass(frozen=True, slots=True)
class _Topology:
    indices: np.ndarray
    edges: np.ndarray
    draw_points: bool

    @classmethod
    def create(cls, pairs: tuple[tuple[int, int], ...], count: int, points: bool) -> _Topology:
        edges = np.asarray(pairs, dtype=np.int32).reshape(-1, 2)
        if edges.size and (edges.min() < 0 or edges.max() >= count):
            raise ValueError("Invalid preview topology")
        indices = np.arange(count, dtype=np.int32) if points else np.unique(edges)
        return cls(indices, np.searchsorted(indices, edges).astype(np.int32), points)


def _draw(
    image: np.ndarray, landmarks: tuple[Landmark, ...], topology: _Topology,
    color: tuple[int, int, int], thickness: int,
) -> None:
    if not landmarks:
        return
    selected = [landmarks[int(index)] for index in topology.indices]
    if not selected:
        return
    height, width = image.shape[:2]
    visible = np.fromiter(
        ((p.presence is None or p.presence >= 0.2)
         and (p.visibility is None or p.visibility >= 0.2) for p in selected),
        dtype=np.bool_, count=len(selected),
    )
    # Project each used vertex once, not for every edge touching that vertex.
    coordinates = np.asarray([(p.x * width, p.y * height) for p in selected])
    if not np.isfinite(coordinates).all() or (np.abs(coordinates) > 2**30).any():
        raise ValueError("Unrenderable landmark coordinates")
    pixels = np.rint(coordinates).astype(np.int32)
    edges = topology.edges
    valid_edges = edges[visible[edges].all(axis=1)]
    if len(valid_edges):
        # Two-point contours retain separate line segments, including disjoint fingers.
        cv2.polylines(image, pixels[valid_edges], False, color, thickness, cv2.LINE_8)
    if topology.draw_points:
        for x, y in pixels[visible]:
            cv2.circle(image, (int(x), int(y)), 2, color, -1, cv2.LINE_AA)


class DisplayPreview:
    """Only immutable text/topology is cached. Every output owns fresh pixels."""

    def __init__(self) -> None:
        pose, hand, face = _load_connections()
        self._pose = _Topology.create(pose, 33, True)
        self._hand = _Topology.create(hand, 21, True)
        self._face = _Topology.create(face, 478, False)
        self._panel = np.full((HEIGHT, PANEL_WIDTH, 3), (21, 25, 33), dtype=np.uint8)
        _text(self._panel, "MOTIONCAPTURE", (22, 34), scale=0.72, thickness=2)
        _text(self._panel, "LIVE 2D LANDMARK PROTOTYPE", (22, 59), color=(97, 202, 255))
        cv2.line(self._panel, (22, 75), (PANEL_WIDTH - 22, 75), (50, 59, 72), 1)
        for y, label in ((104, "Body points"), (132, "Left fingers"),
                         (160, "Right fingers"), (188, "Face contours")):
            _text(self._panel, label, (44, y), scale=0.53)
        _text(self._panel, "FACIAL BLENDSHAPES", (22, 225), scale=0.52,
              color=(255, 164, 222))
        cv2.line(self._panel, (22, 464), (PANEL_WIDTH - 22, 464), (50, 59, 72), 1)
        self._panel.flags.writeable = False

    def compose(
        self, image_bgr: np.ndarray, result: LandmarkResult, metrics: RenderMetrics,
        *, mirror: bool,
    ) -> np.ndarray:
        output = np.empty((HEIGHT, WIDTH + PANEL_WIDTH, 3), dtype=np.uint8)
        preview = output[:, :WIDTH]
        panel = output[:, WIDTH:]
        panel[:] = self._panel
        cv2.resize(image_bgr, (WIDTH, HEIGHT), dst=preview, interpolation=cv2.INTER_AREA)
        _draw(preview, result.pose_landmarks, self._pose, (68, 231, 255), 3)
        _draw(preview, result.left_hand_landmarks, self._hand, (108, 255, 128), 3)
        _draw(preview, result.right_hand_landmarks, self._hand, (255, 155, 82), 3)
        _draw(preview, result.face_landmarks, self._face, (255, 120, 214), 1)
        if mirror:
            cv2.flip(preview, 1, dst=preview)
        # Paint labels after mirroring so only the camera/overlay is reflected.
        cv2.rectangle(preview, (14, 14), (430, 72), (15, 19, 25), -1)
        _text(preview, "LIVE CAMERA - 2D LANDMARKS", (28, 39), scale=0.59, thickness=2)
        _text(preview, "3D reconstruction / retargeting: NOT ACTIVE", (28, 61),
              scale=0.43, color=(104, 194, 255))
        _text(preview, "Q / Esc: quit", (WIDTH - 134, HEIGHT - 18), scale=0.43)
        for y, points, total in (
            (104, result.pose_landmarks, 33), (132, result.left_hand_landmarks, 21),
            (160, result.right_hand_landmarks, 21), (188, result.face_landmarks, 478),
        ):
            color = (92, 225, 142) if points else (92, 103, 120)
            cv2.circle(panel, (28, y - 5), 6, color, -1, cv2.LINE_AA)
            _text(panel, f"{len(points)}/{total}", (238, y), color=color)
        rows = sorted(
            (item for item in result.face_blendshapes
             if item.category_name.lower() not in {"neutral", "_neutral"}),
            key=lambda item: item.score, reverse=True,
        )[:8]
        if not rows:
            _text(panel, "No face result in current frame", (22, 253), color=(126, 137, 153))
        for row, item in enumerate(rows):
            y = 252 + row * 25
            _text(panel, item.category_name[:18], (22, y), scale=0.42)
            cv2.rectangle(panel, (172, y - 11), (317, y - 3), (45, 52, 65), -1)
            cv2.rectangle(panel, (172, y - 11), (172 + int(145 * item.score), y - 3),
                          (224, 101, 190), -1)
            _text(panel, f"{item.score:.2f}", (327, y), scale=0.40)
        _text(panel, f"{metrics.camera_backend} camera {metrics.camera_index}  "
              f"{metrics.frame_width}x{metrics.frame_height}", (22, 486), scale=0.40)
        _text(panel, f"FPS {metrics.processing_fps:4.1f}  "
              f"inference {metrics.inference_ms:5.1f} ms", (22, 506), scale=0.40)
        _text(panel, f"pose {metrics.pose_ms:4.1f}  hands {metrics.hands_ms:4.1f}  "
              f"face {metrics.face_ms:4.1f} ms", (22, 526), scale=0.39)
        return output
