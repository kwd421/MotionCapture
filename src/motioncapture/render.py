"""OpenCV visualization for the honest 2D landmark prototype."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540
PANEL_WIDTH = 400

POSE_CONNECTIONS = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS
HAND_CONNECTIONS = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
FACE_CONNECTIONS = mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS


@dataclass(frozen=True, slots=True)
class RenderMetrics:
    session_id: str
    camera_index: int
    camera_backend: str
    frame_width: int
    frame_height: int
    nominal_fps: float
    processing_fps: float
    inference_ms: float
    pose_ms: float
    hands_ms: float
    face_ms: float
    frame_sequence: int
    timestamp_ms: int
    provider: str


def _visible(landmark: Any, threshold: float = 0.2) -> bool:
    presence = getattr(landmark, "presence", None)
    visibility = getattr(landmark, "visibility", None)
    return (presence is None or presence >= threshold) and (
        visibility is None or visibility >= threshold
    )


def _point(landmark: Any, width: int, height: int) -> tuple[int, int]:
    return int(round(landmark.x * width)), int(round(landmark.y * height))


def _draw_connections(
    image: np.ndarray,
    landmarks: list[Any],
    connections: Iterable[Any],
    color: tuple[int, int, int],
    thickness: int,
    draw_points: bool = True,
) -> None:
    if not landmarks:
        return
    height, width = image.shape[:2]
    for connection in connections:
        if connection.start >= len(landmarks) or connection.end >= len(landmarks):
            continue
        start = landmarks[connection.start]
        end = landmarks[connection.end]
        if not _visible(start) or not _visible(end):
            continue
        cv2.line(image, _point(start, width, height), _point(end, width, height), color, thickness)
    if draw_points:
        for landmark in landmarks:
            if _visible(landmark):
                cv2.circle(image, _point(landmark, width, height), 2, color, -1, cv2.LINE_AA)


def draw_landmarks(image: np.ndarray, result: Any) -> None:
    _draw_connections(image, result.pose_landmarks, POSE_CONNECTIONS, (68, 231, 255), 3)
    _draw_connections(image, result.left_hand_landmarks, HAND_CONNECTIONS, (108, 255, 128), 3)
    _draw_connections(image, result.right_hand_landmarks, HAND_CONNECTIONS, (255, 155, 82), 3)
    _draw_connections(
        image,
        result.face_landmarks,
        FACE_CONNECTIONS,
        (255, 120, 214),
        1,
        draw_points=False,
    )


def _put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.5,
    color: tuple[int, int, int] = (225, 231, 239),
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _status_line(
    panel: np.ndarray, y: int, label: str, active: bool, detail: str
) -> None:
    color = (92, 225, 142) if active else (92, 103, 120)
    cv2.circle(panel, (28, y - 5), 6, color, -1, cv2.LINE_AA)
    _put_text(panel, label, (44, y), scale=0.53, color=(238, 242, 248), thickness=1)
    _put_text(panel, detail, (238, y), scale=0.46, color=color, thickness=1)


def _blendshape_rows(result: Any) -> list[tuple[str, float]]:
    blendshapes = result.face_blendshapes or []
    rows = [
        (item.category_name or f"blend_{item.index}", float(item.score or 0.0))
        for item in blendshapes
        if (item.category_name or "").lower() not in {"_neutral", "neutral"}
    ]
    rows.sort(key=lambda item: item[1], reverse=True)
    return rows[:8]


def _render_panel(result: Any, metrics: RenderMetrics) -> np.ndarray:
    panel = np.full((DISPLAY_HEIGHT, PANEL_WIDTH, 3), (21, 25, 33), dtype=np.uint8)
    _put_text(panel, "MOTIONCAPTURE", (22, 34), scale=0.72, color=(245, 248, 252), thickness=2)
    _put_text(panel, "LIVE 2D LANDMARK PROTOTYPE", (22, 59), scale=0.47, color=(97, 202, 255))
    cv2.line(panel, (22, 75), (PANEL_WIDTH - 22, 75), (50, 59, 72), 1)

    pose_count = len(result.pose_landmarks)
    face_count = len(result.face_landmarks)
    left_count = len(result.left_hand_landmarks)
    right_count = len(result.right_hand_landmarks)

    _status_line(panel, 104, "Body bones", pose_count > 0, f"{pose_count}/33")
    _status_line(panel, 132, "Left fingers", left_count > 0, f"{left_count}/21")
    _status_line(panel, 160, "Right fingers", right_count > 0, f"{right_count}/21")
    _status_line(panel, 188, "Face contours", face_count > 0, f"{face_count}/478")

    _put_text(
        panel,
        "FACIAL BLENDSHAPES",
        (22, 225),
        scale=0.52,
        color=(255, 164, 222),
        thickness=1,
    )
    rows = _blendshape_rows(result)
    if not rows:
        _put_text(panel, "No face result in current frame", (22, 253), color=(126, 137, 153))
    for row_index, (name, score) in enumerate(rows):
        y = 252 + row_index * 25
        short_name = name[:18]
        _put_text(panel, short_name, (22, y), scale=0.42, color=(220, 225, 234))
        bar_x = 172
        bar_width = 145
        cv2.rectangle(panel, (bar_x, y - 11), (bar_x + bar_width, y - 3), (45, 52, 65), -1)
        cv2.rectangle(
            panel,
            (bar_x, y - 11),
            (bar_x + int(bar_width * min(max(score, 0.0), 1.0)), y - 3),
            (224, 101, 190),
            -1,
        )
        _put_text(panel, f"{score:0.2f}", (327, y), scale=0.40, color=(201, 207, 218))

    cv2.line(panel, (22, 464), (PANEL_WIDTH - 22, 464), (50, 59, 72), 1)
    _put_text(
        panel,
        f"{metrics.camera_backend} camera {metrics.camera_index}  "
        f"{metrics.frame_width}x{metrics.frame_height}",
        (22, 486),
        scale=0.40,
        color=(167, 177, 191),
    )
    _put_text(
        panel,
        f"FPS {metrics.processing_fps:4.1f}  inference {metrics.inference_ms:5.1f} ms",
        (22, 506),
        scale=0.40,
        color=(167, 177, 191),
    )
    _put_text(
        panel,
        f"pose {metrics.pose_ms:4.1f}  hands {metrics.hands_ms:4.1f}  "
        f"face {metrics.face_ms:4.1f} ms",
        (22, 526),
        scale=0.39,
        color=(167, 177, 191),
    )
    return panel


def compose_preview(
    image_bgr: np.ndarray,
    result: Any,
    metrics: RenderMetrics,
    *,
    mirror: bool,
) -> np.ndarray:
    annotated = image_bgr.copy()
    draw_landmarks(annotated, result)
    if mirror:
        annotated = cv2.flip(annotated, 1)
    preview = cv2.resize(annotated, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_AREA)

    cv2.rectangle(preview, (14, 14), (356, 72), (15, 19, 25), -1)
    _put_text(preview, "LIVE CAMERA - 2D LANDMARKS", (28, 39), scale=0.59, thickness=2)
    _put_text(
        preview,
        "3D reconstruction / retargeting: NOT ACTIVE",
        (28, 61),
        scale=0.43,
        color=(104, 194, 255),
    )
    _put_text(
        preview,
        "Q / Esc: quit",
        (DISPLAY_WIDTH - 134, DISPLAY_HEIGHT - 18),
        scale=0.43,
        color=(230, 235, 242),
    )
    return np.hstack((preview, _render_panel(result, metrics)))
