"""Owned frame identity and numeric observations for the single-person slice.

Image x/y are normalized; image z is model-relative, not metric depth.
Pose world points are estimated hip-relative meters. Hand world points are
estimated hand-relative meters. Neither is a calibrated studio coordinate.
No MediaPipe objects cross this boundary.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass(frozen=True, slots=True)
class FrameIdentity:
    source_id: str
    stream_id: str
    sequence: int
    received_ns: int
    timestamp_provenance: Literal["host_receive_monotonic"] = "host_receive_monotonic"

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not isinstance(self.stream_id, str):
            raise ValueError("Frame source and stream identifiers must be strings")
        if not self.source_id or not self.stream_id:
            raise ValueError("Frame identity requires a source and a stream")
        if type(self.sequence) is not int or type(self.received_ns) is not int:
            raise ValueError("Frame sequence and host time must be integers")
        if self.sequence < 0 or self.received_ns < 0:
            raise ValueError("Frame sequence and host time must be nonnegative")
        if self.timestamp_provenance != "host_receive_monotonic":
            raise ValueError("This slice supports host receive timestamps only")


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    identity: FrameIdentity
    image_bgr: np.ndarray = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        image = self.image_bgr
        if (
            not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3
            or image.shape[0] == 0 or image.shape[1] == 0
        ):
            raise ValueError("Captured image must be a nonempty uint8 HxWx3 BGR array")


def _number(value: object) -> float:
    if value is None or isinstance(value, (str, bytes, bool)):
        raise ValueError("Missing or nonnumeric observation")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Nonfinite observation")
    return result


def _confidence(value: object) -> float | None:
    if value is None:
        return None
    result = _number(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError("Confidence must lie in [0, 1]")
    return result


@dataclass(frozen=True, slots=True)
class Landmark:
    x: float
    y: float
    z: float
    visibility: float | None = None
    presence: float | None = None


@dataclass(frozen=True, slots=True)
class Blendshape:
    category_name: str
    score: float
    index: int | None = None


def copy_landmarks(items: Iterable[object]) -> tuple[Landmark, ...]:
    """Convert valid model results; absence of detection is an empty tuple.

    Missing x/y/z, corrupt numeric values and adapter errors are NOT converted
    to empty results. The inference owner must propagate them as failures.
    """
    return tuple(
        Landmark(
            x=_number(item.x),
            y=_number(item.y),
            z=_number(item.z),
            visibility=_confidence(getattr(item, "visibility", None)),
            presence=_confidence(getattr(item, "presence", None)),
        )
        for item in items
    )


def copy_blendshapes(items: Iterable[object]) -> tuple[Blendshape, ...]:
    result = []
    names = set()
    for item in items:
        name = item.category_name
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Blendshape names must be present and unique")
        score = _confidence(item.score)
        if score is None:
            raise ValueError("Missing blendshape score is not zero")
        names.add(name)
        result.append(Blendshape(name, score, getattr(item, "index", None)))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class LandmarkResult:
    pose_landmarks: tuple[Landmark, ...]
    pose_world_landmarks: tuple[Landmark, ...]
    left_hand_landmarks: tuple[Landmark, ...]
    left_hand_world_landmarks: tuple[Landmark, ...]
    right_hand_landmarks: tuple[Landmark, ...]
    right_hand_world_landmarks: tuple[Landmark, ...]
    face_landmarks: tuple[Landmark, ...]
    face_blendshapes: tuple[Blendshape, ...]

    def __post_init__(self) -> None:
        for name, expected in (
            ("pose_landmarks", 33), ("pose_world_landmarks", 33),
            ("left_hand_landmarks", 21), ("left_hand_world_landmarks", 21),
            ("right_hand_landmarks", 21), ("right_hand_world_landmarks", 21),
            ("face_landmarks", 478),
        ):
            points = getattr(self, name)
            if not isinstance(points, tuple) or len(points) not in (0, expected):
                raise ValueError(f"Invalid {name} topology: expected zero or {expected} points")

    @property
    def detection_state(self) -> dict[str, str]:
        # Derived from the owned result, not a second mutable set of flags.
        return {
            name: "valid" if points else "missing"
            for name, points in (
                ("pose", self.pose_landmarks),
                ("left_hand", self.left_hand_landmarks),
                ("right_hand", self.right_hand_landmarks),
                ("face", self.face_landmarks),
                ("face_blendshapes", self.face_blendshapes),
            )
        }


@dataclass(frozen=True, slots=True)
class LandmarkTimings:
    input_conversion_ms: float
    pose_ms: float
    hands_ms: float
    face_ms: float
    inference_wall_ms: float
    result_assembly_ms: float
    total_ms: float
