"""Explicit MediaPipe Pose, Hands, and Face task pipeline."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import mediapipe as mp

from motioncapture.contracts import (
    CapturedFrame,
    FrameIdentity,
    LandmarkResult,
    LandmarkTimings,
    copy_blendshapes,
    copy_landmarks,
)
from motioncapture.errors import InferenceError
from motioncapture.model_assets import require_models


@dataclass(frozen=True, slots=True)
class LandmarkOutput:
    identity: FrameIdentity
    model_timestamp_ms: int
    result: LandmarkResult
    timings: LandmarkTimings

    @property
    def inference_ms(self) -> float:
        return self.timings.inference_wall_ms


def _first(items: list[list[Any]]) -> list[Any]:
    return items[0] if items else []


def _subject_hands(hand_result: Any) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    left_image: list[Any] = []
    left_world: list[Any] = []
    right_image: list[Any] = []
    right_world: list[Any] = []

    for index, landmarks in enumerate(hand_result.hand_landmarks):
        categories = hand_result.handedness[index] if index < len(hand_result.handedness) else []
        model_label = categories[0].category_name if categories else None
        world_landmarks = (
            hand_result.hand_world_landmarks[index]
            if index < len(hand_result.hand_world_landmarks)
            else []
        )

        # Preserve the existing prototype's subject-relative handedness mapping.
        # Inference is unmirrored; preview mirroring is display-only.
        if model_label == "Right":
            left_image, left_world = landmarks, world_landmarks
        elif model_label == "Left":
            right_image, right_world = landmarks, world_landmarks
        else:
            raise ValueError("Hand detected without supported handedness")

    return left_image, left_world, right_image, right_world


class MediaPipeLandmarkTracker:
    """Own three pinned MediaPipe CPU tasks and their stream-local model clock."""

    provider_base_name = "MediaPipe 0.10.31 CPU (Pose + Hands + Face)"

    def __init__(
        self,
        model_dir: Path,
        task_scheduling: Literal["serial", "parallel"] = "serial",
    ) -> None:
        if task_scheduling not in {"serial", "parallel"}:
            raise ValueError(f"Unsupported task scheduling mode: {task_scheduling}")
        self.model_paths = require_models(model_dir)
        self.task_scheduling = task_scheduling
        self._pose: Any | None = None
        self._hands: Any | None = None
        self._face: Any | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._stream_id: str | None = None
        self._origin_ns: int | None = None
        self._last_timestamp_ms = -1

    @property
    def provider_name(self) -> str:
        return f"{self.provider_base_name}; {self.task_scheduling} tasks"

    @staticmethod
    def _base(path: Path) -> Any:
        return mp.tasks.BaseOptions(
            model_asset_path=str(path),
            delegate=mp.tasks.BaseOptions.Delegate.CPU,
        )

    def open(self) -> None:
        if any(instance is not None for instance in (self._pose, self._hands, self._face)):
            raise InferenceError("MediaPipe landmark tracker is already open")
        self._stream_id = None
        self._origin_ns = None
        self._last_timestamp_ms = -1
        try:
            self._pose = mp.tasks.vision.PoseLandmarker.create_from_options(
                mp.tasks.vision.PoseLandmarkerOptions(
                    base_options=self._base(self.model_paths["pose"]),
                    running_mode=mp.tasks.vision.RunningMode.VIDEO,
                    num_poses=1,
                    min_pose_detection_confidence=0.5,
                    min_pose_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                    output_segmentation_masks=False,
                )
            )
            self._hands = mp.tasks.vision.HandLandmarker.create_from_options(
                mp.tasks.vision.HandLandmarkerOptions(
                    base_options=self._base(self.model_paths["hands"]),
                    running_mode=mp.tasks.vision.RunningMode.VIDEO,
                    num_hands=2,
                    min_hand_detection_confidence=0.5,
                    min_hand_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            )
            self._face = mp.tasks.vision.FaceLandmarker.create_from_options(
                mp.tasks.vision.FaceLandmarkerOptions(
                    base_options=self._base(self.model_paths["face"]),
                    running_mode=mp.tasks.vision.RunningMode.VIDEO,
                    num_faces=1,
                    min_face_detection_confidence=0.5,
                    min_face_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                    output_face_blendshapes=True,
                    output_facial_transformation_matrixes=False,
                )
            )
            if self.task_scheduling == "parallel":
                self._executor = ThreadPoolExecutor(
                    max_workers=3,
                    thread_name_prefix="landmark-task",
                )
        except Exception as exc:
            self.close()
            raise InferenceError(
                f"Unable to initialize MediaPipe Pose + Hands + Face tasks: {exc}"
            ) from exc

    def process(self, frame: CapturedFrame) -> LandmarkOutput:
        if self._pose is None or self._hands is None or self._face is None:
            raise InferenceError("MediaPipe landmark tracker is not open")
        try:
            identity = frame.identity
            if self._origin_ns is None:
                self._origin_ns = identity.received_ns
                self._stream_id = identity.stream_id
            if identity.stream_id != self._stream_id:
                raise ValueError("New capture stream requires a new tracker lifecycle")
            timestamp_ms = (identity.received_ns - self._origin_ns) // 1_000_000
            if timestamp_ms <= self._last_timestamp_ms:
                raise ValueError("Model millisecond timestamps must strictly increase")
            self._last_timestamp_ms = timestamp_ms
            pipeline_started_ns = time.perf_counter_ns()
            image_rgb = cv2.cvtColor(frame.image_bgr, cv2.COLOR_BGR2RGB)
            media_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            conversion_finished_ns = time.perf_counter_ns()

            inference_started_ns = time.perf_counter_ns()
            if self.task_scheduling == "parallel":
                if self._executor is None:
                    raise InferenceError("Parallel task executor is not open")
                pose_future = self._executor.submit(
                    _timed_detect, self._pose, media_image, timestamp_ms
                )
                hands_future = self._executor.submit(
                    _timed_detect, self._hands, media_image, timestamp_ms
                )
                face_future = self._executor.submit(
                    _timed_detect, self._face, media_image, timestamp_ms
                )
                pose, pose_ms = pose_future.result()
                hands, hands_ms = hands_future.result()
                face, face_ms = face_future.result()
            else:
                pose, pose_ms = _timed_detect(self._pose, media_image, timestamp_ms)
                hands, hands_ms = _timed_detect(self._hands, media_image, timestamp_ms)
                face, face_ms = _timed_detect(self._face, media_image, timestamp_ms)
            inference_finished_ns = time.perf_counter_ns()

            left, left_world, right, right_world = _subject_hands(hands)
            result = LandmarkResult(
                pose_landmarks=copy_landmarks(_first(pose.pose_landmarks)),
                pose_world_landmarks=copy_landmarks(_first(pose.pose_world_landmarks)),
                left_hand_landmarks=copy_landmarks(left),
                left_hand_world_landmarks=copy_landmarks(left_world),
                right_hand_landmarks=copy_landmarks(right),
                right_hand_world_landmarks=copy_landmarks(right_world),
                face_landmarks=copy_landmarks(_first(face.face_landmarks)),
                face_blendshapes=copy_blendshapes(_first(face.face_blendshapes)),
            )
            assembly_finished_ns = time.perf_counter_ns()
            timings = LandmarkTimings(
                input_conversion_ms=(conversion_finished_ns - pipeline_started_ns)
                / 1_000_000.0,
                pose_ms=pose_ms,
                hands_ms=hands_ms,
                face_ms=face_ms,
                inference_wall_ms=(inference_finished_ns - inference_started_ns)
                / 1_000_000.0,
                result_assembly_ms=(assembly_finished_ns - inference_finished_ns)
                / 1_000_000.0,
                total_ms=(assembly_finished_ns - pipeline_started_ns) / 1_000_000.0,
            )
            return LandmarkOutput(identity, timestamp_ms, result, timings)
        except Exception as exc:
            raise InferenceError(
                f"MediaPipe inference failed at sequence={frame.identity.sequence}: {exc}"
            ) from exc

    def close(self) -> None:
        executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        pose, self._pose = self._pose, None
        hands, self._hands = self._hands, None
        face, self._face = self._face, None
        for instance in (face, hands, pose):
            if instance is not None:
                instance.close()

    def __enter__(self) -> MediaPipeLandmarkTracker:
        self.open()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()


def _timed_detect(landmarker: Any, media_image: Any, timestamp_ms: int) -> tuple[Any, float]:
    started_ns = time.perf_counter_ns()
    result = landmarker.detect_for_video(media_image, timestamp_ms)
    return result, (time.perf_counter_ns() - started_ns) / 1_000_000.0
