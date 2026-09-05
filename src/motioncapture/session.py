"""Metadata-only session records for reproducible prototype runs."""

from __future__ import annotations

import json
import os
import platform
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import cv2

from motioncapture import __version__
from motioncapture.capture import CameraObservation, CameraRequest
from motioncapture.contracts import FrameIdentity, LandmarkTimings
from motioncapture.errors import SessionRecordError
from motioncapture.model_assets import MODEL_ASSETS
from motioncapture.runtime import CaptureSnapshot


@dataclass(slots=True)
class DetectionTotals:
    pose_frames: int = 0
    face_frames: int = 0
    left_hand_frames: int = 0
    right_hand_frames: int = 0


@dataclass(slots=True)
class LatencySummary:
    total_ms: float = 0.0
    maximum_ms: float = 0.0

    def observe(self, duration_ms: float) -> None:
        self.total_ms += duration_ms
        self.maximum_ms = max(self.maximum_ms, duration_ms)

    def payload(self, samples: int) -> dict[str, float | None]:
        return {
            "mean_ms": self.total_ms / samples if samples else None,
            "maximum_ms": self.maximum_ms if samples else None,
        }


def _installed_version(distribution: str) -> str | None:
    # Reading diagnostics must not initialize the native inference runtime.
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


class SessionRecord:
    def __init__(
        self,
        camera_request: CameraRequest,
        mirror: bool,
        *,
        inference_provider: str,
        task_scheduling: str,
    ) -> None:
        self.session_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
        self.started_at = datetime.now(UTC)
        self.camera_request = camera_request
        self.mirror = mirror
        self.inference_provider = inference_provider
        self.task_scheduling = task_scheduling
        self.camera_observation: CameraObservation | None = None
        self.capture_snapshot: CaptureSnapshot | None = None
        self.first_frame_identity: FrameIdentity | None = None
        self.last_frame_identity: FrameIdentity | None = None
        self.capture_queue_latency = LatencySummary()
        self.capture_queue_samples = 0
        self.frames = 0
        self.detections = DetectionTotals()
        self.latencies = {
            "input_conversion": LatencySummary(),
            "pose": LatencySummary(),
            "hands": LatencySummary(),
            "face": LatencySummary(),
            "inference_wall": LatencySummary(),
            "result_assembly": LatencySummary(),
            "tracker_total": LatencySummary(),
            "preview_composition": LatencySummary(),
            "host_post_receive_total": LatencySummary(),
        }
        self.first_frame_timestamp_ns: int | None = None
        self.last_frame_timestamp_ns: int | None = None
        self.terminal_status = "running"
        self.terminal_error: str | None = None

    def attach_camera(self, observation: CameraObservation) -> None:
        self.camera_observation = observation

    def attach_capture(self, snapshot: CaptureSnapshot) -> None:
        self.capture_snapshot = snapshot

    def observe(
        self,
        result: Any,
        timings: LandmarkTimings,
        *,
        preview_composition_ms: float,
        host_post_receive_total_ms: float,
        frame_timestamp_ns: int,
        capture_queue_ms: float | None = None,
        frame_identity: FrameIdentity | None = None,
    ) -> None:
        if frame_identity is not None:
            if frame_identity.received_ns != frame_timestamp_ns:
                raise ValueError("Session frame identity and timestamp disagree")
            if self.first_frame_identity is None:
                self.first_frame_identity = frame_identity
            self.last_frame_identity = frame_identity
        if capture_queue_ms is not None:
            self.capture_queue_latency.observe(capture_queue_ms)
            self.capture_queue_samples += 1
        self.frames += 1
        if self.first_frame_timestamp_ns is None:
            self.first_frame_timestamp_ns = frame_timestamp_ns
        self.last_frame_timestamp_ns = frame_timestamp_ns
        self.latencies["input_conversion"].observe(timings.input_conversion_ms)
        self.latencies["pose"].observe(timings.pose_ms)
        self.latencies["hands"].observe(timings.hands_ms)
        self.latencies["face"].observe(timings.face_ms)
        self.latencies["inference_wall"].observe(timings.inference_wall_ms)
        self.latencies["result_assembly"].observe(timings.result_assembly_ms)
        self.latencies["tracker_total"].observe(timings.total_ms)
        self.latencies["preview_composition"].observe(preview_composition_ms)
        self.latencies["host_post_receive_total"].observe(host_post_receive_total_ms)
        self.detections.pose_frames += int(bool(result.pose_landmarks))
        self.detections.face_frames += int(bool(result.face_landmarks))
        self.detections.left_hand_frames += int(bool(result.left_hand_landmarks))
        self.detections.right_hand_frames += int(bool(result.right_hand_landmarks))

    def finish(self, status: str, error: str | None = None) -> None:
        self.terminal_status = status
        self.terminal_error = error

    def payload(self) -> dict[str, Any]:
        finished_at = datetime.now(UTC)
        duration_seconds = max((finished_at - self.started_at).total_seconds(), 0.0)
        frame_span_seconds = None
        if (
            self.first_frame_timestamp_ns is not None
            and self.last_frame_timestamp_ns is not None
            and self.frames >= 2
        ):
            frame_span_seconds = (
                self.last_frame_timestamp_ns - self.first_frame_timestamp_ns
            ) / 1_000_000_000.0
        return {
            "schema_version": 2,
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": duration_seconds,
            "terminal_status": self.terminal_status,
            "terminal_error": self.terminal_error,
            "application": {"name": "motioncapture", "version": __version__},
            "runtime": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "opencv": cv2.__version__,
                "mediapipe": _installed_version("mediapipe"),
                "inference_provider": self.inference_provider,
            },
            "models": [
                {"key": asset.key, "name": asset.name, "sha256": asset.sha256}
                for asset in MODEL_ASSETS
            ],
            "mode": {
                "capture": "live_local_camera",
                "delivery": "latest_only_single_slot",
                "geometry": "monocular_2d_landmarks",
                "preview_mirrored": self.mirror,
                "calibrated_3d": False,
                "retargeting": False,
                "task_scheduling": self.task_scheduling,
            },
            "camera_request": asdict(self.camera_request),
            "camera_observation": (
                asdict(self.camera_observation) if self.camera_observation is not None else None
            ),
            "capture": (
                asdict(self.capture_snapshot) if self.capture_snapshot is not None else None
            ),
            "first_processed_frame": (
                asdict(self.first_frame_identity) if self.first_frame_identity is not None else None
            ),
            "last_processed_frame": (
                asdict(self.last_frame_identity) if self.last_frame_identity is not None else None
            ),
            "frames": self.frames,
            "latency": {
                "provenance": "host_process_perf_counter",
                "camera_sensor_to_display_measured": False,
                "capture_queue_provenance": "host_receive_to_dequeue_monotonic",
                "capture_queue": self.capture_queue_latency.payload(self.capture_queue_samples),
                "post_receive_includes_queue": True,
                "post_receive_provenance": "host_receive_to_preview_monotonic",
                "stages": {
                    name: summary.payload(self.frames)
                    for name, summary in self.latencies.items()
                },
            },
            "mean_processing_fps": (
                (self.frames - 1) / frame_span_seconds
                if frame_span_seconds is not None and frame_span_seconds > 0
                else None
            ),
            "detection_totals": asdict(self.detections),
            "privacy": {"raw_frames_recorded": False, "landmarks_recorded": False},
        }

    def write(self, directory: Path) -> Path:
        directory = directory.expanduser().resolve()
        try:
            directory.mkdir(parents=True, exist_ok=True)
            destination = directory / f"{self.session_id}.json"
            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    prefix="session-",
                    suffix=".partial",
                    dir=directory,
                    delete=False,
                ) as temporary_file:
                    temporary_path = Path(temporary_file.name)
                    json.dump(self.payload(), temporary_file, ensure_ascii=False, indent=2)
                    temporary_file.write("\n")
                os.replace(temporary_path, destination)
                temporary_path = None
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
        except OSError as exc:
            raise SessionRecordError(f"Unable to write session manifest: {exc}") from exc
        return destination
