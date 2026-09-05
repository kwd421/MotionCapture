"""Explicit local-camera capture ownership."""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass

import cv2
import numpy as np

from motioncapture.errors import CameraError


@dataclass(frozen=True, slots=True)
class CameraRequest:
    index: int
    width: int
    height: int
    fps: float


@dataclass(frozen=True, slots=True)
class CameraObservation:
    backend: str
    index: int
    width: int
    height: int
    nominal_fps: float
    timestamp_provenance: str = "host_receive_monotonic"


@dataclass(frozen=True, slots=True)
class CameraFrame:
    sequence: int
    timestamp_ns: int
    image_bgr: np.ndarray


class LocalCamera:
    """Own one explicitly selected OpenCV camera until close."""

    def __init__(self, request: CameraRequest) -> None:
        self.request = request
        self._capture: cv2.VideoCapture | None = None
        self._sequence = 0
        self.observation: CameraObservation | None = None

    def open(self) -> CameraObservation:
        if self._capture is not None:
            raise CameraError("Camera is already open")

        system = platform.system()
        if system == "Darwin":
            backend_id = cv2.CAP_AVFOUNDATION
            backend_name = "AVFoundation"
        elif system == "Windows":
            backend_id = cv2.CAP_MSMF
            backend_name = "Media Foundation"
        else:
            raise CameraError(f"Local camera prototype is unsupported on platform: {system}")

        capture = cv2.VideoCapture(self.request.index, backend_id)
        if not capture.isOpened():
            capture.release()
            raise CameraError(
                f"Unable to open explicitly selected camera index {self.request.index} "
                f"with {backend_name}"
            )

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.request.width))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.request.height))
        capture.set(cv2.CAP_PROP_FPS, self.request.fps)

        observation = CameraObservation(
            backend=backend_name,
            index=self.request.index,
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            nominal_fps=float(capture.get(cv2.CAP_PROP_FPS)),
        )
        self._capture = capture
        self.observation = observation
        return observation

    def read(self) -> CameraFrame:
        if self._capture is None:
            raise CameraError("Camera is not open")
        success, image = self._capture.read()
        timestamp_ns = time.monotonic_ns()
        if not success or image is None:
            raise CameraError(
                "Frame read failed for camera index "
                f"{self.request.index} at sequence {self._sequence}"
            )
        frame = CameraFrame(
            sequence=self._sequence,
            timestamp_ns=timestamp_ns,
            image_bgr=image,
        )
        self._sequence += 1
        return frame

    def close(self) -> None:
        capture, self._capture = self._capture, None
        if capture is not None:
            capture.release()

    def __enter__(self) -> LocalCamera:
        self.open()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()
