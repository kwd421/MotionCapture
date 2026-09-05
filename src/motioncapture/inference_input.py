"""Model-clock selection and optional owner-local RGB storage, not model inference."""
from __future__ import annotations

import cv2
import numpy as np

from motioncapture.contracts import FrameIdentity


class ModelClock:
    """Original PTS wins in explicit offline mode; never fabricate millisecond ticks."""

    def __init__(self) -> None:
        self._key: tuple[str, str, str] | None = None
        self._origin_ns = 0
        self._last_ms = -1
        self._last_sequence = -1

    def accept(self, identity: FrameIdentity) -> int:
        presentation = identity.presentation_timestamp_ns
        basis = "host_receive_monotonic" if presentation is None else "media_presentation"
        time_ns = identity.received_ns if presentation is None else presentation
        key = (identity.source_id, identity.stream_id, basis)
        if self._key is not None and self._key != key:
            raise ValueError("New source, stream or clock requires a new tracker lifecycle")
        origin = time_ns if self._key is None else self._origin_ns
        milliseconds = (time_ns - origin) // 1_000_000
        if milliseconds <= self._last_ms or identity.sequence <= self._last_sequence:
            raise ValueError("Model millisecond timestamps and frame sequence must increase")
        self._key, self._origin_ns = key, origin
        self._last_ms, self._last_sequence = milliseconds, identity.sequence
        return milliseconds


class RgbConverter:
    """Reusable output is borrowed until the next call. One synchronous tracker owner.

    The owner MUST finish all consumers before the next conversion. Do not store
    the reusable output in a frame queue. This class never touches the source.
    """

    def __init__(self, mode: str = "allocated") -> None:
        if mode not in {"allocated", "reuse"}:
            raise ValueError("RGB mode must be allocated or reuse")
        self.mode = mode
        self._buffer: np.ndarray | None = None

    def convert(self, image: np.ndarray) -> np.ndarray:
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not image.size:
            raise ValueError("RGB conversion requires a nonempty uint8 BGR frame")
        if self.mode == "allocated":
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self._buffer is None or self._buffer.shape != image.shape:
            self._buffer = np.empty(image.shape, dtype=np.uint8)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB, dst=self._buffer)

    def close(self) -> None:
        self._buffer = None
