"""Single-camera capture ownership with one pending, latest-only frame.

Only the producer thread opens, reads and closes the camera. A Condition owns
both the slot and its accounting; consumers cannot accidentally clear a newer
frame. Inference remains on the caller. This is not multi-camera synchronization
or native-process crash isolation.
"""

from __future__ import annotations

import math
import threading
import uuid
from dataclasses import dataclass
from typing import Literal, Protocol

from motioncapture.capture import CameraFrame, CameraObservation
from motioncapture.contracts import CapturedFrame, FrameIdentity
from motioncapture.errors import CameraError


class CameraSource(Protocol):
    def open(self) -> CameraObservation: ...
    def read(self) -> CameraFrame: ...
    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class CaptureSnapshot:
    state: str
    stream_id: str
    captured: int
    delivered: int
    replaced: int
    discarded: int
    pending: int
    first_received_ns: int | None
    last_received_ns: int | None
    mean_capture_fps: float | None
    cleanup_complete: bool
    terminal_error: str | None


class CaptureRuntime:
    """Own one camera lifecycle. Construct a new runtime for each new stream."""

    def __init__(self, source: CameraSource, *, timeout: float = 10.0) -> None:
        if not math.isfinite(timeout) or not 0 < timeout <= threading.TIMEOUT_MAX:
            raise ValueError("Capture timeout must be positive, finite and supported")
        self._source = source
        self._timeout = timeout
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._state: Literal["new", "starting", "running", "stopping", "stopped", "failed"] = "new"
        self._error: BaseException | None = None
        self._observation: CameraObservation | None = None
        self._pending: CapturedFrame | None = None
        self._stream_id = uuid.uuid4().hex
        self._captured = self._delivered = self._replaced = self._discarded = 0
        self._first_ns: int | None = None
        self._last_ns: int | None = None
        self._last_sequence = -1
        self._cleanup_complete = False

    @property
    def observation(self) -> CameraObservation | None:
        with self._condition:
            return self._observation

    def snapshot(self) -> CaptureSnapshot:
        with self._condition:
            fps = None
            if self._captured >= 2 and self._first_ns is not None and self._last_ns is not None:
                span = self._last_ns - self._first_ns
                if span > 0:
                    fps = (self._captured - 1) * 1_000_000_000 / span
            return CaptureSnapshot(
                self._state, self._stream_id, self._captured, self._delivered,
                self._replaced, self._discarded, int(self._pending is not None),
                self._first_ns, self._last_ns, fps, self._cleanup_complete,
                str(self._error) if self._error is not None else None,
            )

    def _discard_pending(self) -> None:
        if self._pending is not None:
            self._discarded += 1
            self._pending = None

    def _fail(self, error: BaseException) -> None:
        # Called with the condition lock, including from shutdown on the caller.
        if self._error is None:
            self._error = error
        elif error is not self._error:
            self._error.add_note(f"Additional capture failure: {type(error).__name__}: {error}")
        self._state = "failed"
        self._stop.set()
        self._discard_pending()
        self._condition.notify_all()

    def check(self) -> None:
        """Do not publish an inference result after noticing a capture failure."""
        with self._condition:
            if self._error is not None:
                raise CameraError(f"Capture failed: {self._error}") from self._error

    def open(self) -> CameraObservation:
        with self._condition:
            if self._state != "new":
                raise CameraError("Capture runtime cannot be opened twice")
            self._state = "starting"
            # A wedged native driver cannot be killed safely as a Python thread.
            # Timeout is an explicit failed/cleanup-incomplete outcome, never OK.
            self._thread = threading.Thread(
                target=self._produce, name="camera-capture", daemon=True,
            )
            try:
                self._thread.start()
            except BaseException as exc:
                self._thread = None
                self._fail(exc)
                raise
        try:
            with self._condition:
                if not self._condition.wait_for(lambda: self._state != "starting", self._timeout):
                    self._fail(CameraError("Timed out opening selected camera"))
                self.check()
                if self._state != "running" or self._observation is None:
                    raise CameraError("Camera startup was cancelled")
                return self._observation
        except BaseException as exc:
            try:
                self.close()
            except CameraError as cleanup_error:
                exc.add_note(str(cleanup_error))
            raise

    def _produce(self) -> None:
        try:
            observation = self._source.open()
            if observation.timestamp_provenance != "host_receive_monotonic":
                raise CameraError("Unsupported capture timestamp provenance")
            source_id = f"local:{observation.backend}:{observation.index}"
            with self._condition:
                self._observation = observation
                if not self._stop.is_set():
                    self._state = "running"
                self._condition.notify_all()
            while not self._stop.is_set():
                frame = self._source.read()
                identity = FrameIdentity(
                    source_id, self._stream_id, frame.sequence, frame.timestamp_ns,
                )
                captured_frame = CapturedFrame(identity, frame.image_bgr)
                with self._condition:
                    if frame.sequence <= self._last_sequence or (
                        self._last_ns is not None and frame.timestamp_ns <= self._last_ns
                    ):
                        raise CameraError("Non-monotonic capture sequence or host timestamp")
                    self._captured += 1
                    self._last_sequence = frame.sequence
                    self._last_ns = frame.timestamp_ns
                    if self._first_ns is None:
                        self._first_ns = frame.timestamp_ns
                    if self._stop.is_set():
                        self._discarded += 1
                        break
                    if self._pending is not None:
                        self._replaced += 1
                    self._pending = captured_frame
                    self._condition.notify_all()
        except BaseException as exc:
            with self._condition:
                self._fail(exc)
        finally:
            try:
                self._source.close()
            except BaseException as exc:
                with self._condition:
                    self._fail(exc)
            else:
                with self._condition:
                    self._cleanup_complete = True
            with self._condition:
                if self._state != "failed":
                    self._state = "stopped"
                self._condition.notify_all()

    def read(self) -> CapturedFrame:
        with self._condition:
            if self._state not in {"running", "failed"}:
                raise CameraError(f"Cannot read capture in state {self._state}")
            ready = self._condition.wait_for(
                lambda: self._pending is not None or self._state != "running", self._timeout
            )
            if not ready:
                self._fail(CameraError("Timed out waiting for a camera frame"))
            self.check()
            if self._state != "running" or self._pending is None:
                raise CameraError("Capture stopped before another frame was available")
            frame, self._pending = self._pending, None
            self._delivered += 1
            return frame

    def request_stop(self) -> None:
        with self._condition:
            self._stop.set()
            self._discard_pending()
            if self._state not in {"failed", "stopped"}:
                self._state = "stopping" if self._thread is not None else "stopped"
            self._condition.notify_all()

    def close(self) -> None:
        self.request_stop()
        thread = self._thread
        if thread is not None:
            thread.join(self._timeout)
            if thread.is_alive():
                with self._condition:
                    self._fail(CameraError("Capture shutdown timed out; cleanup is incomplete"))
        self.check()

    def __enter__(self) -> CaptureRuntime:
        self.open()
        return self

    def __exit__(self, _kind: object, exc: BaseException | None, _tb: object) -> None:
        try:
            self.close()
        except CameraError as cleanup_error:
            if exc is None:
                raise
            exc.add_note(str(cleanup_error))
