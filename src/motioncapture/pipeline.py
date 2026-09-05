"""One-request-ahead inference; GUI work never owns the tracker lifecycle."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from motioncapture.capture import CameraObservation
from motioncapture.contracts import CapturedFrame, FrameIdentity, LandmarkResult, LandmarkTimings
from motioncapture.errors import CameraError, InferenceError
from motioncapture.runtime import CaptureRuntime


class TrackingOutput(Protocol):
    identity: FrameIdentity
    model_timestamp_ms: int
    result: LandmarkResult
    timings: LandmarkTimings


class Tracker(Protocol):
    @property
    def provider_name(self) -> str: ...
    def open(self) -> None: ...
    def process(self, frame: CapturedFrame) -> TrackingOutput: ...
    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class TrackedFrame:
    frame: CapturedFrame
    output: TrackingOutput
    dequeued_ns: int
    completed_ns: int
    capture_wait_ms: float

    @property
    def capture_queue_ms(self) -> float:
        return (self.dequeued_ns - self.frame.identity.received_ns) / 1_000_000.0


class _Cancelled(Exception):
    """Internal, intentional stop, not a substitute for tracker/camera failure."""


class FramePipeline:
    """One caller requests/receives; one worker owns every tracker operation.

    A slow GUI can hold ONE completed result. This is bounded prefetch, not an
    unlimited producer or latest-only result dropping. Frame age is measured.
    """

    def __init__(
        self, capture: CaptureRuntime, tracker_factory: Callable[[], Tracker],
        *, timeout: float = 10.0,
    ) -> None:
        if not math.isfinite(timeout) or not 0 < timeout <= threading.TIMEOUT_MAX:
            raise ValueError("Pipeline timeout must be positive, finite and supported")
        self.capture = capture
        self._factory = tracker_factory
        self._timeout = timeout
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None
        self._state = "new"
        self._stop = False
        self._requested = False
        self._in_flight = False
        self._pending: TrackedFrame | None = None
        self._error: BaseException | None = None
        self._camera_ready = False
        self._observation: CameraObservation | None = None
        self._provider: str | None = None
        self._cleanup_complete = False
        self._frames_read = self._completed = self._delivered = self._discarded = 0

    @property
    def observation(self) -> CameraObservation:
        with self._condition:
            if self._observation is None:
                raise InferenceError("Pipeline has no camera observation")
            return self._observation

    @property
    def provider_name(self) -> str:
        with self._condition:
            if self._provider is None:
                raise InferenceError("Pipeline has no initialized tracker")
            return self._provider

    def snapshot(self) -> dict[str, object]:
        with self._condition:
            return {
                "state": self._state,
                "frames_read": self._frames_read,
                "inferences_completed": self._completed,
                "delivered": self._delivered,
                "discarded": self._discarded,
                "pending": int(self._pending is not None),
                "requested": self._requested,
                "in_flight": self._in_flight,
                "max_outstanding": 1,
                "cleanup_complete": self._cleanup_complete,
                "terminal_error": str(self._error) if self._error is not None else None,
            }

    def _fail(self, error: BaseException) -> None:
        with self._condition:
            if self._error is None:
                self._error = error
            elif error is not self._error:
                self._error.add_note(
                    f"Additional pipeline failure: {type(error).__name__}: {error}",
                )
            self._state = "failed"
            self._stop = True
            if self._pending is not None:
                self._pending = None
                self._discarded += 1
            self._condition.notify_all()

    def check(self) -> None:
        with self._condition:
            if self._error is not None:
                raise InferenceError(f"Pipeline failed: {self._error}") from self._error
        self.capture.check()

    def open(self) -> None:
        with self._condition:
            if self._state != "new":
                raise InferenceError("Pipeline cannot be reopened")
            self._state = "starting"
            self._thread = threading.Thread(target=self._run, name="frame-pipeline", daemon=True)
            try:
                self._thread.start()
            except BaseException as exc:
                self._thread = None
                self._fail(exc)
                raise
        try:
            with self._condition:
                ready = self._condition.wait_for(lambda: self._state != "starting", self._timeout)
                if not ready:
                    self._fail(InferenceError("Pipeline startup timed out"))
                self.check()
                if self._state != "running":
                    raise InferenceError("Pipeline startup cancelled")
        except BaseException as exc:
            try:
                self.close()
            except Exception as cleanup_error:
                exc.add_note(str(cleanup_error))
            raise

    def request(self) -> None:
        with self._condition:
            self.check()
            if self._state != "running":
                raise InferenceError("Pipeline is not running")
            if self._requested or self._in_flight or self._pending is not None:
                raise InferenceError("Only one inference request may be outstanding")
            self._requested = True
            self._condition.notify_all()

    def receive(self) -> TrackedFrame:
        with self._condition:
            if not (self._requested or self._in_flight or self._pending is not None):
                self.check()
                raise InferenceError("No inference request is outstanding")
            ready = self._condition.wait_for(
                lambda: self._pending is not None or self._state != "running", self._timeout,
            )
            if not ready:
                self._fail(InferenceError("Timed out waiting for inference"))
            self.check()
            if self._pending is None:
                raise InferenceError("Pipeline stopped before delivery")
            packet, self._pending = self._pending, None
            self._delivered += 1
            return packet

    def _next(self, tracker: Tracker) -> TrackedFrame:
        waiting_ns = time.monotonic_ns()
        try:
            frame = self.capture.read()
        except CameraError:
            with self._condition:
                stopping = self._stop
            if stopping:
                self.capture.check()  # A genuine capture failure is never suppressed.
                raise _Cancelled from None
            raise
        dequeued_ns = time.monotonic_ns()
        with self._condition:
            self._frames_read += 1
            if self._stop:
                raise _Cancelled
        output = tracker.process(frame)
        completed_ns = time.monotonic_ns()
        if output.identity != frame.identity:
            raise InferenceError("Tracker output/frame identity mismatch")
        self.capture.check()
        return TrackedFrame(
            frame, output, dequeued_ns, completed_ns,
            (dequeued_ns - waiting_ns) / 1_000_000.0,
        )

    def _run(self) -> None:
        tracker: Tracker | None = None
        camera_attempted = False
        cleanup_errors: list[BaseException] = []
        try:
            tracker = self._factory()
            tracker.open()
            with self._condition:
                if self._stop:
                    raise _Cancelled
            camera_attempted = True
            observation = self.capture.open()
            with self._condition:
                self._camera_ready = True
                if self._stop:
                    raise _Cancelled
                self._observation = observation
                self._provider = tracker.provider_name
                self._state = "running"
                self._condition.notify_all()
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._requested or self._stop)
                    if self._stop:
                        break
                    self._requested = False
                    self._in_flight = True
                packet = self._next(tracker)
                with self._condition:
                    self._in_flight = False
                    self._completed += 1
                    if self._stop:
                        self._discarded += 1
                    else:
                        self._pending = packet
                    self._condition.notify_all()
                del packet  # Do not retain the previous image while waiting for another request.
        except _Cancelled:
            pass
        except BaseException as exc:
            self._fail(exc)
        finally:
            if camera_attempted:
                try:
                    self.capture.close()
                except BaseException as exc:
                    cleanup_errors.append(exc)
            if tracker is not None:
                try:
                    tracker.close()
                except BaseException as exc:
                    cleanup_errors.append(exc)
            for error in cleanup_errors:
                self._fail(error)
            with self._condition:
                self._in_flight = self._requested = False
                self._cleanup_complete = not cleanup_errors and (
                    not camera_attempted or self.capture.snapshot().cleanup_complete
                )
                if self._state != "failed":
                    self._state = "stopped"
                self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._stop = True
            if self._state not in {"failed", "stopped"}:
                self._state = "stopping" if self._thread is not None else "stopped"
            if self._pending is not None:
                self._pending = None
                self._discarded += 1
            camera_ready = self._camera_ready
            self._condition.notify_all()
        if camera_ready:
            self.capture.request_stop()
        if self._thread is not None:
            self._thread.join(self._timeout)
            if self._thread.is_alive():
                self._fail(InferenceError("Pipeline shutdown timed out; cleanup incomplete"))
        self.check()

    def __enter__(self) -> FramePipeline:
        self.open()
        return self

    def __exit__(self, _kind: object, exc: BaseException | None, _tb: object) -> None:
        try:
            self.close()
        except Exception as cleanup_error:
            if exc is None:
                raise
            exc.add_note(str(cleanup_error))
