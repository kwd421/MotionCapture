"""Bounded detector-next/pose-current pipeline; no skipped or reused observations.

Both sequential and overlap modes use the same dedicated stage owners. One
pending detector and one pending pose at most. Decoding remains caller-owned.
A native call cannot be forcibly cancelled; shutdown waits for it and never
claims cleanup while a model still owns a running call.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import numpy as np

from motioncapture.recording import RecordedFrame
from motioncapture.wholebody_catalog import BenchmarkError
from motioncapture.wholebody_fast_input import check_recipe, fast_pose_tensor
from motioncapture.wholebody_onnx import (
    decode_people, decode_pose, detector_tensor, pose_tensor,
)


@dataclass(frozen=True)
class Detected:
    frame: RecordedFrame
    boxes: np.ndarray
    times: dict[str, float]
    submitted_ns: int
    completed_ns: int


@dataclass(frozen=True)
class Posed:
    detected: Detected
    people: list
    times: dict[str, float]
    per_person_ms: tuple[float, ...]
    completed_ns: int


class StageOwner:
    """One native session's construction/run/destruction stay on one worker."""
    def __init__(self, factory: Callable[[], Any], name: str):
        self._factory = factory
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)
        self._session = None
        self._future: Future | None = None
        self._closed = False
        self._lock = threading.Lock()
        self._failure: BaseException | None = None
        self.cleanup = "not_opened"
        self.owner_thread = None

    def _record_failure(self, future: Future) -> None:
        if not future.cancelled():
            error = future.exception()
            if error is not None:
                with self._lock:
                    if self._failure is None:
                        self._failure = error

    def _create(self):
        self.owner_thread = threading.get_ident()
        self.cleanup = "unknown_after_constructor_failure"
        self._session = self._factory()
        self.cleanup = "open"
        return self._session.metadata

    def open(self):
        if self._closed or self._session is not None or self._future is not None:
            raise BenchmarkError("stage_cannot_reopen")
        self._future = self._executor.submit(self._create)
        self._future.add_done_callback(self._record_failure)
        return self.receive()

    def submit(self, operation, *args):
        self.check()
        if self._closed or self._session is None:
            raise BenchmarkError("stage_not_open")
        if self._future is not None:
            raise BenchmarkError("stage_outstanding_limit_exceeded")
        submitted = time.perf_counter_ns()
        self._future = self._executor.submit(operation, self._session, submitted, *args)
        self._future.add_done_callback(self._record_failure)
        return submitted

    def take_ready(self):
        """Caller-only nonblocking receive; a completed failure is still an error."""
        self.check()
        if self._future is None:
            raise BenchmarkError("stage_has_no_request")
        if not self._future.done():
            return False, None
        return True, self.receive()

    def receive(self):
        if self._future is None:
            raise BenchmarkError("stage_has_no_request")
        future = self._future
        try:
            return future.result()
        finally:
            self._future = None

    def check(self):
        with self._lock:
            error = self._failure
        if error is not None:
            raise error

    def _destroy(self):
        if self._session is not None:
            session, self._session = self._session, None
            try:
                session.close()
            except BaseException:
                self.cleanup = "failed"
                raise
            else:
                self.cleanup = "owner_released"

    def close(self):
        if self._closed:
            return
        self._closed = True
        errors = []
        if self._future is not None:
            try:
                self.receive()
            except BaseException as exc:
                errors.append(exc)
        try:
            self._executor.submit(self._destroy).result()
        except BaseException as exc:
            errors.append(exc)
        finally:
            self._executor.shutdown(wait=True, cancel_futures=True)
        if errors:
            raise errors[0]


def detect(session, submitted, frame):
    start = time.perf_counter_ns()
    tensor, ratio = detector_tensor(frame.image_bgr)
    a = time.perf_counter_ns()
    outputs = session.run(tensor)
    b = time.perf_counter_ns()
    if len(outputs) != 1:
        raise BenchmarkError("invalid_detector_outputs")
    boxes = decode_people(outputs[0], ratio)
    if len(boxes) > 8:
        raise BenchmarkError("person_capacity_exceeded")
    boxes.flags.writeable = False
    end = time.perf_counter_ns()
    return Detected(frame, boxes, {
        "detector_queue_ms": (start-submitted)/1e6,
        "detector_pre_ms": (a-start)/1e6,
        "detector_inference_ms": (b-a)/1e6,
        "detector_post_ms": (end-b)/1e6,
        "detector_stage_ms": (end-start)/1e6,
    }, submitted, end)


def pose(session, submitted, detected, size, threshold, fast, kernel="numpy"):
    start = time.perf_counter_ns()
    pre = inference = post = 0.0
    people, per_person = [], []
    prepare = partial(fast_pose_tensor, kernel=kernel) if fast else pose_tensor
    for box in detected.boxes:
        began = time.perf_counter_ns()
        tensor, center, scale = prepare(detected.frame.image_bgr, box, size)
        a = time.perf_counter_ns()
        outputs = session.run(tensor)
        b = time.perf_counter_ns()
        people.append(decode_pose(outputs, size, center, scale, threshold))
        c = time.perf_counter_ns()
        pre += (a-began)/1e6
        inference += (b-a)/1e6
        post += (c-b)/1e6
        per_person.append((b-a)/1e6)
    end = time.perf_counter_ns()
    times = {**detected.times, "pose_pre_ms": pre, "pose_inference_ms": inference,
             "pose_post_ms": post, "pose_stage_ms": (end-start)/1e6,
             "pose_queue_ms": (start-submitted)/1e6,
             "detector_to_pose_wait_ms": (start-detected.completed_ns)/1e6,
             "decode_read_ms": detected.frame.decode_ms,
             "submit_to_pose_completion_ms": (end-detected.submitted_ns)/1e6}
    times["frame_work_ms"] = (detected.frame.decode_ms + times["detector_stage_ms"]
                              + times["pose_stage_ms"])
    return Posed(detected, people, times, tuple(per_person), end)


@dataclass
class StagePipeline:
    detector_factory: Callable[[], Any]
    pose_factory: Callable[[], Any]
    size: tuple[int, int] = (192, 256)
    threshold: float = .3
    normalization_kernel: str = "numpy"
    detector: StageOwner = field(init=False)
    pose: StageOwner = field(init=False)
    metadata: dict = field(default_factory=dict, init=False)
    read_frames: int = field(default=0, init=False)
    emitted_frames: int = field(default=0, init=False)
    used: bool = field(default=False, init=False)
    advance_pose: bool = field(default=False, init=False)
    pose_requests: int = field(default=0, init=False)
    ready_handoffs: int = field(default=0, init=False)
    unready_handoffs: int = field(default=0, init=False)

    def __post_init__(self):
        if self.normalization_kernel not in {"numpy", "opencv"}:
            raise BenchmarkError("unknown_normalization_kernel")
        self.detector = StageOwner(self.detector_factory, "wholebody-detector")
        self.pose = StageOwner(self.pose_factory, "wholebody-pose")

    def __enter__(self):
        try:
            self.metadata = {"detector": self.detector.open(), "pose": self.pose.open()}
        except BaseException as exc:
            self.__exit__(type(exc), exc, None)
            raise
        return self

    def __exit__(self, _kind, exc, _tb):
        errors = []
        for owner in (self.detector, self.pose):
            try:
                owner.close()
            except BaseException as failure:
                errors.append(failure)
        if errors:
            if exc is not None:
                for error in errors:
                    exc.add_note(f"Stage cleanup: {type(error).__name__}")
            else:
                raise errors[0]

    def check(self):
        self.detector.check()
        self.pose.check()

    def packets(self, frames: Iterable[RecordedFrame], count: int, *, overlap: bool,
                fast: bool, verify_eof: bool = False, advance_pose: bool = False):
        if (self.used or count <= 0 or (advance_pose and not overlap)
                or (not fast and self.normalization_kernel != "numpy")):
            raise BenchmarkError("invalid_pipeline_run")
        self.used = True
        self.advance_pose = advance_pose
        last_pose_end = None
        prestarted = None

        def submit_pose(detected, index):
            if detected.frame.identity.sequence != index:
                raise BenchmarkError("pipeline_detection_identity_mismatch")
            arguments = (detected, self.size, self.threshold, fast)
            if self.normalization_kernel != "numpy":
                arguments += (self.normalization_kernel,)
            submitted = self.pose.submit(pose, *arguments)
            self.pose_requests += 1
            # First request has no predecessor; omit that sample rather than inventing 0.
            return None if last_pose_end is None else (submitted-last_pose_end)/1e6

        iterator = iter(frames)
        def read_frame():
            self.check()
            frame = next(iterator)
            if frame.identity.sequence != self.read_frames:
                raise BenchmarkError("pipeline_frame_sequence_mismatch")
            self.read_frames += 1
            return frame
        first = read_frame()
        # Explicit real-source preflight, NOT counted as tracked people or timed inference.
        if fast:
            h, w = first.image_bgr.shape[:2]
            check_recipe(first.image_bgr, np.array([0., 0., w, h], np.float32), self.size,
                         kernel=self.normalization_kernel)
        self.detector.submit(detect, first)
        del first
        for index in range(count):
            if prestarted is None:
                detected = self.detector.receive()
                submit_gap = submit_pose(detected, index)
            else:
                detected, submit_gap = prestarted
                prestarted = None
            if overlap and index+1 < count:
                upcoming = read_frame()
                self.detector.submit(detect, upcoming)
                del upcoming
            packet = self.pose.receive()
            self.check()  # Do not emit after an observed later-frame failure.
            if packet.detected is not detected:
                raise BenchmarkError("pipeline_pose_identity_mismatch")
            last_pose_end = packet.completed_ns
            if submit_gap is not None:
                packet.times["pose_submit_gap_ms"] = submit_gap
            # Admit no extra source frame and never wait here. In the common
            # pose-bound case the next detection has already completed.
            if advance_pose and index+1 < count:
                ready, upcoming_detection = self.detector.take_ready()
                if ready:
                    next_gap = submit_pose(upcoming_detection, index+1)
                    prestarted = (upcoming_detection, next_gap)
                    self.ready_handoffs += 1
                else:
                    self.unready_handoffs += 1
                del upcoming_detection
                self.check()
            packet.times["result_residence_ms"] = (
                time.perf_counter_ns()-packet.completed_ns)/1e6
            self.emitted_frames += 1
            yield packet
            del packet, detected
            if not overlap and index+1 < count:
                self.detector.submit(detect, read_frame())
        if verify_eof:
            try:
                next(iterator)
            except StopIteration:
                pass
            else:
                raise BenchmarkError("unexpected_extra_source_frame")
        self.check()

    def snapshot(self):
        return {"read_frames": self.read_frames, "emitted_frames": self.emitted_frames,
                "max_outstanding_per_stage": 1, "model_stages": 2,
                "intentional_frame_skips": 0,
                "unemitted_read_frames": self.read_frames-self.emitted_frames,
                "detector_cadence": "every_source_frame",
                "cleanup": {"detector": self.detector.cleanup, "pose": self.pose.cleanup},
                "native_hang_force_cancellation": False,
                "pose_requests": self.pose_requests,
                "unemitted_pose_requests": self.pose_requests-self.emitted_frames,
                "ready_pose_handoff_enabled": self.advance_pose,
                "ready_pose_handoffs": self.ready_handoffs,
                "next_detection_not_ready": self.unready_handoffs,
                "ready_handoff_waits_for_detector": False,
                "normalization_kernel": self.normalization_kernel}
