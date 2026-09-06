"""Bounded detector-next/pose-current pipeline; no skipped or reused observations.

Both sequential and overlap modes use the same dedicated stage owners. One
request per model owner; an opt-in auxiliary pose owner handles disjoint people
of the SAME frame. Source admission is unchanged. Decoding remains caller-owned.
A native call cannot be forcibly cancelled; shutdown waits for it and never
claims cleanup while a model still owns a running call.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any

import numpy as np

from motioncapture.recording import RecordedFrame
from motioncapture.wholebody_catalog import BenchmarkError
from motioncapture.wholebody_fast_input import check_recipe, fast_pose_tensor
from motioncapture.wholebody_onnx import (
    decode_people, decode_pose, detector_tensor, pose_tensor,
)
from motioncapture.wholebody_replay import Release, ReplayCancelled, SourcePacer


@dataclass(frozen=True)
class Detected:
    frame: RecordedFrame
    boxes: np.ndarray
    times: dict[str, float]
    submitted_ns: int
    completed_ns: int
    source_release: Release | None = None


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
            # Intentional pacing cancellation must not mask a native close failure.
            primary = next((e for e in errors if not isinstance(e, ReplayCancelled)), errors[0])
            for secondary in errors:
                if secondary is not primary:
                    primary.add_note(f"Additional stage error: {type(secondary).__name__}")
            raise primary


def detect(session, submitted, frame, pacer=None):
    entered = time.perf_counter_ns()
    release = pacer.wait(frame.identity) if pacer is not None else None
    start = time.perf_counter_ns() if release is not None else entered
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
    times = {
        "detector_queue_ms": (entered-submitted)/1e6,
        "detector_pre_ms": (a-start)/1e6,
        "detector_inference_ms": (b-a)/1e6,
        "detector_post_ms": (end-b)/1e6,
        "detector_stage_ms": (end-start)/1e6,
    }
    if release is not None:
        times["source_pacing_wait_ms"] = release.wait_ms
        times["scheduled_source_to_detector_ms"] = (start-release.due_ns)/1e6
    return Detected(frame, boxes, times, submitted, end, release)


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


def pose_parallel(session, submitted, detected, size, threshold, fast, kernel, auxiliary):
    """Two same-frame lanes; retain every detector slot in its original order.

    Only the primary pose worker touches auxiliary submit/receive. Its caller
    must join the primary before auxiliary close. No concurrent use of a session.
    """
    if len(detected.boxes) < 2:
        return pose(session, submitted, detected, size, threshold, fast, kernel)
    started = time.perf_counter_ns()
    primary_input = replace(detected, boxes=detected.boxes[::2])
    auxiliary_input = replace(detected, boxes=detected.boxes[1::2])
    auxiliary.submit(pose, auxiliary_input, size, threshold, fast, kernel)
    try:
        primary = pose(session, submitted, primary_input, size, threshold, fast, kernel)
    except BaseException as original:
        # Drain the admitted native job before propagating; no orphan or retry.
        try:
            auxiliary.receive()
        except BaseException as secondary:
            original.add_note(f"Auxiliary pose failure: {type(secondary).__name__}")
        raise
    join_started = time.perf_counter_ns()
    other = auxiliary.receive()  # A selected auxiliary failure is NOT a serial fallback.
    join_ms = (time.perf_counter_ns() - join_started) / 1e6
    if primary.detected is not primary_input or other.detected is not auxiliary_input:
        raise BenchmarkError("pose_lane_identity_mismatch")
    people, per_person = [None] * len(detected.boxes), [None] * len(detected.boxes)
    for offset, lane in enumerate((primary, other)):
        expected = len(detected.boxes[offset::2])
        if len(lane.people) != expected or len(lane.per_person_ms) != expected:
            raise BenchmarkError("pose_lane_count_mismatch")
        people[offset::2] = lane.people
        per_person[offset::2] = lane.per_person_ms
    ended = time.perf_counter_ns()
    times = {**detected.times,
             **{key: primary.times[key] + other.times[key]
                for key in ("pose_pre_ms", "pose_inference_ms", "pose_post_ms")},
             "pose_stage_ms": (ended - started) / 1e6,
             "pose_queue_ms": (started - submitted) / 1e6,
             "detector_to_pose_wait_ms": (started - detected.completed_ns) / 1e6,
             "decode_read_ms": detected.frame.decode_ms,
             "submit_to_pose_completion_ms": (ended - detected.submitted_ns) / 1e6,
             "pose_primary_lane_stage_ms": primary.times["pose_stage_ms"],
             "pose_auxiliary_lane_stage_ms": other.times["pose_stage_ms"],
             "pose_auxiliary_join_ms": join_ms}
    times["frame_work_ms"] = (detected.frame.decode_ms + times["detector_stage_ms"]
                              + times["pose_stage_ms"])
    return Posed(detected, people, times, tuple(per_person), ended)


@dataclass
class StagePipeline:
    detector_factory: Callable[[], Any]
    pose_factory: Callable[[], Any]
    size: tuple[int, int] = (192, 256)
    threshold: float = .3
    normalization_kernel: str = "numpy"
    pacer: SourcePacer | None = None
    pose_lanes: int = 1
    auxiliary_pose: StageOwner | None = field(default=None, init=False)
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
        if type(self.pose_lanes) is not int or self.pose_lanes not in (1, 2):
            raise BenchmarkError("unsupported_pose_lane_count")
        self.detector = StageOwner(self.detector_factory, "wholebody-detector")
        self.pose = StageOwner(self.pose_factory, "wholebody-pose")

    def __enter__(self):
        try:
            self.metadata = {"detector": self.detector.open(), "pose": self.pose.open()}
            if self.pose_lanes == 2:
                self.auxiliary_pose = StageOwner(self.pose_factory, "wholebody-pose-auxiliary")
                self.metadata["auxiliary_pose"] = self.auxiliary_pose.open()
            self.metadata["pose_execution"] = {
                "native_sessions": self.pose_lanes,
                "assignment": "even_odd_detector_slots" if self.pose_lanes == 2 else "serial",
                "scope": "within_one_source_frame; slot_is_not_actor_id",
                "native_parallel_speedup_verified": False}
        except BaseException as exc:
            self.__exit__(type(exc), exc, None)
            raise
        return self

    def __exit__(self, _kind, exc, _tb):
        # Wake scheduled-but-unreleased work on error or an early consumer stop.
        if self.pacer is not None and (exc is not None or self.read_frames > self.emitted_frames):
            self.pacer.cancel()
        errors = []
        # Primary may be inside auxiliary.receive(); join it before closing aux.
        for owner in (self.detector, self.pose, self.auxiliary_pose):
            if owner is None:
                continue
            try:
                owner.close()
            except BaseException as failure:
                if not (isinstance(failure, ReplayCancelled) and self.pacer is not None
                        and self.pacer.cancelled):
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
        if self.auxiliary_pose is not None:
            self.auxiliary_pose.check()

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
            if self.auxiliary_pose is None:
                submitted = self.pose.submit(pose, *arguments)
            else:
                submitted = self.pose.submit(
                    pose_parallel, detected, self.size, self.threshold, fast,
                    self.normalization_kernel, self.auxiliary_pose)
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
        def submit_detector(frame):
            arguments = (frame,) if self.pacer is None else (frame, self.pacer)
            self.detector.submit(detect, *arguments)

        first = read_frame()
        # Explicit real-source preflight, NOT counted as tracked people or timed inference.
        if fast:
            h, w = first.image_bgr.shape[:2]
            check_recipe(first.image_bgr, np.array([0., 0., w, h], np.float32), self.size,
                         kernel=self.normalization_kernel)
        submit_detector(first)
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
                submit_detector(upcoming)
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
                submit_detector(read_frame())
        if verify_eof:
            try:
                next(iterator)
            except StopIteration:
                pass
            else:
                raise BenchmarkError("unexpected_extra_source_frame")
        self.check()

    def snapshot(self):
        cleanup = {"detector": self.detector.cleanup, "pose": self.pose.cleanup}
        if self.auxiliary_pose is not None:
            cleanup["auxiliary_pose"] = self.auxiliary_pose.cleanup
        return {"read_frames": self.read_frames, "emitted_frames": self.emitted_frames,
                "max_outstanding_per_stage": 1, "model_stages": 1 + self.pose_lanes,
                "pose_lanes": self.pose_lanes, "frame_parallel_pose": False,
                "intentional_frame_skips": 0,
                "unemitted_read_frames": self.read_frames-self.emitted_frames,
                "detector_cadence": "every_source_frame",
                "cleanup": cleanup,
                "native_hang_force_cancellation": False,
                "pose_requests": self.pose_requests,
                "unemitted_pose_requests": self.pose_requests-self.emitted_frames,
                "ready_pose_handoff_enabled": self.advance_pose,
                "ready_pose_handoffs": self.ready_handoffs,
                "next_detection_not_ready": self.unready_handoffs,
                "ready_handoff_waits_for_detector": False,
                "normalization_kernel": self.normalization_kernel,
                "source_pacing": self.pacer.summary() if self.pacer is not None else None}
