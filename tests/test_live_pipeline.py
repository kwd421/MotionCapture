"""Synthetic, event-controlled tests. No native inference/hardware claims."""
from __future__ import annotations

import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from motioncapture.capture import CameraObservation
from motioncapture.contracts import CapturedFrame, FrameIdentity, LandmarkResult, LandmarkTimings
from motioncapture.errors import CameraError, InferenceError
from motioncapture.pipeline import FramePipeline
from motioncapture.runtime import CaptureSnapshot

EMPTY = LandmarkResult((), (), (), (), (), (), (), ())
TIMINGS = LandmarkTimings(0.1, 1, 2, 1, 2, 0.1, 2.2)


class FakeCapture:
    def __init__(self):
        self.frames = 0
        self.stopped = threading.Event()
        self.opened = False
        self.cleaned = False
        self.error = None
        self.read_entered = threading.Event()
        self.block_read = False

    def open(self):
        self.opened = True
        return CameraObservation("test", 0, 16, 12, 30)

    def read(self):
        self.read_entered.set()
        if self.block_read:
            assert self.stopped.wait(2)
            raise CameraError("stopped intentionally")
        self.check()
        index = self.frames
        self.frames += 1
        return CapturedFrame(FrameIdentity("test", "stream", index, time.monotonic_ns()),
                             np.full((12, 16, 3), index % 256, dtype=np.uint8))

    def check(self):
        if self.error is not None:
            raise self.error

    def request_stop(self):
        self.stopped.set()

    def close(self):
        self.request_stop()
        self.cleaned = True
        self.check()

    def snapshot(self):
        return CaptureSnapshot(
            "stopped" if self.cleaned else "running", "stream", self.frames,
            self.frames, 0, 0, 0, None, None, None, self.cleaned,
            str(self.error) if self.error else None,
        )


class FakeTracker:
    provider_name = "synthetic-test-tracker"

    def __init__(self):
        self.calls = []
        self.processed = []
        self.started = threading.Event()
        self.second_started = threading.Event()
        self.finished = threading.Event()
        self.release = threading.Event()
        self.block = False
        self.failure = None
        self.close_failure = None
        self.bad_identity = False

    def open(self):
        self.calls.append(("open", threading.get_ident()))

    def process(self, frame):
        self.calls.append(("process", threading.get_ident()))
        self.processed.append(frame.identity.sequence)
        self.started.set()
        if len(self.processed) == 2:
            self.second_started.set()
        if self.block:
            assert self.release.wait(3)
        if self.failure:
            raise self.failure
        identity = frame.identity
        if self.bad_identity:
            identity = replace(identity, sequence=identity.sequence + 1)
        return SimpleNamespace(identity=identity, result=EMPTY, timings=TIMINGS,
                               model_timestamp_ms=frame.identity.sequence)

    def close(self):
        self.calls.append(("close", threading.get_ident()))
        self.finished.set()
        if self.close_failure:
            raise self.close_failure


def test_tracker_lifecycle_has_one_non_gui_owner_and_no_unrequested_work():
    capture, tracker = FakeCapture(), FakeTracker()
    with FramePipeline(capture, lambda: tracker) as pipeline:
        assert capture.frames == 0
        for expected in range(3):
            pipeline.request()
            packet = pipeline.receive()
            assert packet.frame.identity.sequence == expected
            assert packet.frame.identity == packet.output.identity
            assert packet.dequeued_ns <= packet.completed_ns
            assert packet.capture_queue_ms >= 0
        assert pipeline.snapshot()["delivered"] == 3
    ids = {ident for _, ident in tracker.calls}
    assert len(ids) == 1 and threading.get_ident() not in ids
    assert tracker.calls[0][0] == "open" and tracker.calls[-1][0] == "close"
    assert pipeline.snapshot()["cleanup_complete"] is True
    assert capture.frames == 3
    pipeline.close()
    with pytest.raises(InferenceError, match="reopened"):
        pipeline.open()


def test_only_one_outstanding_request_even_when_caller_is_slow():
    capture, tracker = FakeCapture(), FakeTracker()
    tracker.block = True
    with FramePipeline(capture, lambda: tracker) as pipeline:
        pipeline.request()
        assert tracker.started.wait(1)
        with pytest.raises(InferenceError, match="one inference"):
            pipeline.request()
        tracker.release.set()
        packet = pipeline.receive()
        assert packet.frame.identity.sequence == 0
        assert capture.frames == 1
        with pytest.raises(InferenceError, match="No inference"):
            pipeline.receive()


def test_next_inference_can_run_while_caller_holds_previous_packet():
    capture, tracker = FakeCapture(), FakeTracker()
    with FramePipeline(capture, lambda: tracker) as pipeline:
        pipeline.request()
        first = pipeline.receive()
        pipeline.request()
        assert tracker.second_started.wait(1)
        # Caller has not requested another receive or released the old pixels.
        assert first.frame.image_bgr[0, 0, 0] == 0
        second = pipeline.receive()
        assert second.frame.image_bgr[0, 0, 0] == 1
        assert not np.shares_memory(first.frame.image_bgr, second.frame.image_bgr)


@pytest.mark.parametrize("problem", [RuntimeError("inference broke"), ValueError("invalid result")])
def test_inference_failure_not_fallback_or_cached_result(problem):
    tracker = FakeTracker()
    tracker.failure = problem
    pipeline = FramePipeline(FakeCapture(), lambda: tracker)
    with pytest.raises(InferenceError, match="Pipeline failed"):
        with pipeline:
            pipeline.request()
            pipeline.receive()
    assert tracker.finished.is_set()
    assert pipeline.snapshot()["state"] == "failed"
    assert pipeline.snapshot()["delivered"] == 0


def test_identity_mismatch_is_fatal():
    tracker = FakeTracker()
    tracker.bad_identity = True
    pipeline = FramePipeline(FakeCapture(), lambda: tracker)
    with pytest.raises(InferenceError, match="identity mismatch"):
        with pipeline:
            pipeline.request()
            pipeline.receive()


def test_camera_failure_does_not_publish_previous_success():
    capture, tracker = FakeCapture(), FakeTracker()
    pipeline = FramePipeline(capture, lambda: tracker)
    pipeline.open()
    pipeline.request()
    pipeline.receive()
    capture.error = CameraError("disconnected")
    with pytest.raises(CameraError, match="disconnected"):
        pipeline.request()
    with pytest.raises(InferenceError, match="disconnected"):
        pipeline.close()
    assert pipeline.snapshot()["delivered"] == 1
    assert tracker.finished.is_set()


def test_shutdown_during_capture_wait_is_intentional_cancellation():
    capture, tracker = FakeCapture(), FakeTracker()
    capture.block_read = True
    pipeline = FramePipeline(capture, lambda: tracker)
    pipeline.open()
    pipeline.request()
    assert capture.read_entered.wait(1)
    pipeline.close()
    state = pipeline.snapshot()
    assert state["state"] == "stopped" and state["cleanup_complete"] is True
    assert state["inferences_completed"] == 0
    assert tracker.processed == []


def test_shutdown_does_not_close_tracker_during_inference():
    capture, tracker = FakeCapture(), FakeTracker()
    tracker.block = True
    pipeline = FramePipeline(capture, lambda: tracker)
    pipeline.open()
    pipeline.request()
    assert tracker.started.wait(1)
    errors = []

    def close():
        try:
            pipeline.close()
        except BaseException as exc:
            errors.append(exc)

    closer = threading.Thread(target=close)
    closer.start()
    assert capture.stopped.wait(1)
    assert not tracker.finished.is_set()
    tracker.release.set()
    closer.join(2)
    assert not closer.is_alive() and not errors
    state = pipeline.snapshot()
    assert state["inferences_completed"] == 1
    assert state["discarded"] == 1 and state["delivered"] == 0
    assert state["cleanup_complete"] is True


def test_timeout_is_failed_not_clean_and_cleanup_remains_owned():
    capture, tracker = FakeCapture(), FakeTracker()
    tracker.block = True
    pipeline = FramePipeline(capture, lambda: tracker, timeout=0.05)
    pipeline.open()
    pipeline.request()
    assert tracker.started.wait(1)
    with pytest.raises(InferenceError, match="shutdown timed out"):
        pipeline.close()
    assert pipeline.snapshot()["cleanup_complete"] is False
    assert not tracker.finished.is_set()
    tracker.release.set()
    assert tracker.finished.wait(1)
    with pytest.raises(InferenceError, match="shutdown timed out"):
        pipeline.close()
    assert pipeline.snapshot()["cleanup_complete"] is True


def test_cleanup_failure_is_reported():
    tracker = FakeTracker()
    tracker.close_failure = RuntimeError("native close failed")
    pipeline = FramePipeline(FakeCapture(), lambda: tracker)
    with pytest.raises(InferenceError, match="native close failed"):
        with pipeline:
            pass
    assert pipeline.snapshot()["cleanup_complete"] is False


def test_factory_failure_terminates_and_does_not_open_camera():
    capture = FakeCapture()

    def fail():
        raise RuntimeError("missing model")

    pipeline = FramePipeline(capture, fail)
    with pytest.raises(InferenceError, match="missing model"):
        pipeline.open()
    assert not capture.opened


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_bad_timeouts(timeout):
    with pytest.raises(ValueError):
        FramePipeline(FakeCapture(), FakeTracker, timeout=timeout)


def test_real_capture_runtime_and_pipeline_keep_separate_resource_owners():
    from queue import Queue

    from motioncapture.capture import CameraFrame
    from motioncapture.runtime import CaptureRuntime

    commands = Queue()
    source_calls = []
    third_read = threading.Event()

    class Source:
        def open(self):
            source_calls.append(("open", threading.get_ident()))
            return CameraObservation("fixture", 0, 16, 12, 30)

        def read(self):
            source_calls.append(("read", threading.get_ident()))
            if sum(name == "read" for name, _ in source_calls) == 3:
                third_read.set()
            index = commands.get(timeout=2)
            return CameraFrame(index, time.monotonic_ns(), np.zeros((12, 16, 3), np.uint8))

        def close(self):
            source_calls.append(("close", threading.get_ident()))

    capture, tracker = CaptureRuntime(Source(), timeout=1), FakeTracker()
    pipeline = FramePipeline(capture, lambda: tracker, timeout=2)
    pipeline.open()
    for index in range(2):
        commands.put(index)
        pipeline.request()
        assert pipeline.receive().frame.identity.sequence == index
    assert third_read.wait(1)
    # Request camera stop before releasing the native-read test double. The
    # real CaptureRuntime must count the last returned frame as discarded.
    capture.request_stop()
    commands.put(2)
    pipeline.close()
    state = capture.snapshot()
    assert state.captured == 3 and state.delivered == 2 and state.discarded == 1
    assert state.cleanup_complete is True
    source_owners = {ident for _, ident in source_calls}
    tracker_owners = {ident for _, ident in tracker.calls}
    assert len(source_owners) == len(tracker_owners) == 1
    assert not source_owners & tracker_owners
    assert threading.get_ident() not in source_owners | tracker_owners
