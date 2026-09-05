"""Controlled camera events are test doubles, not a live fallback source."""

import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import numpy as np
import pytest

from motioncapture.capture import CameraFrame, CameraObservation
from motioncapture.errors import CameraError
from motioncapture.runtime import CaptureRuntime


class ControlledCamera:
    def __init__(self, *, open_error: Exception | None = None) -> None:
        self.commands: Queue[CameraFrame | Exception] = Queue()
        self.read_started: Queue[int] = Queue()
        self.closed = threading.Event()
        self.open_error = open_error
        self.owner_ids: list[int] = []
        self.reads = 0
        self.close_count = 0

    def open(self) -> CameraObservation:
        self.owner_ids.append(threading.get_ident())
        if self.open_error is not None:
            raise self.open_error
        return CameraObservation("test", 0, 2, 2, 30.0)

    def read(self) -> CameraFrame:
        self.owner_ids.append(threading.get_ident())
        self.reads += 1
        self.read_started.put(self.reads)
        command = self.commands.get(timeout=5)
        if isinstance(command, Exception):
            raise command
        return command

    def close(self) -> None:
        self.owner_ids.append(threading.get_ident())
        self.close_count += 1
        self.closed.set()

    def wait_read(self, count: int) -> None:
        assert self.read_started.get(timeout=2) == count


def frame(sequence: int, *, timestamp_ns: int | None = None) -> CameraFrame:
    return CameraFrame(
        sequence,
        (sequence + 1) * 10_000_000 if timestamp_ns is None else timestamp_ns,
        np.zeros((2, 2, 3), dtype=np.uint8),
    )


def finish(runtime: CaptureRuntime, camera: ControlledCamera, sequence: int = 100) -> None:
    # Stop before releasing the in-flight read. No sleeps or timing assumptions.
    runtime.request_stop()
    camera.commands.put(frame(sequence))
    runtime.close()
    assert camera.closed.is_set()


def test_slow_consumer_receives_newest_frame_and_accounts_for_every_frame() -> None:
    camera = ControlledCamera()
    runtime = CaptureRuntime(camera)
    runtime.open()
    camera.wait_read(1)
    try:
        for sequence in range(3):
            camera.commands.put(frame(sequence))
            camera.wait_read(sequence + 2)
        snapshot = runtime.snapshot()
        assert (snapshot.captured, snapshot.replaced, snapshot.pending) == (3, 2, 1)
        selected = runtime.read()
        assert selected.identity.sequence == 2
        assert selected.identity.received_ns == 30_000_000
        assert selected.identity.source_id == "local:test:0"
        assert selected.identity.stream_id == snapshot.stream_id
        assert runtime.snapshot().mean_capture_fps == 100.0
    finally:
        finish(runtime, camera)
    snapshot = runtime.snapshot()
    assert snapshot.state == "stopped"
    assert snapshot.cleanup_complete is True
    assert snapshot.pending == 0
    assert snapshot.captured == snapshot.delivered + snapshot.replaced + snapshot.discarded
    assert camera.close_count == 1
    assert len(set(camera.owner_ids)) == 1
    assert camera.owner_ids[0] != threading.get_ident()


def test_camera_failure_discards_pending_frame_instead_of_replaying_it() -> None:
    camera = ControlledCamera()
    runtime = CaptureRuntime(camera)
    runtime.open()
    camera.wait_read(1)
    camera.commands.put(frame(0))
    camera.wait_read(2)
    camera.commands.put(CameraError("device unplugged"))
    assert camera.closed.wait(2)
    with pytest.raises(CameraError, match="device unplugged"):
        runtime.read()
    with pytest.raises(CameraError, match="device unplugged"):
        runtime.close()
    snapshot = runtime.snapshot()
    assert snapshot.state == "failed"
    assert (snapshot.pending, snapshot.discarded, snapshot.delivered) == (0, 1, 0)
    assert snapshot.cleanup_complete


def test_open_failure_does_not_open_an_alternative_and_is_cleaned_up() -> None:
    camera = ControlledCamera(open_error=CameraError("permission denied"))
    runtime = CaptureRuntime(camera)
    with pytest.raises(CameraError, match="permission denied"):
        runtime.open()
    assert camera.reads == 0
    assert camera.close_count == 1
    assert runtime.snapshot().state == "failed"
    assert runtime.snapshot().cleanup_complete


@pytest.mark.parametrize("sequence,timestamp", [(0, 20_000_000), (1, 10_000_000)])
def test_duplicate_sequence_or_timestamp_is_terminal(sequence: int, timestamp: int) -> None:
    camera = ControlledCamera()
    runtime = CaptureRuntime(camera)
    runtime.open()
    camera.wait_read(1)
    camera.commands.put(frame(0))
    camera.wait_read(2)
    camera.commands.put(frame(sequence, timestamp_ns=timestamp))
    assert camera.closed.wait(2)
    with pytest.raises(CameraError, match="Non-monotonic"):
        runtime.read()
    with pytest.raises(CameraError):
        runtime.close()
    assert runtime.snapshot().delivered == 0


def test_stop_wakes_waiting_consumer_and_closes_on_camera_owner_thread() -> None:
    camera = ControlledCamera()
    runtime = CaptureRuntime(camera)
    runtime.open()
    camera.wait_read(1)
    with ThreadPoolExecutor(max_workers=1) as pool:
        waiting = pool.submit(runtime.read)
        runtime.request_stop()
        with pytest.raises(CameraError, match="stopp"):
            waiting.result(timeout=2)
        camera.commands.put(frame(0))
        runtime.close()
    assert runtime.snapshot().pending == 0
    assert camera.close_count == 1


def test_shutdown_timeout_remains_failed_until_even_after_delayed_cleanup() -> None:
    camera = ControlledCamera()
    runtime = CaptureRuntime(camera, timeout=0.05)
    runtime.open()
    camera.wait_read(1)
    try:
        with pytest.raises(CameraError, match="cleanup is incomplete"):
            runtime.close()
        snapshot = runtime.snapshot()
        assert snapshot.state == "failed"
        assert snapshot.cleanup_complete is False
    finally:
        camera.commands.put(frame(0))
        assert camera.closed.wait(2)
        with pytest.raises(CameraError):
            runtime.close()
    assert runtime.snapshot().cleanup_complete is True
    assert runtime.snapshot().state == "failed"


def test_read_timeout_is_failure_not_a_missing_detection() -> None:
    camera = ControlledCamera()
    runtime = CaptureRuntime(camera, timeout=0.05)
    runtime.open()
    camera.wait_read(1)
    try:
        with pytest.raises(CameraError, match="waiting for a camera frame"):
            runtime.read()
        assert runtime.snapshot().state == "failed"
    finally:
        camera.commands.put(frame(0))
        assert camera.closed.wait(2)
        with pytest.raises(CameraError):
            runtime.close()


def test_new_lifecycle_has_a_new_stream_and_an_old_runtime_cannot_reopen() -> None:
    streams = []
    for _ in range(2):
        camera = ControlledCamera()
        runtime = CaptureRuntime(camera)
        runtime.open()
        camera.wait_read(1)
        streams.append(runtime.snapshot().stream_id)
        finish(runtime, camera)
        with pytest.raises(CameraError, match="twice"):
            runtime.open()
    assert streams[0] != streams[1]


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_timeout_rejected_before_open(timeout: float) -> None:
    camera = ControlledCamera()
    with pytest.raises(ValueError):
        CaptureRuntime(camera, timeout=timeout)
    assert camera.owner_ids == []


def test_invalid_camera_buffer_fails_without_inventing_a_valid_frame() -> None:
    camera = ControlledCamera()
    runtime = CaptureRuntime(camera)
    runtime.open()
    camera.wait_read(1)
    camera.commands.put(CameraFrame(0, 1, np.zeros((2, 2), dtype=np.uint8)))
    assert camera.closed.wait(2)
    with pytest.raises(CameraError, match="BGR"):
        runtime.read()
    with pytest.raises(CameraError):
        runtime.close()
    assert runtime.snapshot().captured == 0


def test_camera_close_failure_is_not_reported_as_clean_shutdown() -> None:
    class FailingCloseCamera(ControlledCamera):
        def close(self) -> None:
            super().close()
            raise CameraError("release failed")

    camera = FailingCloseCamera()
    runtime = CaptureRuntime(camera)
    runtime.open()
    camera.wait_read(1)
    runtime.request_stop()
    camera.commands.put(frame(0))
    with pytest.raises(CameraError, match="release failed"):
        runtime.close()
    assert runtime.snapshot().state == "failed"
    assert runtime.snapshot().cleanup_complete is False
