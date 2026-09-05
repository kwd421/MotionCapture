from types import SimpleNamespace

import pytest

from motioncapture.capture import CameraRequest
from motioncapture.contracts import FrameIdentity, LandmarkTimings
from motioncapture.errors import SessionRecordError
from motioncapture.runtime import CaptureSnapshot
from motioncapture.session import SessionRecord


def test_session_reports_explicit_stage_latency() -> None:
    session = SessionRecord(
        CameraRequest(index=0, width=1280, height=720, fps=30.0),
        True,
        inference_provider="test-provider",
        task_scheduling="serial",
    )
    result = SimpleNamespace(
        pose_landmarks=[object()],
        face_landmarks=[object()],
        left_hand_landmarks=[],
        right_hand_landmarks=[object()],
    )
    timings = LandmarkTimings(
        input_conversion_ms=1.0,
        pose_ms=2.0,
        hands_ms=3.0,
        face_ms=4.0,
        inference_wall_ms=9.0,
        result_assembly_ms=0.5,
        total_ms=10.5,
    )
    identity = FrameIdentity("local:test:0", "stream", 5, 1_000_000_000)
    session.observe(
        result,
        timings,
        preview_composition_ms=5.0,
        host_post_receive_total_ms=18.5,
        frame_timestamp_ns=1_000_000_000,
        capture_queue_ms=3.0,
        frame_identity=identity,
    )
    session.attach_capture(CaptureSnapshot(
        "stopped", "stream", 6, 1, 4, 1, 0,
        950_000_000, 1_000_000_000, 100.0, True, None,
    ))
    payload = session.payload()
    assert payload["schema_version"] == 2
    assert payload["latency"]["provenance"] == "host_process_perf_counter"
    assert payload["latency"]["camera_sensor_to_display_measured"] is False
    assert payload["latency"]["stages"]["pose"] == {"mean_ms": 2.0, "maximum_ms": 2.0}
    assert payload["latency"]["stages"]["tracker_total"] == {
        "mean_ms": 10.5, "maximum_ms": 10.5,
    }
    assert payload["latency"]["stages"]["preview_composition"] == {
        "mean_ms": 5.0, "maximum_ms": 5.0,
    }
    assert payload["latency"]["capture_queue"] == {"mean_ms": 3.0, "maximum_ms": 3.0}
    assert payload["capture"]["captured"] == 6
    assert payload["frames"] == 1
    assert payload["first_processed_frame"]["sequence"] == 5
    assert payload["first_processed_frame"]["timestamp_provenance"] == "host_receive_monotonic"
    assert payload["detection_totals"] == {
        "pose_frames": 1, "face_frames": 1, "left_hand_frames": 0, "right_hand_frames": 1,
    }
    assert payload["privacy"] == {"raw_frames_recorded": False, "landmarks_recorded": False}


def test_unmeasured_queue_and_capture_are_unknown_not_zero() -> None:
    session = SessionRecord(
        CameraRequest(0, 1280, 720, 30), False,
        inference_provider="test", task_scheduling="parallel",
    )
    payload = session.payload()
    assert payload["capture"] is None
    assert payload["first_processed_frame"] is None
    assert payload["latency"]["capture_queue"] == {"mean_ms": None, "maximum_ms": None}


def test_unwritable_manifest_raises_instead_of_reported_success(tmp_path) -> None:
    session = SessionRecord(
        CameraRequest(0, 1280, 720, 30), False,
        inference_provider="test", task_scheduling="parallel",
    )
    not_directory = tmp_path / "file"
    not_directory.write_text("not a directory")
    with pytest.raises(SessionRecordError):
        session.write(not_directory)
