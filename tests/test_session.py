from types import SimpleNamespace

from motioncapture.capture import CameraRequest
from motioncapture.landmarkers import LandmarkTimings
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

    session.observe(
        result,
        timings,
        preview_composition_ms=5.0,
        host_post_receive_total_ms=15.5,
        frame_timestamp_ns=1_000_000_000,
    )
    payload = session.payload()

    assert payload["latency"]["provenance"] == "host_process_perf_counter"
    assert payload["latency"]["camera_sensor_to_display_measured"] is False
    assert payload["latency"]["stages"]["pose"] == {
        "mean_ms": 2.0,
        "maximum_ms": 2.0,
    }
    assert payload["latency"]["stages"]["tracker_total"] == {
        "mean_ms": 10.5,
        "maximum_ms": 10.5,
    }
    assert payload["latency"]["stages"]["preview_composition"] == {
        "mean_ms": 5.0,
        "maximum_ms": 5.0,
    }
    assert payload["detection_totals"] == {
        "pose_frames": 1,
        "face_frames": 1,
        "left_hand_frames": 0,
        "right_hand_frames": 1,
    }
