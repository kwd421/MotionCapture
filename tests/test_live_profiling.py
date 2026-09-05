import math

import pytest

from motioncapture.capture import CameraRequest
from motioncapture.contracts import FrameIdentity, LandmarkResult, LandmarkTimings
from motioncapture.profiling import PerformanceSession, ProfiledLatency


def test_empty_samples_are_unknown_not_zero_success():
    summary = ProfiledLatency(1000 / 30)
    result = summary.distribution()
    assert result["mean_ms"] is None and result["maximum_ms"] is None
    assert result["p95_upper_ms"] is None and result["samples"] == 0


def test_histogram_bounds_means_and_budget():
    summary = ProfiledLatency(3)
    for value in (0, 1.1, 2.2, 3.3, 4.4):
        summary.observe(value)
    stats = summary.distribution()
    assert stats["mean_ms"] == pytest.approx(2.2)
    assert stats["maximum_ms"] == 4.4
    assert stats["p50_upper_ms"] == 2.25
    assert stats["p95_upper_ms"] == 4.5
    assert stats["over_budget_samples"] == 2
    assert summary.payload(5) == {"mean_ms": 2.2, "maximum_ms": 4.4}


def test_overflow_is_not_clamped_into_fake_percentile():
    summary = ProfiledLatency(30)
    summary.observe(1234)
    assert summary.distribution()["p95_upper_ms"] is None
    assert summary.distribution()["overflow_samples"] == 1
    assert summary.distribution()["maximum_ms"] == 1234


@pytest.mark.parametrize("value", [-0.1, math.inf, math.nan])
def test_bad_sample_does_not_partially_update(value):
    summary = ProfiledLatency(30)
    with pytest.raises(ValueError):
        summary.observe(value)
    assert summary.count == 0 and summary.total_ms == 0


def test_histogram_memory_does_not_grow_with_session_duration():
    summary = ProfiledLatency(30)
    size = len(summary.bins)
    for _ in range(50000):
        summary.observe(17)
    assert len(summary.bins) == size and summary.count == 50000
    assert summary.percentile_upper(0.99) == 17


def test_session_retains_v2_values_and_adds_distinct_gui_counters():
    session = PerformanceSession(CameraRequest(0, 1280, 720, 30), True,
                                 inference_provider="fixture", task_scheduling="parallel")
    session.headless = True
    result = LandmarkResult((), (), (), (), (), (), (), ())
    timings = LandmarkTimings(1, 2, 3, 4, 4, 0.5, 5.5)
    identity = FrameIdentity("fixture", "s", 0, 1_000_000)
    session.observe(result, timings, preview_composition_ms=2, host_post_receive_total_ms=9,
                    frame_timestamp_ns=identity.received_ns, frame_identity=identity,
                    capture_queue_ms=1.5)
    session.completed_at(2_000_000)
    payload = session.payload()
    assert payload["schema_version"] == 2
    assert payload["latency"]["stages"]["tracker_total"] == {"mean_ms": 5.5, "maximum_ms": 5.5}
    perf = payload["performance"]
    assert perf["stage_distributions"]["tracker_total"]["samples"] == 1
    assert perf["capture_queue"]["samples"] == 1
    assert perf["additional_stages"]["event_pump"]["samples"] == 0
    assert perf["additional_stages"]["event_pump"]["mean_ms"] is None
    assert perf["sensor_to_photon_measured"] is False
    assert payload["privacy"] == {"raw_frames_recorded": False, "landmarks_recorded": False}


def test_live_session_does_not_claim_gui_measurement_before_any_gui_calls():
    session = PerformanceSession(CameraRequest(0, 1280, 720, 30), True,
                                 inference_provider="unknown", task_scheduling="parallel")
    session.headless = False
    payload = session.payload()["performance"]
    assert payload["presentation_requested"] is True
    assert payload["presentation_measured"] is False
    session.extra["presentation_submit"].observe(1)
    session.extra["event_pump"].observe(2)
    assert session.payload()["performance"]["presentation_measured"] is True
