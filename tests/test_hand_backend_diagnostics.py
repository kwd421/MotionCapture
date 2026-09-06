"""Failure ledger, partial-pass and outlier regressions; no real AI performance claims."""
import concurrent.futures
import json
import urllib.error
from types import SimpleNamespace

import pytest

from motioncapture import recording_bench as bench
from motioncapture.browser_hands import BrowserHandError, BrowserHandLab
from motioncapture.hand_comparison import HandReference
from motioncapture.recording import inspect_recording
from test_browser_hands import next_message, request
from test_hand_comparison import frame, result
from test_recording import tiny_vfr  # noqa: F401


def test_failure_keeps_delivered_request_and_sampled_stage_after_pending_is_cleared():
    with BrowserHandLab({}) as lab:
        request(lab, "/rpc/hello", {})
        with concurrent.futures.ThreadPoolExecutor(1) as executor:
            future = executor.submit(lab.call, "detect", {"timestamp_ms": 55422}, b"secret pixels")
            message, _ = next_message(lab)
            request(lab, "/rpc/progress", {
                "sequence": 4, "worker": {"id": message["id"], "phase": "detect",
                                           "untrusted": lab.token},
                "phase_age_ms": 1234, "visible": True,
            })
            # Simulate the already-expired deadline deterministically, after delivery/progress.
            lab.fail("browser_request_timeout")
            with pytest.raises(BrowserHandError, match="browser_request_timeout"):
                future.result(timeout=2)
            snapshot = lab.failure_snapshot
            assert lab.pending is None
            assert snapshot["pending"]["id"] == message["id"]
            assert snapshot["pending"]["timestamp_ms"] == 55422
            assert snapshot["pending"]["claimed_by_browser_fetch"] is True
            assert snapshot["browser_progress"]["phase"] == "detect"
            assert snapshot["heartbeat_count"] == 1
            assert lab.token not in json.dumps(snapshot)
            assert "secret pixels" not in json.dumps(snapshot)
            lab.fail("browser_disconnected")
            assert lab.failure_snapshot == snapshot  # first failure is immutable


def test_worker_fault_wakes_waiter_without_waiting_for_request_deadline():
    with BrowserHandLab({}, timeout=120) as lab:
        request(lab, "/rpc/hello", {})
        with concurrent.futures.ThreadPoolExecutor(1) as executor:
            future = executor.submit(lab.call, "detect")
            next_message(lab)
            request(lab, "/rpc/fault", {"code": "browser_gpu_context_lost"})
            with pytest.raises(BrowserHandError, match="browser_gpu_context_lost"):
                future.result(timeout=2)
            assert lab.failure_snapshot["pending"]["operation"] == "detect"
            with pytest.raises(BrowserHandError):
                lab.call("open", {"delegate": "CPU"})


def test_stale_heartbeat_cannot_replace_newer_stage_or_visibility():
    with BrowserHandLab({}) as lab:
        request(lab, "/rpc/hello", {})
        request(lab, "/rpc/progress", {"sequence": 2,
            "worker": {"id": 3, "phase": "detect"}, "phase_age_ms": 20, "visible": True})
        request(lab, "/rpc/progress", {"sequence": 1,
            "worker": {"id": 2, "phase": "waiting"}, "phase_age_ms": 0, "visible": False})
        snapshot = lab.snapshot()
        assert snapshot["browser_progress"] == {"id": 3, "phase": "detect", "phase_age_ms": 20}
        assert snapshot["heartbeat_count"] == 1 and snapshot["tab_visible"] is True


def test_arbitrary_fault_text_is_not_persisted():
    with BrowserHandLab({}) as lab:
        request(lab, "/rpc/hello", {})
        with pytest.raises(urllib.error.HTTPError):
            request(lab, "/rpc/fault", {"code": "/private/user/camera.mov"})
        assert lab.failure == "invalid_browser_protocol"
        assert "camera.mov" not in json.dumps(lab.failure_snapshot)


def test_partial_comparison_and_bounded_outlier_locations_without_coordinates():
    ref = HandReference(list(range(30)), 1920, 1080)
    for i in range(30):
        ref.record(frame(i, i), result())
    compare = ref.comparator()
    for i in range(25):
        compare.observe(frame(i, i), result(dx=i / 1920))
    with pytest.raises(ValueError, match="Incomplete"):
        compare.summary()
    summary = compare.summary(allow_partial=True)
    assert summary["comparison_complete"] is False
    assert summary["frames"] == 25
    worst = summary["largest_disagreement_frames"]
    assert len(worst) == 16 and worst[0]["sequence"] == 24
    assert worst[0]["max_xy_displacement_pixels"] == pytest.approx(24)
    assert set(worst[0]) == {"sequence", "pts", "max_xy_displacement_pixels"}


def test_failed_frame_is_not_averaged_and_primary_error_survives_cleanup(tiny_vfr, tmp_path):
    from motioncapture.contracts import LandmarkResult, LandmarkTimings

    class Tracker:
        def __enter__(self):
            return self
        def __exit__(self, *_):
            raise RuntimeError("secondary cleanup failure")
        def process_recorded(self, frame):
            if frame.identity.sequence == 2:
                raise ValueError("primary inference failure")
            return SimpleNamespace(identity=frame.identity,
                timings=LandmarkTimings(0, 1, 2, 3, 3, 0, 3),
                result=LandmarkResult((), (), (), (), (), (), (), ()))

    args = bench._parser().parse_args([str(tiny_vfr), "--output", str(tmp_path / "unused"),
                                       "--preview", "none"])
    snapshot = {}
    with pytest.raises(ValueError, match="primary inference"):
        bench.run_pass(args, inspect_recording(tiny_vfr),
                       tracker_factory=lambda *_: Tracker(), failure_record=snapshot)
    assert snapshot["all_frames"]["frames"] == 2
    assert snapshot["all_frames"]["stages"]["hands_ms"]["samples"] == 2
    assert snapshot["all_frames"]["stages"]["hands_ms"]["mean_ms"] == 2
    assert snapshot["failure_phase"] == "inference"
    assert snapshot["last_completed_frame"]["sequence"] == 1
    assert snapshot["current_frame"]["sequence"] == 2
    assert snapshot["decoder_cleanup_complete"] is True
    assert snapshot["tracker_cleanup_complete"] is False
    assert snapshot["unpaced_loop_fps"] is None
    assert not snapshot["timeout_sample_in_completed_statistics"]
    assert str(tmp_path) not in json.dumps(snapshot)
