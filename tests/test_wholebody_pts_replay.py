"""Real decoder/owners; deterministic pacing clocks; explicit synthetic models."""
from __future__ import annotations

import json
import math
import threading
from dataclasses import replace
from fractions import Fraction

import pytest
from test_recording import tiny_vfr as tiny_vfr
from test_wholebody_handoff import Session, frames

from motioncapture import wholebody_optimize_bench as bench
from motioncapture.recording import RecordedFrame, RecordedIdentity, inspect_recording
from motioncapture.wholebody_catalog import BenchmarkError
from motioncapture.wholebody_replay import Release, ReplayAges, ReplayCancelled, SourcePacer
from motioncapture.wholebody_stages import StagePipeline


class Clock:
    def __init__(self):
        self.ns = 5_000_000_000
        self.waits = []

    def now(self):
        return self.ns

    def wait(self, seconds):
        self.waits.append(seconds)
        self.ns += math.ceil(seconds*1e9)
        return False


def identity(i, pts, tb=Fraction(1, 90000)):
    return RecordedIdentity("fixture", "stream", i, pts, tb)


def test_absolute_rational_pts_schedule_never_rebases_late_frames():
    clock = Clock()
    pacer = SourcePacer(4, clock=clock.now, waiter=clock.wait)
    epoch = clock.ns
    first = pacer.wait(identity(0, 777))
    assert first.due_ns == first.released_ns == epoch
    second = pacer.wait(identity(1, 2294))
    assert second.due_ns == epoch + math.ceil(Fraction(1517, 90000)*1_000_000_000)
    assert second.released_ns >= second.due_ns
    clock.ns += 100_000_000
    late = pacer.wait(identity(2, 3044))
    assert late.wait_ms == 0 and late.released_ns > late.due_ns
    last = pacer.wait(identity(3, 5311))
    assert last.due_ns == epoch + math.ceil(Fraction(4534, 90000)*1_000_000_000)
    assert len(clock.waits) == 1  # No sleeps or reset-to-now when behind.
    summary = pacer.summary()
    assert summary["released_frames"] == 4 and summary["clock_rebases"] == 0
    assert summary["pending_sequence"] is None and summary["frames_skipped"] == 0


def test_early_wait_return_is_rechecked_and_not_published_early():
    clock = Clock()
    early = [True]
    def wait(seconds):
        if early and early.pop():
            clock.ns += 1
        else:
            clock.wait(seconds)
        return False
    pacer = SourcePacer(2, clock=clock.now, waiter=wait)
    pacer.wait(identity(0, 0))
    release = pacer.wait(identity(1, 1500))
    assert release.released_ns >= release.due_ns


@pytest.mark.parametrize("bad", [identity(2, 1), identity(1, 0),
                                   identity(1, 1, Fraction(1, 1)),
                                   replace(identity(1, 1), stream_id="changed")])
def test_bad_source_identity_is_not_synthesized(bad):
    clock = Clock()
    pacer = SourcePacer(2, clock=clock.now, waiter=clock.wait)
    pacer.wait(identity(0, 0))
    with pytest.raises(BenchmarkError, match="source_clock"):
        pacer.wait(bad)
    assert pacer.released == 1


def test_host_clock_regression_and_cancel_do_not_release_a_frame():
    clock = Clock()
    pacer = SourcePacer(2, clock=clock.now, waiter=clock.wait)
    pacer.wait(identity(0, 0))
    clock.ns -= 1
    with pytest.raises(BenchmarkError, match="host_clock"):
        pacer.wait(identity(1, 1500))
    assert pacer.released == 1
    pacer.cancel()
    with pytest.raises(ReplayCancelled):
        pacer.wait(identity(1, 1500))
    assert pacer.summary()["cancel_requested"]


def test_current_result_does_not_wait_for_next_source_due_time():
    waiting, resume = threading.Event(), threading.Event()
    clock = Clock()
    def wait(seconds):
        waiting.set()
        assert resume.wait(2), "consumer failed to receive current result before future release"
        return clock.wait(seconds)
    pacer = SourcePacer(2, clock=clock.now, waiter=wait)
    class Pose(Session):
        def run(self, tensor):
            if self.calls == 0:
                assert waiting.wait(2)
            return super().run(tensor)
    observed = []
    def source():
        for i, f in enumerate(frames(3)):
            observed.append(i)
            yield RecordedFrame(identity(i, i*90000), f.image_bgr, f.decode_ms)
    with StagePipeline(lambda: Session("detector"), Pose, normalization_kernel="opencv",
                       pacer=pacer) as pipeline:
        stream = pipeline.packets(source(), 2, overlap=True, fast=True, advance_pose=True)
        first = next(stream)
        assert first.detected.frame.identity.sequence == 0 and waiting.is_set()
        assert not resume.is_set()
        resume.set()
        second = next(stream)
        assert second.detected.source_release.due_ns - first.detected.source_release.due_ns == 10**9
        with pytest.raises(StopIteration):
            next(stream)
    assert observed == [0, 1]  # Prefix did not read or infer speculative N+1.
    assert pipeline.snapshot()["source_pacing"]["released_frames"] == 2


@pytest.mark.parametrize("fail_pose,fail_close", [(False, False), (True, False),
                                               (False, True), (True, True)])
def test_early_stop_or_native_failure_wakes_wait_and_keeps_primary_error(fail_pose, fail_close):
    entered = threading.Event()
    pacer = SourcePacer(2)
    def wait(seconds):
        entered.set()
        return pacer._stop.wait(seconds)  # Exercise the actual cancellable wait.
    pacer._wait = wait
    made = []
    class Pose(Session):
        def run(self, tensor):
            assert entered.wait(2)
            if fail_pose:
                raise RuntimeError("native_pose_failure")
            return super().run(tensor)
    class Detector(Session):
        def close(self):
            super().close()
            if fail_close:
                raise RuntimeError("native_close_failure")
    def factory(kind):
        model = Pose() if kind == "pose" else Detector("detector")
        made.append(model)
        return model
    incoming = [RecordedFrame(identity(i, i*90000*3600), f.image_bgr, f.decode_ms)
                for i, f in enumerate(frames(2))]
    def consume():
        with StagePipeline(lambda: factory("detector"), lambda: factory("pose"),
                           pacer=pacer) as pipeline:
            stream = pipeline.packets(incoming, 2, overlap=True, fast=True, advance_pose=True)
            next(stream)
            stream.close()  # Stop before the frame due in one hour; no timed sleep test.
        return pipeline
    if fail_pose:
        with pytest.raises(RuntimeError, match="native_pose_failure"):
            consume()
    elif fail_close:
        with pytest.raises(RuntimeError, match="native_close_failure"):
            consume()
    else:
        pipeline = consume()
        assert pipeline.snapshot()["unemitted_read_frames"] == 1
        assert pipeline.snapshot()["cleanup"] == {
            "detector": "owner_released", "pose": "owner_released"}
    assert pacer.cancelled and pacer.released == 1
    assert all(model.closed_on == model.created_on for model in made)


def test_source_age_is_from_due_time_not_late_actual_admission():
    ages = ReplayAges(2)
    assert ages.summary()["source_age_ms"]["mean_ms"] is None
    ages.add(identity(0, 0), Release(0, 10_000_000, 0), 30_000_000)
    ages.add(identity(1, 900000), Release(10**10, 10**10+50_000_000, 0), 10**10+70_000_000)
    result = ages.summary()
    assert result["source_age_ms"]["mean_ms"] == 50
    assert result["detector_release_lateness_ms"]["mean_ms"] == 30
    assert result["last_source_age_ms"] == 70
    assert result["source_age_ms"]["threshold_exceedances"]["50.0"] == 1
    assert "over_budget" not in result["source_age_ms"]
    assert len(result["source_windows"]) == 2 and result["coverage"] == "complete"
    before = ages.summary()
    with pytest.raises(BenchmarkError):
        ages.add(identity(2, 900001), Release(1, 2, 0), 3)
    assert ages.summary() == before
    assert result["live_60fps_verified"] is False


def test_real_vfr_cli_paced_abba_keeps_every_pixel_pts_and_result(tiny_vfr, tmp_path, monkeypatch):
    args = bench.parser().parse_args([str(tiny_vfr), "--research-only", "--suite", "paced",
                                     "--max-frames", "0", "--output", str(tmp_path/"paced.json")])
    monkeypatch.setattr(bench, "verify_asset", lambda root, key: (tmp_path/key, {"fixture": True}))
    def run(args, probe, mode, factories, reference):
        return bench.run_pass(args, probe, mode,
                              (lambda: Session("detector"), lambda: Session("pose")), reference)
    assert bench.execute(args, pass_runner=run) == 0
    data = json.loads(args.output.read_text())
    assert data["configuration"]["plan"] == ["overlap-ready-cvlut", "source-pts-ready-cvlut",
                                               "source-pts-ready-cvlut", "overlap-ready-cvlut"]
    assert data["all_pass_prediction_hashes_equal"]
    assert data["all_pass_detector_hashes_equal"] and data["all_pass_pixel_hashes_equal"]
    n = len(inspect_recording(tiny_vfr).pts)
    for index, row in enumerate(data["runs"]):
        assert row["all_frames"]["frames"] == row["pipeline"]["emitted_frames"] == n
        assert row["live_60fps_verified"] is False
        if index in (1, 2):
            assert row["unpaced_loop_fps"] is None and row["paced_loop_fps"] > 0
            assert row["replay_ages"]["frames"] == n
            assert "NOT replay deadline misses" in row["output_cadence"]["interval_budget_note"]
            assert row["pipeline"]["source_pacing"]["scheduled_source_rate_hz"] > 0
            assert row["all_frames"]["stages"]["source_pacing_wait_ms"]["samples"] == n
            assert row["all_frames"]["stages"]["replay_observer_ms"]["samples"] == n
            assert row["pipeline"]["source_pacing"]["released_frames"] == n
            assert row["loop_s"] >= row["pipeline"]["source_pacing"]["scheduled_source_span_s"]
            assert not row["pipeline"]["source_pacing"]["cancel_requested"]
        else:
            assert row["unpaced_loop_fps"] > 0 and row["paced_loop_fps"] is None
            assert row["replay_ages"] is None
    assert str(tiny_vfr) not in args.output.read_text()
    assert len(list(tmp_path.glob('paced.arm-*.json'))) == 8
    with pytest.raises(FileExistsError):
        bench.execute(args, pass_runner=run)


def test_failed_paced_pass_retains_prefix_and_no_success_fps(tiny_vfr, tmp_path):
    class Failure(Session):
        def run(self, tensor):
            if self.calls == 1:
                raise RuntimeError("injected_second_pose_failure")
            return super().run(tensor)
    args = bench.parser().parse_args([str(tiny_vfr), "--research-only", "--max-frames", "0",
                                     "--output", str(tmp_path/"fail.json")])
    row, reference = bench.run_pass(args, inspect_recording(tiny_vfr), "source-pts-ready-cvlut",
                                    (lambda: Session("detector"), Failure))
    assert row["status"] == "failed" and reference is None
    assert row["unpaced_loop_fps"] is row["paced_loop_fps"] is None
    assert row["all_frames"]["frames"] == row["replay_ages"]["frames"] == 1
    assert row["replay_ages"]["coverage"] == "partial"
    assert row["error"]["last_completed"]["sequence"] == 0
    assert row["pipeline"]["source_pacing"]["cancel_requested"] is True
