"""Dependency progress via events, real codecs/PTS, explicit synthetic neural sessions."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

import numpy as np
import pytest
from test_recording import tiny_vfr as tiny_vfr
from test_wholebody_compute_policy import arguments, fixture_runtime
from test_wholebody_handoff import Session, frames, install_controlled_ops

from motioncapture import wholebody_optimize_bench as bench
from motioncapture import wholebody_stages as stages
from motioncapture.recording import RecordedFrame, RecordedIdentity, inspect_recording
from motioncapture.wholebody_catalog import BenchmarkError
from motioncapture.wholebody_replay import SourcePacer
from motioncapture.wholebody_stage_comparison import update_digest


@pytest.mark.parametrize("deferred", [False, True])
def test_late_detection_starts_pose_during_consumer_validation(monkeypatch, deferred):
    """No sleeps: keep next detection blocked UNTIL consumer holds current output."""
    release_detector, release_pose, next_pose_started = install_controlled_ops(
        monkeypatch, wait_for_detection=False, block_detector=True)
    with stages.StagePipeline(Session, Session) as pipeline:
        stream = pipeline.packets(frames(2), 2, overlap=True, fast=False,
                                  advance_pose=True, defer_pose=deferred)
        try:
            with ThreadPoolExecutor(max_workers=1) as caller:
                # Timeout only guards a deadlock, never defines a speed threshold.
                first = caller.submit(next, stream).result(timeout=2)
            assert first.detected.frame.identity.sequence == 0
            assert not release_detector.is_set()
            assert pipeline.pose_requests == (2 if deferred else 1)
            release_detector.set()  # detection finishes WHILE caller validates frame 0
            if deferred:
                assert next_pose_started.wait(2)
                assert pipeline.emitted_frames == 1  # caller has not resumed the generator
            else:
                assert pipeline.pose_requests == 1  # structurally no request in the control
            release_pose.set()
            assert next(stream).detected.frame.identity.sequence == 1
            with pytest.raises(StopIteration):
                next(stream)
        finally:
            release_detector.set()
            release_pose.set()
            stream.close()
    snapshot = pipeline.snapshot()
    assert snapshot["dependency_pose_handoffs"] == int(deferred)
    assert snapshot["read_frames"] == snapshot["emitted_frames"] == 2
    assert snapshot["model_stages"] == 2
    assert set(snapshot["cleanup"].values()) == {"owner_released"}


def test_borrow_captures_exact_future_and_does_not_release_admission_slot():
    owner = stages.StageOwner(Session, "borrow-test")
    owner.open()
    try:
        with pytest.raises(BenchmarkError, match="has_no_request"):
            owner.borrow_result()
        first, second = object(), object()
        owner.submit(lambda *_: first)
        get_first = owner.borrow_result()
        assert get_first() is first
        with pytest.raises(BenchmarkError, match="outstanding"):
            owner.submit(lambda *_: second)
        assert owner.receive() is first
        owner.submit(lambda *_: second)
        assert get_first() is first  # getter cannot attach to the mutable owner's NEW request
        assert owner.receive() is second
    finally:
        owner.close()
    with pytest.raises(BenchmarkError, match="not_open"):
        owner.borrow_result()


@pytest.mark.parametrize("fault", ["detector", "pose"])
def test_dependency_errors_are_terminal_and_cleanup_admitted_work(monkeypatch, fault):
    release_detector, release_pose, _ = install_controlled_ops(
        monkeypatch, wait_for_detection=False, block_detector=True,
        fail_detector=fault == "detector")
    original_pose = stages.pose
    if fault == "pose":
        def fail(session, submitted, detected, *args):
            if detected.frame.identity.sequence == 1:
                raise BenchmarkError("injected_pose_failure")
            return original_pose(session, submitted, detected, *args)
        monkeypatch.setattr(stages, "pose", fail)
    pipeline = stages.StagePipeline(Session, Session)
    with pytest.raises(BenchmarkError, match=f"injected_{fault}_failure"):
        with pipeline:
            stream = pipeline.packets(frames(4), 4, overlap=True, fast=False,
                                      advance_pose=True, defer_pose=True)
            assert next(stream).detected.frame.identity.sequence == 0
            release_detector.set()
            release_pose.set()
            next(stream)
    state = pipeline.snapshot()
    assert state["emitted_frames"] == 1
    assert state["unemitted_pose_requests"] >= 1
    assert set(state["cleanup"].values()) == {"owner_released"}


@pytest.mark.parametrize("fault", [None, "consumer", "close"])
def test_early_exit_cancels_source_wait_and_dependent_pose_without_deadlock(fault):
    waiting = threading.Event()
    pacer = SourcePacer(2)
    def wait(seconds):
        waiting.set()
        return pacer._stop.wait(seconds)
    pacer._wait = wait
    made = []
    class Detector(Session):
        def close(self):
            super().close()
            if fault == "close":
                raise RuntimeError("injected_close_failure")
    class Pose(Session):
        def run(self, tensor):
            assert waiting.wait(2)
            return super().run(tensor)
    def create(kind):
        s = Detector("detector") if kind == "detector" else Pose()
        made.append(s)
        return s
    incoming = [RecordedFrame(RecordedIdentity("fixture", "s", i, i*3600, Fraction(1, 1)),
                              f.image_bgr, 0.0) for i, f in enumerate(frames(2))]
    pipeline = stages.StagePipeline(lambda: create("detector"), lambda: create("pose"),
                                    pacer=pacer)
    def consume():
        with pipeline:
            stream = pipeline.packets(incoming, 2, overlap=True, fast=True,
                                      advance_pose=True, defer_pose=True)
            assert next(stream).detected.frame.identity.sequence == 0
            assert waiting.is_set() and pipeline.pose_requests == 2
            stream.close()  # do not wait for the original source's next frame due in an hour
            if fault == "consumer":
                raise RuntimeError("injected_consumer_failure")
    if fault is None:
        consume()
    else:
        with pytest.raises(RuntimeError, match=f"injected_{fault}_failure"):
            consume()
    assert pacer.cancelled and pacer.released == 1
    assert len(made) == 2 and all(s.calls == 1 for s in made)
    assert all(s.closed_on == s.created_on for s in made)
    state = pipeline.snapshot()
    assert state["read_frames"] == 2 and state["emitted_frames"] == 1
    assert state["unemitted_pose_requests"] == 1


def test_wrong_dependency_identity_fails_before_model_call():
    session = Session()
    pair = list(frames(2))
    wrong = stages.Detected(pair[1], np.empty((0, 4), np.float32), {}, 1, 2)
    with pytest.raises(BenchmarkError, match="dependency_identity"):
        stages.pose_after_detection(session, time.perf_counter_ns(), lambda: wrong,
                                    pair[0].identity, (192, 256), .3, False, "numpy")
    assert session.calls == 0
    session.close()


class InputDependentSession(Session):
    """Neural fixture only: variable input-dependent outputs expose ordering/reuse bugs."""
    def run(self, tensor):
        self.called_on.append(threading.get_ident())
        self.calls += 1
        if self.kind == "detector":
            result = np.zeros((1, 3549, 85), np.float32)
            people = (1, 2, 0, 3, 8)[int(tensor[0, 0, 0, 0]) % 5]
            for j in range(people):
                result[0, j*5, :6] = [2, 2, 1, 1, .99, .99]
            return [result]
        h = hashlib.sha256(memoryview(tensor)).digest()
        x, y = np.zeros((1, 133, 384), np.float32), np.zeros((1, 133, 512), np.float32)
        x[:, :, h[0]] = .8
        y[:, :, h[1]] = .8
        return [x, y]


@pytest.mark.parametrize("count", [1, 2, 10])
def test_exact_prefix_all_people_and_input_sensitive_outputs_match(count):
    source = list(frames(count+1))
    signatures = []
    for deferred in (False, True):
        made = []
        def create(kind, registry=made):
            s = InputDependentSession(kind)
            registry.append(s)
            return s
        with stages.StagePipeline(lambda: create("detector"), lambda: create("pose"),
                                  normalization_kernel="opencv") as pipeline:
            packets = list(pipeline.packets(source, count, overlap=True, fast=True,
                                           advance_pose=True, defer_pose=deferred))
        digest = hashlib.sha256()
        for i, packet in enumerate(packets):
            assert packet.detected.frame is source[i]
            assert len(packet.people) == (1, 2, 0, 3, 8)[i % 5]
            update_digest(digest, packet.detected.frame, packet.people)
            if deferred and i:
                assert packet.times["pose_dependency_wait_ms"] >= 0
                assert packet.times["pose_owner_dispatch_ms"] >= 0
                assert packet.times["pose_queue_ms"] + 1e-6 >= (
                    packet.times["pose_dependency_wait_ms"]
                    + packet.times["pose_owner_dispatch_ms"])
        signatures.append(digest.hexdigest())
        assert pipeline.read_frames == pipeline.emitted_frames == count
        assert len(made) == 2 and made[0].calls == count
        assert made[1].calls == sum((1, 2, 0, 3, 8)[i % 5] for i in range(count))
        assert all(s.closed_on == s.created_on for s in made)
        assert pipeline.dependency_handoffs == (count-1 if deferred else 0)
    assert signatures[0] == signatures[1]


@pytest.mark.parametrize("advance,overlap,lanes", [(False, True, 1), (True, False, 1),
                                                   (True, True, 2)])
def test_unsupported_combinations_rejected_before_source_read(advance, overlap, lanes):
    with stages.StagePipeline(Session, Session, pose_lanes=lanes) as pipeline:
        with pytest.raises(BenchmarkError, match="invalid_pipeline"):
            next(pipeline.packets(frames(2), 2, overlap=overlap, fast=False,
                                  advance_pose=advance, defer_pose=True))
        assert pipeline.read_frames == pipeline.pose_requests == 0


def test_real_vfr_cli_abba_manifest_hashes_and_unchanged_session_count(
        tiny_vfr, tmp_path, monkeypatch):
    args = arguments(tiny_vfr, tmp_path, "--suite", "dependency-handoff")
    made = fixture_runtime(monkeypatch, tmp_path)
    assert bench.execute(args) == 0
    d = json.loads(args.output.read_text())
    n = len(inspect_recording(tiny_vfr).pts)
    assert [r["mode"] for r in d["runs"]] == [
        "source-pts-ready-cvlut", "source-pts-dependent-cvlut",
        "source-pts-dependent-cvlut", "source-pts-ready-cvlut",
    ]
    assert len(made) == 8  # four passes, ONE detector and ONE pose session per pass
    assert d["all_pass_prediction_hashes_equal"]
    assert d["all_pass_detector_hashes_equal"] and d["all_pass_pixel_hashes_equal"]
    assert d["accuracy_verified"] is False and d["live_60fps_verified"] is False
    for i, row in enumerate(d["runs"]):
        assert row["all_frames"]["frames"] == row["replay_ages"]["frames"] == n
        assert row["source_pacing"] == "original_pts" and row["pose_lanes"] == 1
        assert row["pipeline"]["source_pacing"]["clock_rebases"] == 0
        assert row["pipeline"]["unemitted_read_frames"] == 0
        assert row["pipeline"]["source_pacing"]["frames_skipped"] == 0
        assert row["pipeline"]["dependency_pose_handoffs"] == (n-1 if i in (1, 2) else 0)
        assert set(row["pipeline"]["cleanup"].values()) == {"owner_released"}
        assert row["unpaced_loop_fps"] is None
        if i:
            assert row["provider_disagreement"]["per_frame_predictions"]["checked_frames"] == n
            assert row["provider_disagreement"]["per_frame_detector_boxes"]["checked_frames"] == n
    with pytest.raises(FileExistsError):
        bench.execute(args)


def test_failed_continuation_retains_terminal_prefix_and_no_success_fps(
        tiny_vfr, tmp_path, monkeypatch):
    args = arguments(tiny_vfr, tmp_path, "--suite", "dependency-handoff")
    made = fixture_runtime(monkeypatch, tmp_path)
    original = stages.pose_after_detection
    def fail(session, submitted, result, expected_identity, *params):
        if expected_identity.sequence == 2:
            raise BenchmarkError("injected_continuation_failure")
        return original(session, submitted, result, expected_identity, *params)
    monkeypatch.setattr(stages, "pose_after_detection", fail)
    assert bench.execute(args) == 2
    d = json.loads(args.output.read_text())
    assert len(d["runs"]) == 2 and d["runs"][0]["status"] == "completed"
    row = d["runs"][1]
    assert row["status"] == "failed"
    assert row["error"]["code"] == "injected_continuation_failure"
    assert row["all_frames"]["frames"] < row["requested_frames"]
    assert row["hash_scope"] == "completed_prefix" and row["paced_loop_fps"] is None
    assert row["replay_ages"]["coverage"] == "partial"
    assert not args.output.with_name("result.arm-03.started.json").exists()
    assert all(s.closed_on == s.created_on for s in made)
