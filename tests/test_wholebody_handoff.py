"""Event-controlled real pipeline tests; synthetic sessions do not benchmark AI."""
from __future__ import annotations

import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from motioncapture import wholebody_optimize_bench as bench
from motioncapture import wholebody_stages as stages
from motioncapture.recording import inspect_recording
from motioncapture.wholebody_catalog import BenchmarkError


class Session:
    metadata = {"kind": "synthetic_test_double_not_native_inference"}

    def __init__(self, kind="pose"):
        self.kind = kind
        self.created_on = threading.get_ident()
        self.called_on = []
        self.closed_on = None
        self.calls = 0

    def run(self, tensor):
        self.called_on.append(threading.get_ident())
        self.calls += 1
        if self.kind == "detector":
            result = np.zeros((1, 3549, 85), np.float32)
            result[0, 0, :6] = [26, 20, 3, 3, .99, .99]
            return [result]
        x, y = np.zeros((1, 133, 384), np.float32), np.zeros((1, 133, 512), np.float32)
        x[:, :, 192] = .8
        y[:, :, 256] = .8
        return [x, y]

    def close(self):
        self.closed_on = threading.get_ident()
        assert self.closed_on == self.created_on
        assert all(i == self.created_on for i in self.called_on)


def frames(count):
    for i in range(count):
        # Deliberately variable PTS intervals, never index / 60 timestamps.
        yield SimpleNamespace(
            identity=SimpleNamespace(sequence=i, pts=i*1500+(i//2)*80,
                                     time_base=Fraction(1, 90000), source_id="fixture"),
            image_bgr=np.full((48, 64, 3), i, np.uint8), decode_ms=0.0)


def install_controlled_ops(monkeypatch, *, wait_for_detection=True, block_detector=False,
                           fail_detector=False, wrong_identity=False):
    done = threading.Event()
    release_detector, release_pose, pose_one_started = (threading.Event() for _ in range(3))
    native_callback = stages.StageOwner._record_failure

    def callback(owner, future):
        try:
            native_callback(owner, future)
        finally:
            if not future.cancelled():
                error = future.exception()
                if error is not None:
                    if getattr(error, "code", None) == "injected_detector_failure":
                        done.set()
                else:
                    value = future.result()
                    if isinstance(value, stages.Detected) and value.frame.identity.sequence != 0:
                        done.set()  # future is DONE, even when callback runs on the caller.

    def detect(session, submitted, frame):
        if frame.identity.sequence == 1:
            if block_detector:
                assert release_detector.wait(3), "test release not signalled"
            if fail_detector:
                raise BenchmarkError("injected_detector_failure")
            if wrong_identity:
                frame = SimpleNamespace(**vars(frame))
                frame.identity = SimpleNamespace(**vars(frame.identity))
                frame.identity.sequence = 77
        return stages.Detected(frame, np.empty((0, 4), np.float32), {},
                               submitted, time.perf_counter_ns())

    def pose(session, submitted, detected, *args):
        if detected.frame.identity.sequence == 0 and wait_for_detection:
            assert done.wait(3), "next detector never completed"
        if detected.frame.identity.sequence == 1:
            pose_one_started.set()
            assert release_pose.wait(3), "test release not signalled"
        return stages.Posed(detected, [], {"pose_inference_ms": 0.0}, (),
                            time.perf_counter_ns())

    monkeypatch.setattr(stages.StageOwner, "_record_failure", callback)
    monkeypatch.setattr(stages, "detect", detect)
    monkeypatch.setattr(stages, "pose", pose)
    return release_detector, release_pose, pose_one_started


@pytest.mark.parametrize("advance", [False, True])
def test_next_pose_can_start_while_caller_validates_current_packet(monkeypatch, advance):
    rd, rp, began = install_controlled_ops(monkeypatch)
    with stages.StagePipeline(Session, Session) as pipeline:
        stream = pipeline.packets(frames(2), 2, overlap=True, fast=False, advance_pose=advance)
        try:
            first = next(stream)
            before = first.detected.frame.image_bgr.copy()
            assert pipeline.pose_requests == (2 if advance else 1)
            if advance:
                assert began.wait(1), "next pose not admitted while observer holds frame 0"
            else:
                assert not began.is_set()  # No second request exists in the control.
            np.testing.assert_array_equal(first.detected.frame.image_bgr, before)
        finally:
            rd.set()
            rp.set()
        second = next(stream)
        assert second.detected.frame.identity.sequence == 1
        assert not np.shares_memory(first.detected.frame.image_bgr, second.detected.frame.image_bgr)
        assert "pose_submit_gap_ms" not in first.times
        assert second.times["pose_submit_gap_ms"] >= 0
        with pytest.raises(StopIteration):
            next(stream)
    assert pipeline.snapshot()["ready_pose_handoffs"] == int(advance)
    assert pipeline.snapshot()["unemitted_pose_requests"] == 0


def test_unfinished_detector_does_not_hold_current_result(monkeypatch):
    rd, rp, _ = install_controlled_ops(monkeypatch, wait_for_detection=False, block_detector=True)
    with stages.StagePipeline(Session, Session) as pipeline:
        stream = pipeline.packets(frames(2), 2, overlap=True, fast=False, advance_pose=True)
        with ThreadPoolExecutor(max_workers=1) as caller:
            current = caller.submit(next, stream)
            try:
                # Timeout guards a deadlock regression, not a performance expectation.
                assert current.result(timeout=1).detected.frame.identity.sequence == 0
                assert pipeline.pose_requests == 1
                assert pipeline.snapshot()["next_detection_not_ready"] == 1
            finally:
                rd.set()
                rp.set()
        assert [p.detected.frame.identity.sequence for p in stream] == [1]


@pytest.mark.parametrize("fault", ["error", "identity"])
def test_ready_next_failure_is_not_published_as_current_success(monkeypatch, fault):
    rd, rp, _ = install_controlled_ops(monkeypatch, fail_detector=fault == "error",
                                      wrong_identity=fault == "identity")
    with pytest.raises(BenchmarkError):
        with stages.StagePipeline(Session, Session) as pipeline:
            try:
                next(pipeline.packets(frames(2), 2, overlap=True, fast=False, advance_pose=True))
            finally:
                rd.set()
                rp.set()
    assert pipeline.emitted_frames == 0
    assert pipeline.pose_requests == 1


def test_early_consumer_close_finishes_only_admitted_work(monkeypatch):
    rd, rp, began = install_controlled_ops(monkeypatch)
    with stages.StagePipeline(Session, Session) as pipeline:
        stream = pipeline.packets(frames(5), 5, overlap=True, fast=False, advance_pose=True)
        try:
            first = next(stream)
            assert began.wait(1)
            stream.close()
            assert first.detected.frame.identity.sequence == 0
            assert pipeline.read_frames == 2 and pipeline.pose_requests == 2
        finally:
            rd.set()
            rp.set()
    snapshot = pipeline.snapshot()
    assert snapshot["emitted_frames"] == 1
    assert snapshot["unemitted_pose_requests"] == snapshot["unemitted_read_frames"] == 1
    assert snapshot["cleanup"] == {"detector": "owner_released", "pose": "owner_released"}


@pytest.mark.parametrize("advance", [False, True])
@pytest.mark.parametrize("count", [1, 2, 7])
def test_real_codecs_exact_n_order_and_owner_lifetime(advance, count):
    owners = []
    def make(kind):
        def construct():
            owner = Session(kind)
            owners.append(owner)
            return owner
        return construct
    with stages.StagePipeline(make("detector"), make("pose")) as pipeline:
        emitted = list(pipeline.packets(frames(count+1), count, overlap=True, fast=True,
                                        advance_pose=advance))
    actual_pts = [p.detected.frame.identity.pts for p in emitted]
    assert actual_pts == [f.identity.pts for f in frames(count)]
    assert pipeline.read_frames == pipeline.pose_requests == pipeline.emitted_frames == count
    assert [o.calls for o in owners] == [count, count]
    assert len({o.created_on for o in owners}) == 2
    assert all(o.closed_on == o.created_on != threading.get_ident() for o in owners)
    assert len([p for p in emitted if "pose_submit_gap_ms" in p.times]) == count-1


def test_completed_but_unreceived_result_still_occupies_owner_slot():
    with stages.StagePipeline(Session, Session) as pipeline:
        completed = threading.Event()
        pipeline.pose.submit(lambda *_: None)
        pipeline.pose._future.add_done_callback(lambda _: completed.set())
        assert completed.wait(1)
        with pytest.raises(BenchmarkError, match="outstanding"):
            pipeline.pose.submit(lambda *_: None)
        assert pipeline.pose.take_ready() == (True, None)
        with pytest.raises(BenchmarkError, match="has_no_request"):
            pipeline.pose.take_ready()


def test_incompatible_sequential_ready_mode_rejected_before_read():
    with stages.StagePipeline(Session, Session) as pipeline:
        with pytest.raises(BenchmarkError, match="invalid_pipeline_run"):
            next(pipeline.packets(frames(2), 2, overlap=False, fast=False, advance_pose=True))
        assert pipeline.read_frames == pipeline.pose_requests == 0


@pytest.fixture
def video(tmp_path):
    target = tmp_path / "fixture.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=64x48:rate=30:duration=0.2", "-vf",
                    "setpts=(N+floor(N/2))/(30*TB)", "-fps_mode", "passthrough",
                    "-c:v", "libx264", "-threads", "1", str(target)],
                   check=True, timeout=10)
    return target


def args_for(video, target):
    return bench.parser().parse_args([str(video), "--research-only", "--max-frames", "0",
                                     "--suite", "handoff", "--output", str(target)])


def test_actual_runner_abba_all_hashes_equal_with_real_vfr_decode(video, tmp_path, monkeypatch):
    args = args_for(video, tmp_path/"run.json")
    monkeypatch.setattr(bench, "verify_asset", lambda root, key: (Path(key), {"test_double": True}))
    monkeypatch.setattr(bench, "OrtModel", lambda path, *a, **k:
                        Session("detector" if str(path) == "yolox-tiny" else "pose"))
    assert bench.execute(args) == 0
    report = json.loads(args.output.read_text())
    assert report["configuration"]["plan"] == ["overlap-lut", "overlap-ready-lut",
                                               "overlap-ready-lut", "overlap-lut"]
    assert report["all_pass_prediction_hashes_equal"] is True
    assert report["all_pass_pixel_hashes_equal"] is True
    assert report["all_pass_detector_hashes_equal"] is True
    count = len(inspect_recording(video).pts)
    for row in report["runs"]:
        assert row["status"] == "completed"
        assert row["all_frames"]["frames"] == row["pipeline"]["pose_requests"] == count
        assert row["all_frames"]["stages"]["pose_submit_gap_ms"]["samples"] == count-1
        assert row["all_frames"]["stages"]["verified_output_interval_ms"]["samples"] == count-1
        assert row["live_60fps_verified"] is False
    assert len(list(tmp_path.glob("run.arm-*.started.json"))) == 4
    assert str(video) not in args.output.read_text()
    with pytest.raises(FileExistsError):
        bench.execute(args)


def test_original_six_arm_plan_preserved(video, tmp_path, monkeypatch):
    args = args_for(video, tmp_path/"old-plan.json")
    args.suite = "optimization"
    monkeypatch.setattr(bench, "verify_asset", lambda root, key: (Path(key), {}))
    def stub(*a):
        return {"status": "completed", "predictions_sha256": "test"}, None
    assert bench.execute(args, pass_runner=stub) == 0
    report = json.loads(args.output.read_text())
    assert report["configuration"]["plan"] == ["sequential-reference", "sequential-lut",
        "overlap-lut", "overlap-lut", "sequential-lut", "sequential-reference"]
    assert report["all_pass_pixel_hashes_equal"] is False  # Missing data is not a pass.


def test_missing_input_is_distinguished_and_no_paths_leaked(tmp_path):
    args = args_for(tmp_path/"secret_name.mp4", tmp_path/"failure.json")
    assert bench.execute(args) == 2
    report = json.loads(args.output.read_text())
    assert report["error"]["code"] == "input_video_missing"
    assert report["runs"] == [] and "secret_name" not in args.output.read_text()


def test_next_pose_failure_preserves_partial_metrics(video, tmp_path):
    args, probe = args_for(video, tmp_path/"partial.json"), inspect_recording(video)
    class Bad(Session):
        def run(self, tensor):
            if self.calls == 3:
                raise BenchmarkError("injected_pose_failure")
            return super().run(tensor)
    report, bank = bench.run_pass(args, probe, "overlap-ready-lut",
                                  (lambda: Session("detector"), Bad))
    assert report["status"] == "failed" and bank is None
    assert 0 < report["all_frames"]["frames"] < 6
    assert report["unpaced_loop_fps"] is None
    assert report["error"]["code"] == "injected_pose_failure"
    assert report["hash_scope"] == "completed_prefix"
    assert report["pipeline"]["cleanup"] == {"detector": "owner_released", "pose": "owner_released"}
