"""Small deterministic tests plus real FFmpeg file decode; no fake live source."""
from __future__ import annotations

import hashlib
import shutil
import subprocess
from fractions import Fraction
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from motioncapture import video_benchmark as bench
from motioncapture.contracts import FrameIdentity, LandmarkResult, LandmarkTimings
from motioncapture.inference_input import ModelClock, RgbConverter
from motioncapture.video_input import TimestampedVideo, VideoInputError, parse_probe, probe_video


def ident(seq, host, pts=None, stream="s"):
    return FrameIdentity("file:test", stream, seq, host, presentation_timestamp_ns=pts)


def test_model_clock_uses_pts_not_decoding_wall_clock():
    clock = ModelClock()
    assert clock.accept(ident(0, 100_000, 3_000_000_000)) == 0
    assert clock.accept(ident(1, 100_001, 3_033_333_333)) == 33
    assert clock.accept(ident(2, 100_002, 3_100_000_000)) == 100


def test_live_model_clock_unchanged():
    clock = ModelClock()
    assert clock.accept(ident(0, 100_000_000)) == 0
    assert clock.accept(ident(1, 135_000_000)) == 35


@pytest.mark.parametrize("next_frame", [ident(1, 140_000_000, 33_000_000),
                                       ident(1, 140_000_000, stream="other"),
                                       ident(0, 140_000_000), ident(1, 100_100_000)])
def test_model_clock_never_changes_domain_or_fabricates_ticks(next_frame):
    clock = ModelClock()
    clock.accept(ident(0, 100_000_000))
    with pytest.raises(ValueError):
        clock.accept(next_frame)


def test_original_negative_pts_retained_and_normalized_only_for_model():
    clock = ModelClock()
    start = ident(0, 100, -100_000_000)
    assert clock.accept(start) == 0 and start.presentation_timestamp_ns == -100_000_000
    assert clock.accept(ident(1, 101, -60_000_000)) == 40


@pytest.mark.parametrize("pts", [1.2, "3", True])
def test_invalid_media_clock_not_silently_coerced(pts):
    with pytest.raises(ValueError):
        ident(0, 1, pts)


def test_rgb_reuse_is_exact_and_borrowed_not_source_alias():
    image = np.arange(360, dtype=np.uint8).reshape(10, 12, 3)
    original = image.copy()
    converter = RgbConverter("reuse")
    first = converter.convert(image)
    assert not np.shares_memory(first, image)
    np.testing.assert_array_equal(first, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    next_image = image[:, ::-1]  # Non-contiguous input is supported by OpenCV.
    second = converter.convert(next_image)
    assert second is first
    np.testing.assert_array_equal(second, cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB))
    np.testing.assert_array_equal(image, original)
    third = converter.convert(image[:5])
    assert not np.shares_memory(third, second)
    converter.close()
    assert converter._buffer is None


def test_allocated_rgb_remains_independent():
    converter = RgbConverter()
    frame = np.zeros((4, 4, 3), np.uint8)
    first = converter.convert(frame)
    frame[:] = 255
    second = converter.convert(frame)
    assert not first.any() and second.all()
    assert not np.shares_memory(first, second)


def probe_fixture():
    return {"streams": [{"width": 96, "height": 64, "codec_name": "h264",
                         "time_base": "1/90000", "avg_frame_rate": "30/1"}],
            "frames": [{"pts": x, "width": 96, "height": 64} for x in (0, 3000, 9000)]}


def test_variable_pts_gap_is_preserved_not_coerced_to_nominal_fps():
    info = parse_probe(probe_fixture(), "hash", "test")
    assert info.presentation_ns(2) == 100_000_000
    assert info.summary()["pts_interval_seconds_histogram"] == {"1/30": 1, "1/15": 1}
    assert info.summary()["observed_pts_fps"] == 20


@pytest.mark.parametrize("case", ["missing", "duplicate", "geometry", "rotation", "count"])
def test_probe_hard_failures(case):
    data = probe_fixture()
    if case == "missing":
        del data["frames"][0]["pts"]
        data["frames"][0]["best_effort_timestamp"] = 0
    elif case == "duplicate":
        data["frames"][1]["pts"] = 0
    elif case == "geometry":
        data["frames"][0]["height"] = 63
    elif case == "rotation":
        data["streams"][0]["tags"] = {"rotate": "90"}
    else:
        data["streams"][0]["nb_frames"] = "4"
    with pytest.raises(VideoInputError):
        parse_probe(data, "hash", "test")


def test_empty_statistics_are_unknown():
    values = bench.distribution([], 1000 / 60)
    assert values["samples"] == 0 and values["p95_ms"] is None


@pytest.mark.parametrize("values", [[float("nan")], [float("inf")], [-1]])
def test_invalid_statistics_fail(values):
    with pytest.raises(ValueError):
        bench.distribution(values, 16.667)


def test_digest_distinguishes_missing_and_nonzero_results():
    from motioncapture.contracts import Blendshape
    empty = LandmarkResult((), (), (), (), (), (), (), ())
    with_face = LandmarkResult((), (), (), (), (), (), (), (Blendshape("jawOpen", 0.0),))
    one, two = hashlib.sha256(), hashlib.sha256()
    bench.result_digest_update(one, empty)
    bench.result_digest_update(two, with_face)
    assert one.digest() != two.digest()


@pytest.fixture
def video(tmp_path):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("Real decoder fixture requires FFmpeg, not a substitute decoder")
    path = tmp_path / "test video.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=96x64:rate=30:duration=0.4", "-c:v", "libx264",
                    "-threads", "1", str(path)], timeout=30, check=True)
    return path


def test_real_video_complete_count_pts_and_cleanup(video):
    info = probe_video(video)
    assert len(info.pts) == 12
    clock = ModelClock()
    frames = []
    with TimestampedVideo(video, info) as source:
        while (frame := source.read()) is not None:
            assert clock.accept(frame.identity) == int(Fraction(frame.identity.sequence, 30) * 1000)
            frames.append(frame)
        assert source.complete
        assert source.read() is None
    assert source._capture is None
    assert len(frames) == 12
    assert not np.shares_memory(frames[0].image_bgr, frames[1].image_bgr)


def test_input_mutation_rejected(video):
    info = probe_video(video)
    video.write_bytes(video.read_bytes() + b"modification")
    with pytest.raises(VideoInputError, match="changed"):
        with TimestampedVideo(video, info):
            pass


def test_real_preprocessing_complete_and_bit_identical(video):
    info = probe_video(video)
    result = bench.preprocess(video, info, 2, 2, 16.667)
    assert result["inference_executed"] is False
    for trial in result["trials"]:
        assert trial["pixel_equal_frames"] == 12
        assert trial["stages"]["allocated"]["samples"] == 10


def test_unavailable_inference_never_substitutes_decode_success(video, monkeypatch):
    def unavailable(*a, **k):
        raise ModuleNotFoundError("test-only missing native runtime")
    monkeypatch.setattr(bench, "create_tracker", unavailable)
    args = bench._parser().parse_args([str(video), "--warmup-frames", "2"])
    report, code = bench.run(args)
    assert code == 2 and report["status"] == "failed"
    assert report["requested_stage"] == "track" and not report["inference_executed"]
    assert not report["trials"]


def test_runner_resets_trackers_abba_and_processes_every_frame(video, monkeypatch):
    owners = []
    empty = LandmarkResult((), (), (), (), (), (), (), ())
    timings = LandmarkTimings(1, 2, 3, 1, 3, 0.1, 4.1)

    class SyntheticTracker:
        provider_name = "TEST ONLY - NOT REAL INFERENCE"
        def __init__(self, mode):
            self.mode = mode
            self.clock = ModelClock()
            self.frames = 0
            self.closed = False
            owners.append(self)
        def open(self):
            pass
        def process(self, frame):
            self.frames += 1
            return SimpleNamespace(identity=frame.identity, result=empty, timings=timings,
                                   model_timestamp_ms=self.clock.accept(frame.identity))
        def close(self):
            self.closed = True
    monkeypatch.setattr(bench, "create_tracker", lambda _p, _s, mode: SyntheticTracker(mode))
    args = bench._parser().parse_args([str(video), "--rounds", "2", "--warmup-frames", "2"])
    report, code = bench.run(args)
    assert code == 0 and report["status"] == "completed"
    assert [t.mode for t in owners] == ["allocated", "reuse", "reuse", "allocated"]
    assert all(t.frames == 12 and t.closed for t in owners)
    for trial in report["trials"]:
        assert trial["stages"]["total_ms"]["samples"] == 10
    assert report["prediction_digests_all_equal"] is True
    assert report["live_60fps_capture_verified"] is False


@pytest.mark.parametrize("failure_phase", ["open", "process"])
def test_primary_failure_preserved_and_tracker_cleanup_attempted(video, monkeypatch, failure_phase):
    cleaned = []
    class FailingTracker:
        def open(self):
            if failure_phase == "open":
                raise ValueError("primary initialization failure")
        def process(self, frame):
            raise ValueError("primary processing failure")
        def close(self):
            cleaned.append(True)
            raise RuntimeError("secondary cleanup failure")
    monkeypatch.setattr(bench, "create_tracker", lambda *a: FailingTracker())
    args = bench._parser().parse_args([str(video), "--warmup-frames", "2"])
    report, code = bench.run(args)
    assert code == 2 and report["error_type"] == "ValueError"
    assert cleaned == [True]
