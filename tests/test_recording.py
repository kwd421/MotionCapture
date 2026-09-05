from __future__ import annotations

import copy
import subprocess
from fractions import Fraction

import cv2
import numpy as np
import pytest

from motioncapture.recording import (
    RecordedDecoder,
    RecordedIdentity,
    RecordingProbe,
    inspect_recording,
)


def metadata():
    return {"streams": [{"width": 64, "height": 48, "codec_name": "h264",
                         "time_base": "1/1000", "start_pts": 100, "nb_frames": "3",
                         "duration_ts": 84, "avg_frame_rate": "50/1"}],
            "frames": [{"pts": t, "width": 64, "height": 48} for t in (100, 117, 150)]}


def test_source_time_is_rational_and_not_host_receive():
    identity = RecordedIdentity("sha", "stream", 0, 1517, Fraction(1, 90000))
    assert identity.source_ns == Fraction(1517 * 1_000_000_000, 90000)
    assert not hasattr(identity, "received_ns")


@pytest.mark.parametrize("defect", ["missing_pts", "duplicate", "reverse", "dimensions",
                                    "count", "rotation", "time_base"])
def test_invalid_probe_is_not_turned_into_cfr(defect):
    data = metadata()
    if defect == "missing_pts":
        del data["frames"][1]["pts"]
    elif defect == "duplicate":
        data["frames"][1]["pts"] = 100
    elif defect == "reverse":
        data["frames"][1]["pts"] = 90
    elif defect == "dimensions":
        data["frames"][1]["width"] = 32
    elif defect == "count":
        data["streams"][0]["nb_frames"] = "4"
    elif defect == "rotation":
        data["streams"][0]["side_data_list"] = [{"rotation": 90}]
    else:
        data["streams"][0]["time_base"] = "0/1"
    with pytest.raises((ValueError, KeyError)):
        RecordingProbe.parse("sha", data)


def test_probe_fps_is_derived_from_pts_not_header():
    probe = RecordingProbe.parse("sha", metadata())
    assert probe.summary()["pts_span_fps"] == 40
    assert probe.summary()["declared_average_rate"] == "50/1"
    assert probe.summary()["interval_ms"]["max"] == 33


@pytest.fixture
def tiny_vfr(tmp_path):
    path = tmp_path / "vfr.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i",
        "testsrc2=size=64x48:rate=30:duration=0.2", "-vf",
        "setpts=(N+floor(N/2))/(30*TB)", "-fps_mode", "passthrough",
        "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p", str(path),
    ], check=True, timeout=15)
    return path


def test_real_ffmpeg_and_opencv_preserve_variable_pts_and_pixels(tiny_vfr):
    probe = inspect_recording(tiny_vfr)
    assert len(set(b - a for a, b in zip(probe.pts, probe.pts[1:]))) > 1
    with RecordedDecoder(tiny_vfr, probe, threads=1) as decoder:
        frames = list(decoder)
        assert decoder.complete
    assert [x.identity.pts for x in frames] == list(probe.pts)
    assert len({x.identity.stream_id for x in frames}) == 1
    first = frames[0].image_bgr.copy()
    with RecordedDecoder(tiny_vfr, probe, threads=4) as decoder:
        second = list(decoder)
    for a, b in zip(frames, second):
        np.testing.assert_array_equal(a.image_bgr, b.image_bgr)
    np.testing.assert_array_equal(frames[0].image_bgr, first)


@pytest.mark.parametrize("defect", ["early_eof", "bad_pts", "extra_frame"])
def test_decoder_mismatch_is_fatal_and_releases(monkeypatch, tmp_path, defect):
    probe = RecordingProbe.parse("sha", metadata())

    class Capture:
        index = 0
        released = False

        def isOpened(self):
            return True

        def getBackendName(self):
            return "FFMPEG"

        def set(self, *_):
            return True

        def get(self, prop):
            if prop == cv2.CAP_PROP_N_THREADS:
                return 1
            if prop == cv2.CAP_PROP_POS_MSEC:
                if defect == "bad_pts":
                    return -999
                return float((probe.pts[self.index - 1] - 100) * probe.time_base * 1000)
            return 0

        def read(self):
            self.index += 1
            if defect == "early_eof" or (self.index > 3 and defect != "extra_frame"):
                return False, None
            return True, np.zeros((48, 64, 3), np.uint8)

        def release(self):
            self.released = True

    capture = Capture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **kw: capture)
    decoder = RecordedDecoder(tmp_path / "placeholder", probe, threads=1)
    with pytest.raises(ValueError):
        with decoder:
            list(decoder)
    assert capture.released and not decoder.complete


def test_probe_does_not_mutate_source_metadata():
    data = metadata()
    before = copy.deepcopy(data)
    RecordingProbe.parse("sha", data)
    assert data == before
