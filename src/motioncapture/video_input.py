"""Explicit local-file decode with probed PTS; no inferred constant-frame clock."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import cv2

from motioncapture.contracts import CapturedFrame, FrameIdentity


class VideoInputError(RuntimeError):
    """Cannot establish an honest complete file/clock contract."""


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True, slots=True)
class VideoInfo:
    sha256: str
    width: int
    height: int
    codec: str
    time_base: Fraction
    pts: tuple[int, ...]
    declared_rate: str
    duration_seconds: float
    ffprobe_version: str
    duration_provenance: str

    def presentation_ns(self, sequence: int) -> int:
        # Integer rational conversion; neither frame index/FPS nor processing time.
        return round(self.pts[sequence] * self.time_base * 1_000_000_000)

    def summary(self) -> dict[str, object]:
        intervals: dict[str, int] = {}
        for first, second in zip(self.pts, self.pts[1:]):
            key = str((second - first) * self.time_base)
            intervals[key] = intervals.get(key, 0) + 1
        span = float((self.pts[-1] - self.pts[0]) * self.time_base)
        return {
            "sha256": self.sha256, "width": self.width, "height": self.height,
            "codec": self.codec, "time_base": str(self.time_base),
            "frames": len(self.pts), "declared_frame_rate": self.declared_rate,
            "duration_seconds": self.duration_seconds,
            "duration_provenance": self.duration_provenance,
            "presentation_span_seconds": span,
            "observed_pts_fps": (len(self.pts) - 1) / span,
            "pts_interval_seconds_histogram": intervals,
            "timestamp_basis": "original_frame_pts_times_stream_time_base",
            "ffprobe_version": self.ffprobe_version,
        }


def parse_probe(data: dict, digest: str, version: str) -> VideoInfo:
    streams = data.get("streams", [])
    frames = data.get("frames", [])
    if len(streams) != 1 or not 2 <= len(frames) <= 100_000:
        raise VideoInputError("Select one video stream with 2..100000 frames")
    stream = streams[0]
    width, height = int(stream["width"]), int(stream["height"])
    if width <= 0 or height <= 0 or width * height > 3840 * 2160:
        raise VideoInputError("Unsupported video dimensions")
    rotation = stream.get("tags", {}).get("rotate", "0")
    rotated = any(float(s.get("rotation", 0)) for s in stream.get("side_data_list", []))
    if float(rotation) or rotated:
        raise VideoInputError("Rotated video requires an explicit geometry path; not auto-rotated")
    base = Fraction(stream["time_base"])
    if base <= 0:
        raise VideoInputError("Invalid stream time_base")
    pts = []
    for frame in frames:
        if "pts" not in frame or type(frame["pts"]) is not int:
            raise VideoInputError("Missing original integer PTS; no frame-index fallback")
        if frame.get("width") != width or frame.get("height") != height:
            raise VideoInputError("Dynamic frame dimensions are unsupported")
        if frame.get("interlaced_frame", 0):
            raise VideoInputError("Interlaced input requires a separate explicit mode")
        pts.append(frame["pts"])
    if any(b <= a for a, b in zip(pts, pts[1:])):
        raise VideoInputError("Nonincreasing presentation timestamps")
    reported = stream.get("nb_frames")
    if reported is not None and str(reported).isdigit() and int(reported) != len(pts):
        raise VideoInputError("Declared frame count and fully probed count disagree")
    duration = float(stream.get("duration", float((pts[-1] - pts[0]) * base)))
    if not 0 < duration <= 600:
        raise VideoInputError("This benchmark supports clips up to ten minutes")
    return VideoInfo(digest, width, height, stream["codec_name"], base, tuple(pts),
                     stream.get("avg_frame_rate", "unknown"), duration, version,
                     "stream_metadata" if "duration" in stream else "pts_span_excluding_last_duration")


def probe_video(path: Path) -> VideoInfo:
    if not path.is_file():
        raise VideoInputError("Input must be an existing local video file")
    executable = shutil.which("ffprobe")
    if executable is None:
        raise VideoInputError("ffprobe is required; install FFmpeg explicitly")
    digest = file_hash(path)
    entries = (
        "stream=codec_name,width,height,time_base,avg_frame_rate,duration,nb_frames:"
        "stream_tags=rotate:stream_side_data=rotation:"
        "frame=pts,width,height,interlaced_frame"
    )
    try:
        result = subprocess.run(
            [executable, "-v", "error", "-select_streams", "v:0", "-show_frames",
             "-show_streams", "-show_entries", entries, "-of", "json", str(path.resolve())],
            capture_output=True, timeout=180, check=False,
        )
        version = subprocess.run([executable, "-version"], capture_output=True,
                                 timeout=10, check=True).stdout.decode().splitlines()[0]
    except (OSError, subprocess.SubprocessError) as exc:
        raise VideoInputError("ffprobe failed or timed out") from exc
    if result.returncode or result.stderr.strip():
        raise VideoInputError("ffprobe reported a decode error; input is not accepted")
    if len(result.stdout) > 64 * 1024 * 1024:
        raise VideoInputError("Probe manifest exceeds supported size")
    return parse_probe(json.loads(result.stdout), digest, version)


class TimestampedVideo:
    """Serial, all-frame, unpaced file source; no background prefetch/drop policy.

    FFprobe supplies original PTS. OpenCV's FFmpeg decoder must independently
    agree on each presentation time within 0.1 ms AND produce the exact count.
    Unsupported backend timestamps fail instead of silently substituting a clock.
    """

    def __init__(self, path: Path, info: VideoInfo) -> None:
        self.path, self.info = path, info
        self._capture = None
        self._sequence = 0
        self._stream = uuid.uuid4().hex
        self.complete = False

    def __enter__(self) -> TimestampedVideo:
        if file_hash(self.path) != self.info.sha256:
            raise VideoInputError("Input changed since probe")
        capture = cv2.VideoCapture(str(self.path.resolve()), cv2.CAP_FFMPEG)
        if not capture.isOpened():
            capture.release()
            raise VideoInputError("OpenCV FFmpeg file decoder is unavailable")
        self._capture = capture
        return self

    def read(self) -> CapturedFrame | None:
        if self._capture is None:
            raise VideoInputError("File source is not open")
        if self.complete:
            return None
        ok, image = self._capture.read()
        received_ns = time.monotonic_ns()
        if self._sequence == len(self.info.pts):
            if ok:
                raise VideoInputError("Decoder produced more frames than probe")
            self.complete = True
            return None
        if not ok or image is None:
            raise VideoInputError("Decoder ended before the probed frame count")
        if image.shape != (self.info.height, self.info.width, 3):
            raise VideoInputError("Decoded geometry disagrees with probe")
        pts_ns = self.info.presentation_ns(self._sequence)
        decoded_ms = self._capture.get(cv2.CAP_PROP_POS_MSEC)
        if not abs(decoded_ms - pts_ns / 1_000_000) <= 0.1:
            raise VideoInputError("Decoder PTS is unavailable or disagrees with ffprobe")
        identity = FrameIdentity(
            "file:" + self.info.sha256, self._stream, self._sequence, received_ns,
            presentation_timestamp_ns=pts_ns,
        )
        self._sequence += 1
        return CapturedFrame(identity, image)

    def __exit__(self, _type, _value, _tb) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
