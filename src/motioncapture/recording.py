"""Explicit recorded input with integer PTS; never poses as a live camera."""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class RecordedIdentity:
    source_id: str
    stream_id: str
    sequence: int
    pts: int
    time_base: Fraction

    def __post_init__(self) -> None:
        if not self.source_id or not self.stream_id or self.sequence < 0:
            raise ValueError("Invalid recording identity")
        if type(self.pts) is not int or type(self.sequence) is not int:
            raise ValueError("PTS and sequence must be integers")
        if not isinstance(self.time_base, Fraction) or self.time_base <= 0:
            raise ValueError("Recording time base must be a positive rational")

    @property
    def source_ns(self) -> Fraction:
        return self.pts * self.time_base * 1_000_000_000


@dataclass(frozen=True, slots=True)
class RecordedFrame:
    identity: RecordedIdentity
    image_bgr: np.ndarray = field(repr=False, compare=False)
    decode_ms: float

    def __post_init__(self) -> None:
        image = self.image_bgr
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.ndim != 3 or image.shape[2] != 3 or 0 in image.shape):
            raise ValueError("Recorded frame must be nonempty uint8 BGR")


@dataclass(frozen=True, slots=True)
class RecordingProbe:
    sha256: str
    width: int
    height: int
    codec: str
    time_base: Fraction
    start_pts: int
    pts: tuple[int, ...]
    declared_rate: str
    duration_s: float | None

    @classmethod
    def parse(cls, sha256: str, metadata: dict) -> RecordingProbe:
        streams = metadata.get("streams", [])
        frames = metadata.get("frames", [])
        if len(streams) != 1 or not 2 <= len(frames) <= 120_000:
            raise ValueError("Require one video stream and 2..120000 decoded frames")
        stream = streams[0]
        rotation = [x.get("rotation", 0) for x in stream.get("side_data_list", [])]
        if any(float(x) % 360 != 0 for x in rotation):
            raise ValueError("Rotated recordings require an explicit orientation adapter")
        pts = tuple(x["pts"] for x in frames)  # Missing PTS is not synthesized.
        if any(type(x) is not int for x in pts) or any(b <= a for a, b in zip(pts, pts[1:])):
            raise ValueError("Decoded PTS must strictly increase")
        width, height = int(stream["width"]), int(stream["height"])
        if width <= 0 or height <= 0:
            raise ValueError("Invalid video dimensions")
        if any((x["width"], x["height"]) != (width, height) for x in frames):
            raise ValueError("Mid-stream dimension changes are unsupported")
        count = stream.get("nb_frames")
        if count is not None and int(count) != len(pts):
            raise ValueError("Declared and decoded frame counts disagree")
        tb = Fraction(stream["time_base"])
        if tb <= 0:
            raise ValueError("Invalid video time base")
        duration = stream.get("duration_ts")
        return cls(sha256, width, height, stream["codec_name"], tb,
                   int(stream["start_pts"]), pts, stream.get("avg_frame_rate", "unknown"),
                   float(int(duration) * tb) if duration is not None else None)

    def summary(self) -> dict:
        intervals = np.asarray([float((b - a) * self.time_base * 1000)
                                for a, b in zip(self.pts, self.pts[1:])])
        return {
            "sha256": self.sha256, "dimensions": [self.width, self.height],
            "codec": self.codec, "frames": len(self.pts), "duration_s": self.duration_s,
            "time_base": str(self.time_base), "start_pts": self.start_pts,
            "first_pts": self.pts[0], "last_pts": self.pts[-1],
            "declared_average_rate": self.declared_rate,
            "pts_span_fps": (len(self.pts) - 1) / float(
                (self.pts[-1] - self.pts[0]) * self.time_base),
            "interval_ms": dict(zip(("min", "p50", "p95", "p99", "max"),
                                    map(float, np.quantile(intervals, [0, .5, .95, .99, 1])))),
            "timestamp_kind": "ffprobe_integer_pts_and_rational_time_base",
            "sensor_exposure_uniqueness_verified": False,
        }


def inspect_recording(path: Path, ffprobe: str = "ffprobe") -> RecordingProbe:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    command = [
        ffprobe, "-v", "error", "-threads", "4", "-select_streams", "v:0",
        "-show_entries",
        "stream=index,codec_name,width,height,time_base,start_pts,duration_ts,nb_frames,"
        "avg_frame_rate:stream_side_data=rotation:frame=pts,width,height",
        "-of", "json", str(path.resolve()),
    ]
    proc = subprocess.run(command, capture_output=True, timeout=600, check=False)
    if proc.returncode or proc.stderr.strip():
        raise ValueError(f"FFprobe failed or reported video errors (exit={proc.returncode})")
    return RecordingProbe.parse(digest.hexdigest(), json.loads(proc.stdout))


class RecordedDecoder:
    """One sequential FFmpeg decoder; checks PTS/dimensions/count at every frame."""

    def __init__(self, path: Path, probe: RecordingProbe, *, threads: int = 0) -> None:
        if not 0 <= threads <= 64:
            raise ValueError("Decoder threads must be in 0..64")
        self.path, self.probe, self.threads = path, probe, threads
        self._capture = None
        self.actual_threads: float | None = None
        self._stream = uuid.uuid4().hex
        self._consumed = False
        self.complete = False

    def __enter__(self) -> RecordedDecoder:
        if self._capture is not None or self._consumed:
            raise ValueError("Decoder cannot be reused")
        cap = cv2.VideoCapture(str(self.path.resolve()), cv2.CAP_FFMPEG,
                               [cv2.CAP_PROP_N_THREADS, self.threads])
        try:
            if not cap.isOpened() or cap.getBackendName() != "FFMPEG":
                raise ValueError("Requested FFmpeg file decoder is unavailable")
            self.actual_threads = cap.get(cv2.CAP_PROP_N_THREADS)
            if self.threads and self.actual_threads != self.threads:
                raise ValueError("Requested decoder thread count was not applied")
            cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
            if cap.get(cv2.CAP_PROP_ORIENTATION_AUTO) != 0:
                raise ValueError("Cannot disable automatic file orientation")
        except BaseException:
            cap.release()
            raise
        self._capture = cap
        return self

    def __iter__(self) -> Iterator[RecordedFrame]:
        import time

        cap = self._capture
        if cap is None or self._consumed:
            raise ValueError("Open a new decoder for each pass")
        self._consumed = True
        for index, pts in enumerate(self.probe.pts):
            start = time.perf_counter_ns()
            ok, image = cap.read()
            decode_ms = (time.perf_counter_ns() - start) / 1_000_000
            if not ok or image is None:
                raise ValueError(f"Decoder stopped early at frame {index}")
            if image.shape != (self.probe.height, self.probe.width, 3):
                raise ValueError(f"Decoded dimensions disagree at frame {index}")
            expected_ms = float((pts - self.probe.start_pts) * self.probe.time_base * 1000)
            actual_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            tolerance_ms = max(0.001, float(self.probe.time_base * 1000) / 2)
            if not math.isfinite(actual_ms) or abs(actual_ms - expected_ms) > tolerance_ms:
                raise ValueError(f"Decoder/FFprobe PTS mismatch at frame {index}")
            identity = RecordedIdentity(self.probe.sha256, self._stream, index,
                                        pts, self.probe.time_base)
            yield RecordedFrame(identity, image, decode_ms)
        ok, _ = cap.read()
        if ok:
            raise ValueError("Decoder returned more frames than FFprobe")
        self.complete = True

    def __exit__(self, *_: object) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
