"""Original-PTS file replay with every-frame DWPose 2D preview and honest telemetry."""
from __future__ import annotations

import argparse
import json
import platform
import queue
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import cv2
import numpy as np

from motioncapture.recording import inspect_recording
from motioncapture.recording_bench import Samples, _source_revision, write_report
from motioncapture.wholebody_catalog import ASSETS, BenchmarkError, verify_asset
from motioncapture.wholebody_onnx import PARTS, RESEARCH_ORT_VERSIONS, OrtModel
from motioncapture.wholebody_optimize_bench import run_pass
from motioncapture.wholebody_pose_batch import PoseBatchModels
from motioncapture.wholebody_pose_batch_pipeline_lab import (
    BatchStagePipeline,
    _fix_batch_timing_labels,
)
from motioncapture.wholebody_replay import ReplayAges

# COCO body indices and the five wrist-to-fingertip chains within each 21-point hand.
BODY_EDGES = ((0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
              (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
              (11, 13), (13, 15), (12, 14), (14, 16))
COLORS = {"body": (70, 220, 90), "feet": (30, 210, 255), "face": (210, 180, 150),
          "left_hand": (255, 180, 40), "right_hand": (200, 70, 255)}
HEADER = 154
WINDOW = "MotionCapture | Recorded 2D preview"


def compose(packet, lines, width=960):
    """Display transform only; never writes into source pixels or model arrays."""
    source = packet.detected.frame.image_bgr
    h, w = source.shape[:2]
    display_width = min(width, w)
    display_height = max(1, round(h * display_width / w))
    image = cv2.resize(source, (display_width, display_height))
    scale = np.array([display_width / w, display_height / h])
    for slot, person in enumerate(packet.people):
        in_frame = (person.valid & (person.xy[:, 0] >= 0) & (person.xy[:, 0] < w)
                    & (person.xy[:, 1] >= 0) & (person.xy[:, 1] < h))
        # Cast only bounded coordinates: out-of-frame estimates may be arbitrarily large.
        points = {i: tuple(np.rint(person.xy[i] * scale).astype(int))
                  for i in np.flatnonzero(in_frame)}
        if points:
            anchor = min(points.values(), key=lambda p: p[1])
            cv2.putText(image, f"slot {slot} (frame only)", anchor,
                        cv2.FONT_HERSHEY_SIMPLEX, .4, (240, 240, 240), 1, cv2.LINE_AA)
        for name, (start, end) in PARTS.items():
            edges = BODY_EDGES if name == "body" else ()
            if name in ("left_hand", "right_hand"):
                edges = tuple((start + a, start + b)
                              for base in (1, 5, 9, 13, 17)
                              for a, b in ((0, base), (base, base + 1),
                                           (base + 1, base + 2), (base + 2, base + 3)))
            for a, b in edges:
                if a in points and b in points:
                    cv2.line(image, points[a], points[b], COLORS[name], 1, cv2.LINE_AA)
            for i in range(start, end):
                if i in points:
                    cv2.circle(image, points[i], 1 if name == "face" else 2,
                               COLORS[name], -1, cv2.LINE_AA)
    canvas = np.full((display_height + HEADER, width, 3), 22, np.uint8)
    left = (width - display_width) // 2
    canvas[HEADER:, left:left + display_width] = image
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (12, 23 + i * 24), cv2.FONT_HERSHEY_SIMPLEX,
                    .49, (225, 230, 235), 1, cv2.LINE_AA)
    return canvas


class Preview:
    def __init__(self, probe, count, provider, *, snapshot_dir=None, snapshot_frames=(),
                 display_backend="opencv"):
        self.probe, self.count, self.provider = probe, count, provider
        self.source_rate = probe.summary()["pts_span_fps"]
        self.frames = 0
        self.cleanup = "not_opened"
        self.compose_ms, self.ui_ms, self.intervals = Samples(), Samples(), Samples()
        self.ages = ReplayAges(count)
        self.recent = deque(maxlen=120)
        self.first_ns = self.last_ns = None
        self.snapshot_dir = snapshot_dir
        self.snapshot_frames = frozenset(snapshot_frames)
        self.snapshots_written = []
        self.snapshot_ms = Samples()
        self.display_backend = display_backend
        self.display = None
        if display_backend == "sdl":
            from motioncapture.recorded_display import SDLDisplay

            self.display = SDLDisplay(WINDOW)
        elif display_backend != "opencv":
            raise BenchmarkError("unknown_display_backend")

    def __call__(self, packet):
        frame = packet.detected.frame
        if frame.identity.sequence != self.frames:
            raise BenchmarkError("preview_identity_mismatch")
        if self.cleanup == "not_opened":
            self.cleanup = "open_attempted"
            if self.display is None:
                cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
            self.cleanup = "open"
        age = (time.perf_counter_ns() - packet.detected.source_release.due_ns) / 1e6
        fps = ((len(self.recent) - 1) * 1e9 / (self.recent[-1] - self.recent[0])
               if len(self.recent) > 1 else None)
        rate = "warming up" if fps is None else f"{fps:.2f} Hz"
        valid = {name: sum(int(p.valid[a:b].sum()) for p in packet.people)
                 for name, (a, b) in PARTS.items()}
        source_s = float((frame.identity.pts - self.probe.pts[0]) * frame.identity.time_base)
        lines = [
            f"RECORDED FILE | {self.display_backend} | original PTS | 2D | Esc / Q to stop",
            f"DWPose-m | {self.provider} | same-frame batch 2 | every source frame",
            f"Source {self.probe.width}x{self.probe.height} | "
            f"{self.source_rate:.3f} Hz"
            f" | frame {self.frames+1}/{self.count} | {source_s:.2f}s",
            f"UI submissions {rate} | result age before drawing {age:.1f} ms | "
            f"detected slots {len(packet.people)}",
            f"Valid points: body {valid['body']} / feet {valid['feet']} / "
            f"left hand {valid['left_hand']} / right hand {valid['right_hand']} / "
            f"face {valid['face']}",
            "Green body | yellow feet | blue left hand | pink right hand | "
            "No 3D / retargeting / facial expressions",
        ]
        start = time.perf_counter_ns()
        image = compose(packet, lines)
        composed = time.perf_counter_ns()
        if frame.identity.sequence in self.snapshot_frames:
            path = self.snapshot_dir / f"frame-{frame.identity.sequence:06d}.png"
            if path.exists() or not cv2.imwrite(str(path), image):
                raise BenchmarkError("preview_snapshot_write_failed")
            self.snapshots_written.append(frame.identity.sequence)
            self.snapshot_ms.add((time.perf_counter_ns() - composed) / 1e6)
        ui_started = time.perf_counter_ns()
        key = -1
        if self.display is None:
            cv2.imshow(WINDOW, image)
            key = cv2.waitKey(1) & 0xff
        else:
            self.display.show(image)
            # Drain events as part of measured submission; count the frame even
            # when the quit event interrupts immediately after its blit.
            try:
                self.display.poll()
            except KeyboardInterrupt:
                key = 27
        returned = time.perf_counter_ns()
        self.compose_ms.add((composed - start) / 1e6)
        self.ui_ms.add((returned - ui_started) / 1e6)
        self.ages.add(frame.identity, packet.detected.source_release, returned,
                      people=len(packet.people), pose_stage_ms=packet.times["pose_stage_ms"])
        if self.last_ns is not None:
            self.intervals.add((returned - self.last_ns) / 1e6)
        if self.first_ns is None:
            self.first_ns = returned
        self.last_ns = returned
        self.recent.append(returned)
        self.frames += 1
        if key in (27, ord("q")) or (self.display is None
                and cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1):
            raise KeyboardInterrupt

    def close(self):
        if self.cleanup in ("open", "open_attempted"):
            self.cleanup = "failed"
            if self.display is None:
                cv2.destroyWindow(WINDOW)
                cv2.waitKey(1)
            else:
                self.display.close()
            self.cleanup = "owner_released"

    def poll(self):
        if self.cleanup == "open":
            if self.display is not None:
                self.display.poll()
                return
            key = cv2.waitKey(1) & 0xff
            if key in (27, ord("q")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                raise KeyboardInterrupt

    def summary(self):
        ages = self.ages.summary()
        ages["scope"] = "scheduled file release to display submit/event return; NOT photon latency"
        return {
            "submitted_frames": self.frames, "requested_frames": self.count,
            "cleanup": self.cleanup, "width_limit": 960,
            "display": (self.display.metadata if self.display is not None
                        else {"backend": "opencv", "version": cv2.__version__,
                              "driver": cv2.currentUIFramework()}),
            "snapshots_written": self.snapshots_written,
            "snapshot_write_ms": self.snapshot_ms.summary(1000/60),
            "mean_submission_rate_hz": ((self.frames - 1) * 1e9 / (self.last_ns - self.first_ns)
                                        if self.frames > 1 else None),
            "compose_ms": self.compose_ms.summary(1000/60),
            "display_submit_and_event_pump_ms": self.ui_ms.summary(1000/60),
            "submission_interval_ms": self.intervals.summary(1000/60),
            "ui_return_ages": ages,
            "physical_display_timing_verified": False,
            "valid_point_counts_are_accuracy": False,
        }


def run_with_preview(args, probe, factories, preview, *, runner=run_pass):
    """Inference consumer on one worker; all OpenCV window work stays on main.

    The queue holds one validated packet. A slow display backpressures inference;
    it never replaces a packet with a newer frame. Cancellation wakes blocked
    putters and joins model/decoder owners before the caller closes the window.
    """
    packets = queue.Queue(maxsize=1)
    stop = threading.Event()
    waits = Samples()
    error = None

    def enqueue(packet):
        began = time.perf_counter_ns()
        try:
            while not stop.is_set():
                try:
                    packets.put(packet, timeout=.05)
                    return
                except queue.Full:
                    continue
            raise KeyboardInterrupt
        finally:
            waits.add((time.perf_counter_ns() - began) / 1e6)

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="recorded-inference") as owner:
        future = owner.submit(runner, args, probe, "source-pts-ready-cvlut", factories,
                              pipeline_factory=BatchStagePipeline, packet_consumer=enqueue)
        try:
            while True:
                if future.done() and future.result()[0]["status"] != "completed":
                    break
                try:
                    packet = packets.get(timeout=.02)
                except queue.Empty:
                    if future.done():
                        break
                    preview.poll()
                    continue
                preview(packet)
                del packet
        except BaseException as exc:
            error = exc
        finally:
            stop.set()
        row, reference = future.result()
    row["inference_status_before_preview_outcome"] = row["status"]
    if error is not None and row["status"] != "failed":
        row["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        row["error"] = {"phase": "preview", "type": type(error).__name__,
                        "code": getattr(error, "code", None)}
        row["paced_loop_fps"] = row["unpaced_loop_fps"] = None
    row["preview_handoff"] = {
        "queue_capacity_packets": 1,
        "queue_wait_ms": waits.summary(1000/60),
        "frames_replaced": 0,
        "undisplayed_queued_packets_at_exit": packets.qsize(),
        "inference_owner_joined": True,
        "policy": "bounded FIFO with backpressure; no dropping",
    }
    row["timing_scope"]["loop"] = "inference, verification, bounded preview enqueue; UI on caller"
    return row, reference


def execute(args):
    manifest = args.output.with_suffix(".manifest.json")
    prepared = args.output.with_suffix(".prepared.json")
    snapshot_dir = args.output.with_suffix(".snapshots") if args.snapshot_frame else None
    if any(path.exists() for path in (args.output, manifest, prepared)) or (
            snapshot_dir is not None and snapshot_dir.exists()):
        raise FileExistsError(args.output)
    report = {
        "schema_version": 2, "mode": "recorded_file_2d_preview", "status": "running",
        "error": None, "cleanup_errors": [],
        "privacy": {"frames_written": bool(args.snapshot_frame), "coordinates_written": False,
                    "annotated_snapshots_explicitly_selected": args.snapshot_frame,
                    "audio_processed": False},
        "live_60fps_verified": False, "accuracy_verified": False,
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "source": _source_revision()},
    }
    preview = None
    phase = "inspect"
    write_report(manifest, report)
    try:
        phase = "runtime_configuration"
        expected_ort = getattr(args, "expected_ort_version", None)
        if expected_ort is not None:
            import onnxruntime as ort

            report["runtime"]["onnxruntime_version"] = ort.__version__
            report["runtime"]["expected_onnxruntime_version"] = expected_ort
            if ort.__version__ != expected_ort:
                raise BenchmarkError("requested_onnxruntime_version_mismatch")
        phase = "display_configuration"
        cv_threads = getattr(args, "opencv_threads", None)
        if cv_threads is not None:
            cv2.setNumThreads(cv_threads)
        report["runtime"]["opencv_threads"] = cv2.getNumThreads()
        report["runtime"]["opencv_threads_requested"] = cv_threads
        phase = "inspect"
        probe = inspect_recording(args.input)
        count = min(args.max_frames or len(probe.pts), len(probe.pts))
        if any(i < 0 or i >= count for i in args.snapshot_frame):
            raise BenchmarkError("snapshot_frame_outside_run")
        if snapshot_dir is not None:
            snapshot_dir.mkdir(parents=True, exist_ok=False)
        report["source"] = probe.summary()
        phase = "assets"
        assets = {key: verify_asset(args.asset_dir, key) for key in ("yolox-tiny", "dwpose-m")}
        report["assets"] = {key: value[1] for key, value in assets.items()}
        report["configuration"] = {"provider": args.provider,
                                   "allow_cpu_partitions": args.allow_cpu_partitions,
                                   "pose_batch_size": 2, "pacing": "original_pts",
                                   "preview": "display", "model": "dwpose-m",
                                   "display_backend": getattr(args, "display_backend", "opencv")}
        factories = (
            partial(OrtModel, assets["yolox-tiny"][0], ASSETS["yolox-tiny"].shape,
                    args.provider, allow_cpu=args.allow_cpu_partitions, threads=4),
            partial(PoseBatchModels, assets["dwpose-m"][0], ASSETS["dwpose-m"].shape,
                    args.provider, allow_cpu=args.allow_cpu_partitions, threads=4),
        )
        args.model, args.suite = "dwpose-m", "pose-parallel"
        args.detector_provider = args.pose_provider = args.provider
        args.ort_threads = args.pose_intra_op_threads = 4
        label = "CoreML ALL | CPU partitions allowed" if args.provider == "coreml-all" else "CPU"
        if expected_ort is not None:
            label += f" | ORT {expected_ort}"
        preview = Preview(probe, count, label, snapshot_dir=snapshot_dir,
                          snapshot_frames=args.snapshot_frame,
                          display_backend=getattr(args, "display_backend", "opencv"))
        phase = "replay"
        write_report(prepared, report)
        replay_started = time.perf_counter_ns()
        row, _ = run_with_preview(args, probe, factories, preview)
        report["replay_wall_s_including_setup_and_ui_drain"] = (
            time.perf_counter_ns() - replay_started) / 1e9
        _fix_batch_timing_labels(row)
        report["run"] = row
        report["status"], report["error"] = row["status"], row["error"]
        if row["status"] == "completed" and preview.frames != row["all_frames"]["frames"]:
            raise BenchmarkError("preview_frame_coverage_mismatch")
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["error"] = {"phase": phase, "type": type(exc).__name__,
                           "code": getattr(exc, "code", None)}
    finally:
        if preview is not None:
            try:
                preview.close()
            except BaseException as exc:
                error = {"phase": "preview_cleanup", "type": type(exc).__name__}
                report["cleanup_errors"].append(error)
                if report["status"] == "completed":
                    report["status"], report["error"] = "failed", error
            report["preview"] = preview.summary()
        # macOS reports bytes, Linux KiB. This is a lifetime peak, including setup.
        report["process_peak_rss_bytes"] = None
        if platform.system() in ("Darwin", "Linux"):
            import resource

            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            report["process_peak_rss_bytes"] = (
                peak if platform.system() == "Darwin" else peak * 1024)
        report["memory_scope"] = "process lifetime peak including model setup"
        write_report(args.output, report)
    print(json.dumps({"status": report["status"], "error": report["error"]}), flush=True)
    return 0 if report["status"] == "completed" else 130 if report["status"] == "interrupted" else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--provider", choices=("coreml-all", "cpu"), required=True)
    p.add_argument("--allow-cpu-partitions", action="store_true")
    p.add_argument("--research-only", action="store_true", required=True)
    p.add_argument("--asset-dir", type=Path, default=Path("models/wholebody"))
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--decode-threads", type=int, default=0)
    p.add_argument("--opencv-threads", type=int, default=None,
                   help="Explicit OpenCV CPU budget; omitted keeps runtime default")
    p.add_argument("--display-backend", choices=("opencv", "sdl"), default="opencv")
    p.add_argument("--expected-ort-version", choices=sorted(RESEARCH_ORT_VERSIONS),
                   default="1.22.1", help="Reject a different installed runtime version")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--snapshot-frame", type=int, action="append", default=[],
                   help="Explicitly save this zero-based frame's annotated preview as PNG")
    args = p.parse_args()
    if not 0 <= args.max_frames <= 120000 or not 0 <= args.decode_threads <= 64:
        p.error("Invalid frame or decoder thread limit")
    if args.opencv_threads is not None and not 1 <= args.opencv_threads <= 64:
        p.error("OpenCV thread limit must be 1..64")
    if args.provider == "coreml-all" and not args.allow_cpu_partitions:
        p.error("This selected CoreML path requires --allow-cpu-partitions")
    if args.provider == "cpu" and args.allow_cpu_partitions:
        p.error("CPU partitions flag is only meaningful with CoreML")
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
