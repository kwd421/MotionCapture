"""Opt-in optimized live path. The original motioncapture-demo stays unchanged."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from functools import partial
from pathlib import Path
from typing import Any

import cv2

from motioncapture.capture import CameraRequest, LocalCamera
from motioncapture.errors import PrototypeError, SessionRecordError
from motioncapture.model_assets import DEFAULT_MODEL_DIR
from motioncapture.pipeline import FramePipeline, Tracker
from motioncapture.profiling import PerformanceSession, revision_metadata
from motioncapture.runtime import CaptureRuntime

WINDOW_NAME = "MotionCapture - Live Performance Candidate"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--capture-timeout", type=float, default=10.0)
    parser.add_argument("--task-scheduling", choices=("parallel", "serial"), default="parallel")
    parser.add_argument(
        "--pipeline-scheduling", choices=("overlap", "sequential"), default="overlap",
    )
    parser.add_argument("--preview-mode", choices=("display", "native"), default="display")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--session-dir", type=Path, default=Path("sessions"))
    parser.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--headless-frames", type=int, default=0)
    parser.add_argument(
        "--max-frames", type=int, default=0, help="live frame limit; zero is unlimited",
    )
    parser.add_argument("--self-check", action="store_true")
    return parser


def _new_tracker(model_dir: Path, task_scheduling: str) -> Tracker:
    from motioncapture.landmarkers import MediaPipeLandmarkTracker

    return MediaPipeLandmarkTracker(model_dir, task_scheduling=task_scheduling)


def _preview_tools(mode: str) -> tuple[Any, Any]:
    from motioncapture.render import RenderMetrics, compose_preview

    if mode == "native":
        return compose_preview, RenderMetrics
    from motioncapture.preview import DisplayPreview

    return DisplayPreview().compose, RenderMetrics


def _fps(timestamps: deque[int]) -> float:
    if len(timestamps) < 2:
        return 0.0
    span = timestamps[-1] - timestamps[0]
    return (len(timestamps) - 1) * 1_000_000_000 / span if span > 0 else 0.0


def run(args: argparse.Namespace) -> int:
    if args.self_check:
        # Preserve the baseline self-check: models only; no camera is opened.
        from motioncapture.app import _self_check

        return _self_check(args.model_dir, args.task_scheduling)
    if args.headless_frames < 0 or args.max_frames < 0:
        raise ValueError("Frame limits must be nonnegative")
    if args.headless_frames and args.max_frames:
        raise ValueError("Select headless-frames or max-frames, not both")
    if args.width <= 0 or args.height <= 0 or not math.isfinite(args.fps) or args.fps <= 0:
        raise ValueError("Camera dimensions and FPS must be positive and finite")
    limit = args.headless_frames or args.max_frames
    request = CameraRequest(args.camera_index, args.width, args.height, args.fps)
    capture = CaptureRuntime(LocalCamera(request), timeout=args.capture_timeout)
    pipeline = FramePipeline(
        capture, partial(_new_tracker, args.model_dir, args.task_scheduling),
        timeout=args.capture_timeout,
    )
    session = PerformanceSession(
        request, mirror=args.mirror, inference_provider="unknown (not initialized)",
        task_scheduling=args.task_scheduling,
    )
    session.pipeline_scheduling = args.pipeline_scheduling
    session.preview_mode = args.preview_mode
    session.headless = bool(args.headless_frames)
    session.source = revision_metadata(Path(__file__).resolve().parents[2])
    times: deque[int] = deque(maxlen=60)
    window_created = False
    manifest: Path | None = None
    try:
        compose, metrics_type = _preview_tools(args.preview_mode)
        with pipeline:
            observation = pipeline.observation
            session.attach_camera(observation)
            session.inference_provider = pipeline.provider_name
            if not args.headless_frames:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                window_created = True
                cv2.resizeWindow(WINDOW_NAME, 1360, 540)
            pipeline.request()
            while True:
                iteration_ns = time.monotonic_ns()
                packet = pipeline.receive()
                received_ns = time.monotonic_ns()
                frame, output = packet.frame, packet.output
                session.extra["result_wait"].observe((received_ns - iteration_ns) / 1_000_000.0)
                session.extra["result_residence"].observe(
                    (received_ns - packet.completed_ns) / 1_000_000.0,
                )
                session.extra["capture_wait"].observe(packet.capture_wait_ms)
                another = not limit or session.frames + 1 < limit
                # Start the next inference BEFORE any composition/HighGUI work.
                if another and args.pipeline_scheduling == "overlap":
                    pipeline.request()
                times.append(packet.completed_ns)
                metrics = metrics_type(
                    session_id=session.session_id,
                    camera_index=observation.index, camera_backend=observation.backend,
                    frame_width=observation.width, frame_height=observation.height,
                    nominal_fps=observation.nominal_fps, processing_fps=_fps(times),
                    inference_ms=output.timings.inference_wall_ms,
                    pose_ms=output.timings.pose_ms, hands_ms=output.timings.hands_ms,
                    face_ms=output.timings.face_ms, frame_sequence=frame.identity.sequence,
                    timestamp_ms=output.model_timestamp_ms, provider=pipeline.provider_name,
                )
                preview_started = time.monotonic_ns()
                preview = compose(frame.image_bgr, output.result, metrics, mirror=args.mirror)
                snapshot = capture.snapshot()
                capture_fps = (
                    f"{snapshot.mean_capture_fps:.1f}"
                    if snapshot.mean_capture_fps is not None else "unknown"
                )
                cv2.putText(
                    preview,
                    f"{args.pipeline_scheduling}/{args.preview_mode} | capture {capture_fps} FPS"
                    f" | replaced {snapshot.replaced} | queue {packet.capture_queue_ms:.1f} ms",
                    (28, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (245, 245, 245), 1, cv2.LINE_AA,
                )
                preview_finished = time.monotonic_ns()
                pipeline.check()
                session.observe(
                    output.result, output.timings,
                    preview_composition_ms=(preview_finished - preview_started) / 1_000_000.0,
                    host_post_receive_total_ms=(preview_finished - frame.identity.received_ns)
                    / 1_000_000.0,
                    frame_timestamp_ns=frame.identity.received_ns,
                    capture_queue_ms=packet.capture_queue_ms, frame_identity=output.identity,
                )
                session.completed_at(packet.completed_ns)
                key = -1
                if not args.headless_frames:
                    pipeline.check()
                    submit_started = time.monotonic_ns()
                    cv2.imshow(WINDOW_NAME, preview)
                    submitted = time.monotonic_ns()
                    key = cv2.waitKey(1) & 0xFF
                    pumped = time.monotonic_ns()
                    session.extra["presentation_submit"].observe(
                        (submitted - submit_started) / 1_000_000.0,
                    )
                    session.extra["event_pump"].observe((pumped - submitted) / 1_000_000.0)
                    session.extra["host_receive_to_gui_submit"].observe(
                        (submitted - frame.identity.received_ns) / 1_000_000.0,
                    )
                    session.extra["host_receive_to_event_pump"].observe(
                        (pumped - frame.identity.received_ns) / 1_000_000.0,
                    )
                session.extra["iteration_wall"].observe(
                    (time.monotonic_ns() - iteration_ns) / 1_000_000.0,
                )
                if key in {27, ord("q"), ord("Q")} or not another:
                    break
                if args.pipeline_scheduling == "sequential":
                    pipeline.request()
        session.finish("completed")
    except KeyboardInterrupt:
        session.finish("interrupted", "KeyboardInterrupt")
        raise
    except BaseException as exc:
        session.finish("failed", f"{type(exc).__name__}: {exc}")
        raise
    finally:
        session.attach_capture(capture.snapshot())
        session.pipeline_snapshot = pipeline.snapshot()
        primary = sys.exception()
        cleanup_error = None
        if window_created:
            try:
                cv2.destroyAllWindows()
            except Exception as exc:
                cleanup_error = exc
                session.finish("failed", f"Window cleanup failed: {exc}")
                if primary is not None:
                    primary.add_note(str(exc))
        try:
            manifest = session.write(args.session_dir)
        except SessionRecordError as exc:
            if primary is None and cleanup_error is None:
                raise
            (primary if primary is not None else cleanup_error).add_note(str(exc))
            print(json.dumps({"status": "session_record_failed", "error": str(exc)}),
                  file=sys.stderr)
        if primary is None and cleanup_error is not None:
            raise cleanup_error
    print(json.dumps({"status": session.terminal_status, "frames": session.frames,
                      "manifest": str(manifest) if manifest is not None else None}))
    return 0


def main() -> int:
    try:
        return run(_parser().parse_args())
    except KeyboardInterrupt:
        return 130
    except (PrototypeError, ValueError, ModuleNotFoundError) as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__,
                          "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
