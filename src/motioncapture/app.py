"""Live MacBook camera body, hand/finger, and facial landmark prototype."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from pathlib import Path

import cv2

from motioncapture.capture import CameraRequest, LocalCamera
from motioncapture.errors import PrototypeError, SessionRecordError
from motioncapture.landmarkers import MediaPipeLandmarkTracker
from motioncapture.model_assets import DEFAULT_MODEL_DIR, require_models
from motioncapture.render import RenderMetrics, compose_preview
from motioncapture.runtime import CaptureRuntime
from motioncapture.session import SessionRecord

WINDOW_NAME = "MotionCapture - Live 2D Body + Face Prototype"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--capture-timeout", type=float, default=10.0,
        help="maximum seconds per capture startup, frame wait, or shutdown wait",
    )
    parser.add_argument(
        "--task-scheduling",
        choices=("serial", "parallel"),
        default="parallel",
        help="explicitly select serial or parallel Pose/Hands/Face task execution",
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--session-dir", type=Path, default=Path("sessions"))
    parser.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--headless-frames",
        type=int,
        default=0,
        help="infer exactly this many selected real camera frames without opening a window",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="verify the pinned models and initialize the tracker without opening a camera",
    )
    return parser


def _self_check(model_dir: Path, task_scheduling: str) -> int:
    verified = require_models(model_dir)
    with MediaPipeLandmarkTracker(model_dir, task_scheduling=task_scheduling) as tracker:
        provider_name = tracker.provider_name
    print(
        json.dumps(
            {
                "status": "ok",
                "models": {key: str(path) for key, path in verified.items()},
                "provider": provider_name,
                "camera_opened": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


def _rolling_fps(timestamps_ns: deque[int]) -> float:
    if len(timestamps_ns) < 2:
        return 0.0
    elapsed = (timestamps_ns[-1] - timestamps_ns[0]) / 1_000_000_000.0
    return (len(timestamps_ns) - 1) / elapsed if elapsed > 0 else 0.0


def run(args: argparse.Namespace) -> int:
    if args.self_check:
        return _self_check(args.model_dir, args.task_scheduling)
    if args.headless_frames < 0:
        raise ValueError("--headless-frames must be zero or greater")
    if args.width <= 0 or args.height <= 0 or not math.isfinite(args.fps) or args.fps <= 0:
        raise ValueError("Camera dimensions and FPS must be positive and finite")

    request = CameraRequest(args.camera_index, args.width, args.height, args.fps)
    provider_name = (
        f"{MediaPipeLandmarkTracker.provider_base_name}; {args.task_scheduling} tasks"
    )
    session = SessionRecord(
        request,
        mirror=args.mirror,
        inference_provider=provider_name,
        task_scheduling=args.task_scheduling,
    )
    capture = CaptureRuntime(LocalCamera(request), timeout=args.capture_timeout)
    frame_times: deque[int] = deque(maxlen=60)
    manifest_path: Path | None = None

    try:
        # Initialize the selected model before starting the continuous camera read.
        with MediaPipeLandmarkTracker(
            args.model_dir, task_scheduling=args.task_scheduling,
        ) as tracker, capture:
            observation = capture.observation
            if observation is None:
                raise RuntimeError("Camera opened without an observation record")
            session.attach_camera(observation)
            if not args.headless_frames:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW_NAME, 1360, 540)

            while True:
                frame = capture.read()
                queue_wait_ms = (time.monotonic_ns() - frame.identity.received_ns) / 1_000_000.0
                output = tracker.process(frame)
                capture.check()
                frame_times.append(time.monotonic_ns())
                metrics = RenderMetrics(
                    session_id=session.session_id,
                    camera_index=observation.index,
                    camera_backend=observation.backend,
                    frame_width=observation.width,
                    frame_height=observation.height,
                    nominal_fps=observation.nominal_fps,
                    processing_fps=_rolling_fps(frame_times),
                    inference_ms=output.inference_ms,
                    pose_ms=output.timings.pose_ms,
                    hands_ms=output.timings.hands_ms,
                    face_ms=output.timings.face_ms,
                    frame_sequence=frame.identity.sequence,
                    timestamp_ms=output.model_timestamp_ms,
                    provider=tracker.provider_name,
                )
                preview_started_ns = time.perf_counter_ns()
                preview = compose_preview(
                    frame.image_bgr, output.result, metrics, mirror=args.mirror,
                )
                snapshot = capture.snapshot()
                # Overlay uses host capture FPS, NOT claimed sensor or display FPS.
                capture_fps = (
                    f"{snapshot.mean_capture_fps:.1f}"
                    if snapshot.mean_capture_fps is not None else "unknown"
                )
                cv2.putText(
                    preview,
                    f"Latest-only | capture {capture_fps} FPS | replaced {snapshot.replaced}"
                    f" | queue {queue_wait_ms:.1f} ms",
                    (28, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (245, 245, 245), 1, cv2.LINE_AA,
                )
                preview_ms = (time.perf_counter_ns() - preview_started_ns) / 1_000_000.0
                receive_to_preview_ms = (
                    time.monotonic_ns() - frame.identity.received_ns
                ) / 1_000_000.0
                capture.check()
                session.observe(
                    output.result,
                    output.timings,
                    preview_composition_ms=preview_ms,
                    host_post_receive_total_ms=receive_to_preview_ms,
                    frame_timestamp_ns=frame.identity.received_ns,
                    capture_queue_ms=queue_wait_ms,
                    frame_identity=output.identity,
                )
                if args.headless_frames:
                    if session.frames >= args.headless_frames:
                        break
                    continue
                cv2.imshow(WINDOW_NAME, preview)
                key = cv2.waitKey(1) & 0xFF
                if key in {27, ord("q"), ord("Q")}:
                    break

        session.finish("completed")
    except KeyboardInterrupt:
        session.finish("interrupted", "KeyboardInterrupt")
        raise
    except Exception as exc:
        session.finish("failed", f"{type(exc).__name__}: {exc}")
        raise
    finally:
        session.attach_capture(capture.snapshot())
        try:
            cv2.destroyAllWindows()
        finally:
            primary_error = sys.exception()
            try:
                manifest_path = session.write(args.session_dir)
            except SessionRecordError as write_error:
                if primary_error is None:
                    raise
                primary_error.add_note(str(write_error))
                print(json.dumps({"status": "session_record_failed", "error": str(write_error)}),
                      file=sys.stderr)

    print(
        json.dumps(
            {
                "status": session.terminal_status,
                "frames": session.frames,
                "manifest": str(manifest_path) if manifest_path is not None else None,
            },
            ensure_ascii=False,
        )
    )
    return 0


def main() -> int:
    args = _parser().parse_args()
    try:
        return run(args)
    except PrototypeError as exc:
        print(
            json.dumps(
                {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2
    except ValueError as exc:
        print(
            json.dumps(
                {"status": "invalid", "error_type": type(exc).__name__, "error": str(exc)},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
