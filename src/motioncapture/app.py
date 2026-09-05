"""Live MacBook camera body, hand/finger, and facial landmark prototype."""

from __future__ import annotations

import argparse
import json
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
from motioncapture.session import SessionRecord

WINDOW_NAME = "MotionCapture - Live 2D Body + Face Prototype"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
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
        help="process exactly this many real camera frames without opening a window",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="verify the pinned model and initialize the landmarker without opening a camera",
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

    request = CameraRequest(
        index=args.camera_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    provider_name = (
        f"{MediaPipeLandmarkTracker.provider_base_name}; {args.task_scheduling} tasks"
    )
    session = SessionRecord(
        request,
        mirror=args.mirror,
        inference_provider=provider_name,
        task_scheduling=args.task_scheduling,
    )
    require_models(args.model_dir)
    frame_times: deque[int] = deque(maxlen=60)
    start_timestamp_ns: int | None = None
    last_timestamp_ms = -1
    manifest_path: Path | None = None

    try:
        with LocalCamera(request) as camera, MediaPipeLandmarkTracker(
            args.model_dir,
            task_scheduling=args.task_scheduling,
        ) as tracker:
            if camera.observation is None:
                raise RuntimeError("Camera opened without an observation record")
            session.attach_camera(camera.observation)

            if not args.headless_frames:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW_NAME, 1360, 540)

            while True:
                frame = camera.read()
                host_processing_started_ns = time.perf_counter_ns()
                if start_timestamp_ns is None:
                    start_timestamp_ns = frame.timestamp_ns
                timestamp_ms = max(
                    last_timestamp_ms + 1,
                    (frame.timestamp_ns - start_timestamp_ns) // 1_000_000,
                )
                last_timestamp_ms = timestamp_ms

                output = tracker.process(frame.image_bgr, timestamp_ms)
                frame_times.append(time.perf_counter_ns())

                observation = camera.observation
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
                    frame_sequence=frame.sequence,
                    timestamp_ms=timestamp_ms,
                    provider=tracker.provider_name,
                )
                preview_composition_started_ns = time.perf_counter_ns()
                preview = compose_preview(
                    frame.image_bgr,
                    output.result,
                    metrics,
                    mirror=args.mirror,
                )
                preview_composition_ms = (
                    time.perf_counter_ns() - preview_composition_started_ns
                ) / 1_000_000.0
                host_post_receive_total_ms = (
                    time.perf_counter_ns() - host_processing_started_ns
                ) / 1_000_000.0
                session.observe(
                    output.result,
                    output.timings,
                    preview_composition_ms=preview_composition_ms,
                    host_post_receive_total_ms=host_post_receive_total_ms,
                    frame_timestamp_ns=frame.timestamp_ns,
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
        cv2.destroyAllWindows()
        try:
            manifest_path = session.write(args.session_dir)
        except SessionRecordError:
            if sys.exc_info()[0] is None:
                raise

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
