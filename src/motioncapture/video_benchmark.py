"""Reproducible fixed-video benchmark; all-frame throughput is NOT live 60-fps capture."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from array import array
from collections import Counter, defaultdict
from dataclasses import fields
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from motioncapture.inference_input import RgbConverter
from motioncapture.video_input import TimestampedVideo, VideoInfo, file_hash, probe_video


def distribution(samples, budget: float) -> dict:
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("Budget must be positive and finite")
    if not len(samples):
        return {"samples": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None,
                "p99_ms": None, "maximum_ms": None, "over_budget_fraction": None}
    values = np.asarray(samples, dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid duration samples")
    q50, q95, q99 = np.percentile(values, [50, 95, 99], method="linear")
    return {"samples": len(values), "mean_ms": float(values.mean()),
            "p50_ms": float(q50), "p95_ms": float(q95), "p99_ms": float(q99),
            "maximum_ms": float(values.max()),
            "over_budget_fraction": float((values > budget).mean()),
            "percentile_method": "numpy_linear_exact_sample_percentile",
            "budget_ms": budget}


def result_digest_update(digest, result) -> None:
    """An aggregate equality witness, not ground truth or saved biometric records."""
    for field in fields(result):
        items = getattr(result, field.name)
        digest.update(field.name.encode() + len(items).to_bytes(4, "little"))
        if field.name == "face_blendshapes":
            for item in items:
                digest.update(item.category_name.encode() + b"\0")
                digest.update(np.asarray([item.score], dtype="<f8").tobytes())
        else:
            values = [(p.x, p.y, p.z,
                       np.nan if p.visibility is None else p.visibility,
                       np.nan if p.presence is None else p.presence) for p in items]
            digest.update(np.asarray(values, dtype="<f8").tobytes())


def create_tracker(model_dir: Path, scheduling: str, rgb_mode: str):
    # Import lazily: inspect/preprocess are independent *explicit* stages.
    # Failure never turns a requested inference benchmark into decode-only.
    from motioncapture.landmarkers import MediaPipeLandmarkTracker

    if version("mediapipe") != "0.10.31":
        raise RuntimeError("Benchmark requires the repository-pinned MediaPipe 0.10.31")
    return MediaPipeLandmarkTracker(model_dir, task_scheduling=scheduling, rgb_mode=rgb_mode)


def inspect(path: Path, info: VideoInfo) -> dict:
    decode = array("d")
    duplicate_adjacent = 0
    previous_digest = None
    input_digest = hashlib.sha256()
    start = time.perf_counter_ns()
    with TimestampedVideo(path, info) as source:
        while True:
            t0 = time.perf_counter_ns()
            frame = source.read()
            t1 = time.perf_counter_ns()
            if frame is None:
                break
            decode.append((t1 - t0) / 1e6)
            if len(decode) % 600 == 0:
                print(f"inspected {len(decode)}/{len(info.pts)} frames",
                      file=sys.stderr, flush=True)
            pixels = memoryview(frame.image_bgr)
            current = hashlib.sha256(pixels).digest()
            duplicate_adjacent += int(current == previous_digest)
            previous_digest = current
            input_digest.update(current)
        assert source.complete
    return {"status": "completed", "decoded_frames": len(decode),
            "adjacent_exact_decoded_duplicates": duplicate_adjacent,
            "decoded_sequence_sha256": input_digest.hexdigest(),
            "decode_read": distribution(decode, 1000 / 60),
            "wall_seconds_including_hashing": (time.perf_counter_ns() - start) / 1e9,
            "inference_executed": False}


def preprocess(path: Path, info: VideoInfo, rounds: int, warmup: int, budget: float) -> dict:
    trials = []
    for trial in range(rounds):
        converters = {name: RgbConverter(name) for name in ("allocated", "reuse")}
        times = {name: array("d") for name in converters}
        verified = 0
        with TimestampedVideo(path, info) as source:
            while (frame := source.read()) is not None:
                order = ("allocated", "reuse")
                if (frame.identity.sequence + trial) % 2:
                    order = order[::-1]
                outputs = {}
                for name in order:
                    started = time.perf_counter_ns()
                    rgb = converters[name].convert(frame.image_bgr)
                    elapsed = (time.perf_counter_ns() - started) / 1e6
                    if frame.identity.sequence >= warmup:
                        times[name].append(elapsed)
                    outputs[name] = rgb
                if not np.array_equal(outputs["allocated"], outputs["reuse"]):
                    raise RuntimeError("RGB experiment is not pixel-identical")
                verified += 1
            assert source.complete
        trials.append({"trial": trial, "pixel_equal_frames": verified,
                       "stages": {n: distribution(v, budget) for n, v in times.items()}})
    return {"status": "completed", "trials": trials, "inference_executed": False,
            "scope": "paired BGR-to-RGB only; warmed shared input; excludes decoder/model"}


def track_pass(path: Path, info: VideoInfo, args, mode: str, trial: int) -> dict:
    started = time.perf_counter_ns()
    tracker = create_tracker(args.model_dir, args.task_scheduling, mode)
    metrics = defaultdict(lambda: array("d"))
    hand_groups = defaultdict(lambda: array("d"))
    transitions = defaultdict(lambda: array("d"))
    counts = Counter()
    previous_hands = None
    digest = hashlib.sha256()
    count = 0
    warmup_tracker_ms = 0.0
    telemetry_ms = 0.0
    try:
        tracker.open()
        initialized = time.perf_counter_ns()
        with TimestampedVideo(path, info) as source:
            loop_started = time.perf_counter_ns()
            while True:
                decode_start = time.perf_counter_ns()
                frame = source.read()
                decode_ms = (time.perf_counter_ns() - decode_start) / 1e6
                if frame is None:
                    break
                t0 = time.perf_counter_ns()
                output = tracker.process(frame)
                wall_ms = (time.perf_counter_ns() - t0) / 1e6
                if output.identity != frame.identity:
                    raise RuntimeError("Output identity does not match input frame")
                telemetry_start = time.perf_counter_ns()
                result = output.result
                hand_count = (int(bool(result.left_hand_landmarks))
                              + int(bool(result.right_hand_landmarks)))
                counts[f"hands_{hand_count}"] += 1
                counts["body"] += bool(result.pose_landmarks)
                counts["face"] += bool(result.face_landmarks)
                counts["left_hand"] += bool(result.left_hand_landmarks)
                counts["right_hand"] += bool(result.right_hand_landmarks)
                result_digest_update(digest, result)
                if count >= args.warmup_frames:
                    metrics["decode_read"].append(decode_ms)
                    metrics["process_call"].append(wall_ms)
                    for field in fields(output.timings):
                        metrics[field.name].append(getattr(output.timings, field.name))
                    hand_groups[str(hand_count)].append(output.timings.hands_ms)
                    if previous_hands is not None:
                        key = f"{previous_hands}->{hand_count}"
                        transitions[key].append(output.timings.hands_ms)
                else:
                    warmup_tracker_ms += wall_ms
                previous_hands = hand_count
                count += 1
                telemetry_ms += (time.perf_counter_ns() - telemetry_start) / 1e6
                if count % 600 == 0:
                    print(f"round {trial + 1}, rgb={mode}: {count}/{len(info.pts)} frames",
                          file=sys.stderr)
            assert source.complete
            loop_finished = time.perf_counter_ns()
    finally:
        primary_error = sys.exception()
        try:
            tracker.close()
        except Exception as cleanup_error:
            if primary_error is None:
                raise
            primary_error.add_note(f"Tracker cleanup also failed: {type(cleanup_error).__name__}")
    budget = 1000 / args.target_fps
    summaries = {key: distribution(values, budget) for key, values in metrics.items()}
    mean = summaries["process_call"]["mean_ms"]
    return {
        "status": "completed", "rgb_mode": mode, "trial": trial,
        "provider": tracker.provider_name, "frames_processed": count,
        "warmup_frames_processed_but_excluded": args.warmup_frames,
        "warmup_tracker_wall_ms": warmup_tracker_ms,
        "initialization_ms": (initialized - started) / 1e6,
        "loop_wall_seconds_including_telemetry": (loop_finished - loop_started) / 1e9,
        "telemetry_ms": telemetry_ms, "stages": summaries,
        "inverse_mean_process_call_fps": 1000 / mean,
        "detection_counts_all_frames": dict(counts),
        "hand_timing_by_populated_output_slots": {
            k: distribution(v, budget) for k, v in hand_groups.items()},
        "hand_timing_by_output_slot_transition": {
            k: distribution(v, budget) for k, v in transitions.items()},
        "prediction_sequence_sha256": digest.hexdigest(),
        "hand_groups_are_palm_detector_trace": False,
        "inference_executed": True,
    }


def _parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video", type=Path)
    p.add_argument("--stage", choices=("inspect", "preprocess", "track"), default="track")
    p.add_argument("--model-dir", type=Path, default=Path("models"))
    p.add_argument("--task-scheduling", choices=("parallel", "serial"), default="parallel")
    p.add_argument("--rgb-modes", nargs="+", choices=("allocated", "reuse"),
                   default=["allocated", "reuse"])
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--warmup-frames", type=int, default=60)
    p.add_argument("--target-fps", type=float, default=60)
    p.add_argument("--opencv-threads", type=int, default=None)
    p.add_argument("--output", type=Path, default=None)
    return p


def code_identity() -> dict:
    root = Path(__file__).resolve().parent
    names = ("video_benchmark.py", "video_input.py", "inference_input.py",
             "landmarkers.py", "contracts.py", "model_assets.py")
    result = {"sha256": {name: file_hash(root / name) for name in names},
              "revision": None, "working_tree_dirty": None}
    try:
        revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, timeout=3)
        status = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                                capture_output=True, text=True, timeout=3)
        if revision.returncode == 0:
            result["revision"] = revision.stdout.strip()
        if status.returncode == 0:
            result["working_tree_dirty"] = bool(status.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        pass  # Source digests remain authoritative; unknown git state is explicit null.
    return result


def run(args) -> tuple[dict, int]:
    report = {"schema_version": 1, "mode": "fixed_video_all_frames_unpaced",
              "requested_stage": args.stage, "status": "running", "trials": [],
              "inference_executed": False, "live_60fps_capture_verified": False,
              "sensor_to_photon_measured": False,
              "privacy": {"video_copied": False, "landmarks_saved": False,
                          "input_path_saved": False}}
    report["code"] = code_identity()
    old_threads = cv2.getNumThreads()
    try:
        if not 1 <= args.rounds <= 10 or args.warmup_frames < 0:
            raise ValueError("Invalid round or warmup count")
        if not math.isfinite(args.target_fps) or args.target_fps <= 0:
            raise ValueError("Target FPS must be positive and finite")
        if args.opencv_threads is not None:
            if not 1 <= args.opencv_threads <= 64:
                raise ValueError("Explicit OpenCV thread count must be in 1..64")
            cv2.setNumThreads(args.opencv_threads)
        report["environment"] = {
            "python": platform.python_version(), "platform": platform.platform(),
            "opencv": cv2.__version__, "numpy": np.__version__,
            "opencv_threads": cv2.getNumThreads(), "task_scheduling": args.task_scheduling,
        }
        try:
            report["environment"]["mediapipe"] = version("mediapipe")
        except PackageNotFoundError:
            report["environment"]["mediapipe"] = None
        info = probe_video(args.video)
        report["video"] = info.summary()
        report["target_compute_budget_ms"] = 1000 / args.target_fps
        report["warmup_frames"] = args.warmup_frames
        if args.warmup_frames >= len(info.pts) and args.stage != "inspect":
            raise ValueError("Warmup must be shorter than the clip")
        if args.stage == "inspect":
            report["result"] = inspect(args.video, info)
        elif args.stage == "preprocess":
            report["result"] = preprocess(args.video, info, args.rounds, args.warmup_frames,
                                           1000 / args.target_fps)
        else:
            from motioncapture.model_assets import MODEL_ASSETS

            report["models"] = [{"name": a.name, "sha256": a.sha256} for a in MODEL_ASSETS]
            for trial in range(args.rounds):
                order = args.rgb_modes if trial % 2 == 0 else args.rgb_modes[::-1]
                for mode in order:
                    report["inference_executed"] = True if report["trials"] else None
                    result = track_pass(args.video, info, args, mode, trial)
                    report["trials"].append(result)
                    report["inference_executed"] = True
            report["prediction_digests_all_equal"] = len({
                r["prediction_sequence_sha256"] for r in report["trials"]
            }) == 1
            report["digest_equality_is_accuracy_validation"] = False
        if file_hash(args.video) != info.sha256:
            raise RuntimeError("Input changed during the benchmark")
        report["status"] = "completed"
        return report, 0
    except Exception as exc:
        report["status"] = "failed"
        # Error strings from native dependencies may contain user paths. Keep
        # the exception class, not arbitrary dependency text, in shareable JSON.
        report["error_type"] = type(exc).__name__
        if (isinstance(exc, ModuleNotFoundError) and not report["trials"]
                and report.get("environment", {}).get("mediapipe") is None):
            report["inference_executed"] = False
        print(f"Benchmark failed ({type(exc).__name__}); requested stage not substituted",
              file=sys.stderr)
        return report, 2
    finally:
        cv2.setNumThreads(old_threads)


def main() -> int:
    args = _parser().parse_args()
    output = args.output or Path("sessions/video-bench") / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8] + ".json"
    )
    if output.exists() or output.resolve() == args.video.resolve():
        print("Refusing to overwrite an existing output or input", file=sys.stderr)
        return 2
    report, status = run(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "report": str(output)}))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
