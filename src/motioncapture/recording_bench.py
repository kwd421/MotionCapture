"""Unpaced, all-frame recording benchmark; never substitutes for live capture."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import struct
import subprocess
import tempfile
import time
from array import array
from contextlib import ExitStack
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

from motioncapture.recording import RecordedDecoder, RecordingProbe, inspect_recording


class Samples:
    """Exact per-pass summaries; bounded by the validated file's frame count."""

    def __init__(self) -> None:
        self.data = array("d")

    def add(self, value: float) -> None:
        if not math.isfinite(value) or value < 0:
            raise ValueError("Invalid benchmark duration")
        self.data.append(value)

    def summary(self, budget_ms: float) -> dict:
        a = np.frombuffer(self.data, dtype=np.float64)
        if not len(a):
            return {"samples": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None,
                    "p99_ms": None, "max_ms": None, "over_budget": None}
        q = np.quantile(a, [.5, .95, .99, 1])
        return {"samples": len(a), "mean_ms": float(a.mean()),
                **dict(zip(("p50_ms", "p95_ms", "p99_ms", "max_ms"), map(float, q))),
                "over_budget": int(np.count_nonzero(a > budget_ms))}


class Group:
    def __init__(self) -> None:
        self.frames = 0
        self.face = 0
        self.pose = 0
        self.hands = [0, 0]
        self.stages: dict[str, Samples] = {}

    def add(self, durations: dict[str, float], result: Any) -> None:
        self.frames += 1
        for name, value in durations.items():
            self.stages.setdefault(name, Samples()).add(value)
        if result is not None:
            self.face += bool(result.face_landmarks)
            self.pose += bool(result.pose_landmarks)
            self.hands[0] += bool(result.left_hand_landmarks)
            self.hands[1] += bool(result.right_hand_landmarks)

    def summary(self, budget: float, inference: bool) -> dict:
        return {"frames": self.frames,
                "detections": {"face": self.face, "pose": self.pose,
                               "left_hand": self.hands[0], "right_hand": self.hands[1]}
                if inference else None,
                "stages": {k: v.summary(budget) for k, v in self.stages.items()}}


def _make_tracker(model_dir: Path, scheduling: str):
    if version("mediapipe") != "0.10.31":
        raise ValueError("Recording benchmark requires pinned MediaPipe 0.10.31")
    from motioncapture.landmarkers import MediaPipeLandmarkTracker

    return MediaPipeLandmarkTracker(model_dir, task_scheduling=scheduling)


def _make_preview(mode: str):
    if mode == "none":
        return None
    if mode == "display":
        from motioncapture.preview import DisplayPreview

        return DisplayPreview().compose
    from motioncapture.render import compose_preview

    return compose_preview


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for part in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


_LANDMARK_FIELDS = (
    "pose_landmarks", "pose_world_landmarks", "left_hand_landmarks",
    "left_hand_world_landmarks", "right_hand_landmarks", "right_hand_world_landmarks",
    "face_landmarks",
)


def _digest_result(digest: Any, frame: Any, result: Any) -> None:
    """Fixed v1 framing; missing confidence and zero confidence hash differently."""
    identity = frame.identity
    digest.update(b"motioncapture-prediction-v1\0")
    digest.update(struct.pack("<Qqqq", identity.sequence, identity.pts,
                              identity.time_base.numerator, identity.time_base.denominator))
    for name in _LANDMARK_FIELDS:
        points = getattr(result, name)
        digest.update(struct.pack("<I", len(points)))
        for point in points:
            digest.update(struct.pack(
                "<5dBB", point.x, point.y, point.z,
                0.0 if point.visibility is None else point.visibility,
                0.0 if point.presence is None else point.presence,
                point.visibility is not None, point.presence is not None,
            ))
    digest.update(struct.pack("<I", len(result.face_blendshapes)))
    for shape in result.face_blendshapes:
        name = shape.category_name.encode("utf-8")
        digest.update(struct.pack("<I", len(name)))
        digest.update(name)
        digest.update(struct.pack("<dBq", shape.score, shape.index is not None,
                                  0 if shape.index is None else shape.index))


def _python_sources_digest(root: Path) -> str | None:
    """Identify local source bytes without logging content, paths or git diffs."""
    directory = root / "src" / "motioncapture"
    paths = sorted(directory.rglob("*.py"))
    if not paths:
        return None
    digest = hashlib.sha256()
    try:
        for path in paths:
            digest.update(path.relative_to(directory).as_posix().encode("utf-8") + b"\0")
            data = path.read_bytes()
            digest.update(struct.pack("<Q", len(data)))
            digest.update(data)
    except OSError:
        return None
    return digest.hexdigest()


def run_pass(args, probe: RecordingProbe) -> dict:
    inference = args.mode == "track"
    whole, steady = Group(), Group()
    windows: dict[str, Group] = {}
    workload: dict[str, Group] = {}
    pixel_digest = hashlib.sha256() if args.verify_pixels else None
    result_digest = hashlib.sha256() if args.verify_results and inference else None
    setup_started = time.perf_counter_ns()
    with ExitStack() as stack:
        tracker = stack.enter_context(_make_tracker(args.model_dir, args.task_scheduling)) \
            if inference else None
        compose = _make_preview(args.preview) if inference else None
        decoder = stack.enter_context(RecordedDecoder(args.input, probe,
                                                      threads=args.decode_threads))
        setup_ms = (time.perf_counter_ns() - setup_started) / 1_000_000
        loop_started = time.perf_counter_ns()
        for frame in decoder:
            start = time.perf_counter_ns()
            durations = {"decode_read_ms": frame.decode_ms}
            output = tracker.process_recorded(frame) if tracker is not None else None
            result = output.result if output is not None else None
            if output is not None:
                if output.identity != frame.identity:
                    raise ValueError("Recorded tracker output identity mismatch")
                durations.update(asdict(output.timings))
            if compose is not None:
                metrics = SimpleNamespace(
                    camera_backend="FFMPEG FILE", camera_index=0,
                    frame_width=probe.width, frame_height=probe.height,
                    processing_fps=0, inference_ms=output.timings.inference_wall_ms,
                    pose_ms=output.timings.pose_ms, hands_ms=output.timings.hands_ms,
                    face_ms=output.timings.face_ms,
                )
                before = time.perf_counter_ns()
                preview = compose(frame.image_bgr, result, metrics, mirror=False)
                durations["preview_ms"] = (time.perf_counter_ns() - before) / 1_000_000
                del preview
            durations["frame_service_ms"] = frame.decode_ms + (
                time.perf_counter_ns() - start) / 1_000_000
            if pixel_digest is not None:
                before = time.perf_counter_ns()
                pixel_digest.update(memoryview(frame.image_bgr).cast("B"))
                durations["pixel_verification_ms"] = (time.perf_counter_ns() - before) / 1e6
            if result_digest is not None:
                before = time.perf_counter_ns()
                _digest_result(result_digest, frame, result)
                durations["result_verification_ms"] = (time.perf_counter_ns() - before) / 1e6
            whole.add(durations, result)
            if frame.identity.sequence >= args.warmup_frames:
                steady.add(durations, result)
            relative_s = (frame.identity.pts - probe.pts[0]) * probe.time_base
            window = int(relative_s // 10) * 10
            windows.setdefault(f"{window}-{window + 10}s", Group()).add(durations, result)
            if result is not None:
                hands = (int(bool(result.left_hand_landmarks))
                         + int(bool(result.right_hand_landmarks)))
                key = f"resolved_hands={hands},face={int(bool(result.face_landmarks))}"
                workload.setdefault(key, Group()).add(durations, result)
        loop_s = (time.perf_counter_ns() - loop_started) / 1e9
        if not decoder.complete or whole.frames != len(probe.pts):
            raise ValueError("Incomplete recording benchmark")
        actual_threads = decoder.actual_threads
    # Hash after cleanup, outside the measured loop: detect input replacement.
    if _sha256(args.input) != probe.sha256:
        raise ValueError("Recording changed during the benchmark")
    budget = 1000 / args.target_fps
    return {
        "status": "completed", "decoder_cleanup_complete": True,
        "inference_executed": inference, "setup_ms": setup_ms,
        "task_scheduling": args.task_scheduling if inference else "not_run",
        "predictions_sha256": result_digest.hexdigest() if result_digest is not None else None,
        "result_hash_overhead_in_loop_fps": result_digest is not None,
        "loop_s": loop_s, "unpaced_loop_fps": whole.frames / loop_s,
        "decode_threads_reported": actual_threads,
        "pixels_sha256": pixel_digest.hexdigest() if pixel_digest else None,
        "pixel_hash_overhead_in_loop_fps": pixel_digest is not None,
        "all_frames": whole.summary(budget, inference),
        "steady_after_initial_frames": steady.summary(budget, inference),
        "windows_source_time": {k: g.summary(budget, inference) for k, g in windows.items()},
        "resolved_detection_workloads": {
            k: g.summary(budget, inference) for k, g in workload.items()
        } if inference else None,
    }


def _installed(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _source_revision() -> dict:
    root = Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=3, check=False)
        state = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=3, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return {"revision": None, "dirty": None,
                "python_sources_sha256": _python_sources_digest(root)}
    return {"revision": sha.stdout.strip() if sha.returncode == 0 else None,
            "dirty": bool(state.stdout.strip()) if state.returncode == 0 else None,
            "python_sources_sha256": _python_sources_digest(root)}


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".partial", delete=False) as f:
            temporary = Path(f.name)
            json.dump(report, f, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        # Atomic publication with no clobber, including a competing writer.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--mode", choices=("track", "decode"), default="track")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--decode-threads", type=int, default=0)
    parser.add_argument(
        "--task-scheduling", choices=("parallel", "serial", "staggered"), default="parallel",
    )
    parser.add_argument(
        "--compare-scheduling", action="store_true",
        help="four fresh passes: parallel, staggered, staggered, parallel; repeats must be 1",
    )
    parser.add_argument(
        "--verify-results", action="store_true",
        help="hash all numeric predictions and PTS; reports separate verification cost",
    )
    parser.add_argument("--preview", choices=("display", "native", "none"), default="display")
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--target-fps", type=float, default=60)
    parser.add_argument("--warmup-frames", type=int, default=60)
    parser.add_argument("--verify-pixels", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if (not 1 <= args.repeats <= 10 or args.warmup_frames < 0
            or not math.isfinite(args.target_fps) or args.target_fps <= 0
            or not 0 <= args.decode_threads <= 64):
        raise ValueError("Invalid benchmark limits")
    if args.compare_scheduling and (
        args.mode != "track" or args.repeats != 1 or args.task_scheduling != "parallel"
    ):
        raise ValueError("Comparison requires track mode, repeats=1 and default task scheduling")
    if args.verify_results and args.mode != "track":
        raise ValueError("Prediction verification requires track mode")
    plan = (["parallel", "staggered", "staggered", "parallel"]
            if args.compare_scheduling else [args.task_scheduling] * args.repeats)
    if args.output.exists():
        raise FileExistsError("Report already exists; select a new output name")
    report = {
        "schema_version": 1, "status": "running", "stage": "preflight",
        "mode": args.mode, "inference_requested": args.mode == "track",
        "runtime": {"platform": platform.platform(), "python": platform.python_version(),
                    "opencv": cv2.__version__, "numpy": np.__version__,
                    "mediapipe": _installed("mediapipe"), "cpu_count": os.cpu_count(),
                    "opencv_threads": cv2.getNumThreads(), "source": _source_revision()},
        "configuration": {"decode_threads_requested": args.decode_threads,
                          "task_scheduling": "per_run" if args.compare_scheduling
                          else args.task_scheduling,
                          "scheduling_plan": plan,
                          "verify_results": args.verify_results,
                          "preview": args.preview if args.mode == "track" else "not_run",
                          "target_fps": args.target_fps, "budget_ms": 1000 / args.target_fps,
                          "warmup_frames_included_in_all_frames": args.warmup_frames,
                          "repeats": len(plan), "resized": False, "paced": False},
        "source": None, "runs": [], "error": None,
        "privacy": {"raw_frames_written": False, "audio_processed": False,
                    "landmarks_written": False, "source_paths_written": False},
        "scope": "all-frame unpaced file service; not live camera/display latency or accuracy",
    }
    try:
        # Fail track mode immediately when the pinned runtime is missing, not as
        # a successful decode-only run after a lengthy video scan.
        if args.mode == "track" and _installed("mediapipe") != "0.10.31":
            raise RuntimeError("Pinned MediaPipe 0.10.31 is not installed")
        report["stage"] = "probe"
        started = time.perf_counter_ns()
        probe = inspect_recording(args.input, args.ffprobe)
        report["probe_ms"] = (time.perf_counter_ns() - started) / 1e6
        report["source"] = probe.summary()
        if args.mode == "track":
            from motioncapture.model_assets import MODEL_ASSETS

            report["models"] = [{"key": x.key, "sha256": x.sha256} for x in MODEL_ASSETS]
        report["stage"] = "passes"
        for scheduling in plan:
            pass_args = argparse.Namespace(**vars(args))
            pass_args.task_scheduling = scheduling
            report["runs"].append(run_pass(pass_args, probe))
        digests = [r["predictions_sha256"] for r in report["runs"]]
        report["prediction_equivalence"] = {
            "kind": "exact ordered numeric outputs and original PTS, not accuracy",
            "checked": args.verify_results,
            "all_passes_equal": (len(set(digests)) == 1
                                 if args.verify_results and all(digests) else None),
        }
        report["status"] = "completed"
        report["stage"] = "finished"
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        # No arbitrary exception message/path/FFmpeg metadata in persistent logs.
        report["error"] = {"type": type(exc).__name__, "stage": report["stage"]}
        if report["stage"] == "preflight":
            report["error"]["reason"] = "pinned_mediapipe_unavailable"
        write_report(args.output, report)
        print(json.dumps({"status": report["status"], "error": report["error"]}))
        return 130 if isinstance(exc, KeyboardInterrupt) else 2
    write_report(args.output, report)
    print(json.dumps({"status": "completed", "mode": args.mode,
                      "frames_each": len(probe.pts), "repeats": len(report["runs"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
