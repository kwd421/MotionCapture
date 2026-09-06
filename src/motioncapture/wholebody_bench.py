"""Screen alternate whole-body ONNX models with explicit assets and providers.

CLI examples live in README_WHOLEBODY_BENCH.md. No GPU/ANE speed is assumed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import struct
import time
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from motioncapture.recording import RecordedDecoder, inspect_recording
from motioncapture.recording_bench import Samples, _source_revision, run_pass, write_report
from motioncapture.wholebody_catalog import (
    ASSETS, CANDIDATES, BenchmarkError, catalog_digest, fetch_assets, sha256,
)
from motioncapture.wholebody_onnx import CAPABILITIES, PARTS, PROVIDERS, WholebodyEstimator


def installed(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


class ObservationStats:
    def __init__(self):
        self.frames = 0
        self.people = 0
        self.empty_frames = 0
        self.multi_person_frames = 0
        self.any_point_frames = dict.fromkeys(PARTS, 0)
        self.confident_points = dict.fromkeys(PARTS, 0)
        self.stages: dict[str, Samples] = {}

    def add(self, people, durations):
        if any(not math.isfinite(v) or v < 0 for v in durations.values()):
            raise BenchmarkError("invalid_stage_duration")
        self.frames += 1
        self.people += len(people)
        self.empty_frames += not people
        self.multi_person_frames += len(people) > 1
        for name, (start, end) in PARTS.items():
            count = sum(int(p.valid[start:end].sum()) for p in people)
            self.any_point_frames[name] += count > 0
            self.confident_points[name] += count
        for name, value in durations.items():
            self.stages.setdefault(name, Samples()).add(value)

    def summary(self, budget):
        return {"frames": self.frames, "person_observations": self.people,
                "empty_person_frames": self.empty_frames,
                "multi_person_frames": self.multi_person_frames,
                "frames_with_any_confident_point": self.any_point_frames.copy(),
                "confident_point_observations": self.confident_points.copy(),
                "counts_are_accuracy": False,
                "stages": {key: value.summary(budget) for key, value in self.stages.items()}}


class ReferenceBank:
    """Bounded RAM only, keyed by exact original sequence and PTS, not file IO time."""
    def __init__(self, count: int):
        self.xy = np.full((count, 133, 2), np.nan, np.float64)
        self.valid = np.zeros((count, 133), np.bool_)
        self.people = np.full(count, -1, np.int16)
        self.pts = np.zeros(count, np.int64)
        self.predictions_sha256 = None

    def store(self, frame, people):
        i = frame.identity.sequence
        self.people[i] = len(people)
        self.pts[i] = frame.identity.pts
        if len(people) == 1:
            self.xy[i] = people[0].xy
            self.valid[i] = people[0].valid


class Disagreement:
    def __init__(self):
        self.distances = {name: [] for name in PARTS}
        self.reference_only = dict.fromkeys(PARTS, 0)
        self.candidate_only = dict.fromkeys(PARTS, 0)
        self.frames = self.ambiguous = self.unavailable = 0
        self.worst: list[dict] = []

    def add(self, bank, frame, people):
        if bank is None:
            self.unavailable += 1
            return
        i = frame.identity.sequence
        if bank.people[i] < 0:
            self.unavailable += 1
            return
        if bank.pts[i] != frame.identity.pts:
            raise BenchmarkError("reference_pts_mismatch")
        self.frames += 1
        if bank.people[i] > 1 or len(people) > 1:
            self.ambiguous += 1
            return  # no invented actor association
        ref = bank.valid[i] if bank.people[i] else np.zeros(133, np.bool_)
        cand = people[0].valid if people else np.zeros(133, np.bool_)
        common = ref & cand
        distance = np.zeros(133, np.float64)
        if common.any():
            distance[common] = np.linalg.norm(bank.xy[i, common] - people[0].xy[common], axis=1)
            self.worst.append({"sequence": i, "pts": frame.identity.pts,
                               "max_pixel_disagreement": float(distance[common].max())})
            self.worst.sort(key=lambda x: x["max_pixel_disagreement"], reverse=True)
            del self.worst[16:]
        for name, (a, b) in PARTS.items():
            self.reference_only[name] += int((ref[a:b] & ~cand[a:b]).sum())
            self.candidate_only[name] += int((cand[a:b] & ~ref[a:b]).sum())
            if common[a:b].any():
                self.distances[name].append(distance[a:b][common[a:b]].copy())

    def summary(self):
        result = {}
        for name, chunks in self.distances.items():
            if chunks:
                values = np.concatenate(chunks)
                result[name] = {"matched_points": len(values), "mean_pixels": float(values.mean()),
                                **dict(zip(("p50", "p95", "p99", "max"),
                                           map(float, np.quantile(values, [.5, .95, .99, 1])),
                                           strict=True))}
            else:
                result[name] = {"matched_points": 0, "mean_pixels": None,
                                "p50": None, "p95": None, "p99": None, "max": None}
        return {"reference": "same model's first completed CPU pass, NOT ground truth",
                "checked_frames": self.frames, "reference_unavailable_frames": self.unavailable,
                "reference_storage_dtype": "float64",
                "ambiguous_multi_person_frames": self.ambiguous, "parts": result,
                "reference_only_point_observations": self.reference_only,
                "candidate_only_point_observations": self.candidate_only,
                "worst_frames": self.worst, "accuracy_verified": False}


def update_digest(digest, frame, people):
    identity = frame.identity
    digest.update(b"mocap-wholebody133-v1\0")
    digest.update(struct.pack("<QqqqI", identity.sequence, identity.pts,
                              identity.time_base.numerator, identity.time_base.denominator,
                              len(people)))
    for person in people:
        digest.update(np.asarray(person.xy, dtype="<f8").tobytes())
        digest.update(np.asarray(person.scores, dtype="<f8").tobytes())
        digest.update(person.valid.tobytes())


def _location(frame):
    return None if frame is None else {"sequence": frame.identity.sequence,
                                      "pts": frame.identity.pts,
                                      "time_base": str(frame.identity.time_base)}


def candidate_pass(args, probe, model, provider, bank=None,
                   *, estimator_factory=WholebodyEstimator):
    target = min(args.max_frames or len(probe.pts), len(probe.pts))
    whole, steady, windows = ObservationStats(), ObservationStats(), {}
    comparisons = Disagreement()
    # Allocate only when enrolling this model's first CPU reference.
    new_bank = ReferenceBank(target) if provider == "cpu" and bank is None else None
    estimator = None
    decoder = None
    frame = last = None
    phase = "assets_and_session_preflight"
    started = time.perf_counter_ns()
    loop_start = None
    loop_end = None
    report = {"kind": "wholebody_onnx", "model": model, "provider": provider,
              "capabilities": CAPABILITIES, "status": "running", "cleanup_errors": [],
              "requested_frames": target, "scope": "full_file" if target == len(probe.pts)
              else "explicit_prefix", "backend": None, "error": None}
    digest = hashlib.sha256()
    try:
        estimator = estimator_factory(args.asset_dir, model, provider,
                                      allow_cpu=args.allow_cpu_partitions, threads=args.ort_threads,
                                      keypoint_threshold=args.keypoint_threshold)
        report["backend"] = estimator.metadata
        decoder = RecordedDecoder(args.input, probe, threads=args.decode_threads)
        phase = "decoder_open"
        decoder.__enter__()
        report["setup_and_preflight_ms"] = (time.perf_counter_ns() - started) / 1e6
        loop_start = time.perf_counter_ns()
        phase = "decode"
        for frame in decoder:
            phase = "wholebody_inference"
            people, durations = estimator.process(frame.image_bgr)
            durations["decode_read_ms"] = frame.decode_ms
            durations["frame_service_ms"] = frame.decode_ms + durations["wholebody_service_ms"]
            observe_start = time.perf_counter_ns()
            phase = "comparison"
            if new_bank is not None:
                new_bank.store(frame, people)
            else:
                comparisons.add(bank, frame, people)
            update_digest(digest, frame, people)
            durations["verification_and_comparison_ms"] = (
                time.perf_counter_ns() - observe_start) / 1e6
            whole.add(people, durations)
            if frame.identity.sequence >= args.warmup_frames:
                steady.add(people, durations)
            seconds = float((frame.identity.pts - probe.start_pts) * probe.time_base)
            bucket = math.floor(seconds / 10) * 10
            group = windows.setdefault(f"{bucket}-{bucket + 10}s", ObservationStats())
            group.add(people, durations)
            last = frame
            if whole.frames % 300 == 0:
                print(json.dumps({"model": model, "provider": provider,
                                  "completed_frames": whole.frames}), flush=True)
            if whole.frames == target and target < len(probe.pts):
                break
            phase = "decode"
            frame = None  # A failed next decode has no validated current identity.
        loop_end = time.perf_counter_ns()
        if whole.frames != target or (target == len(probe.pts) and not decoder.complete):
            raise BenchmarkError("frame_coverage_mismatch")
        phase = "source_integrity"
        if sha256(args.input) != probe.sha256:
            raise BenchmarkError("source_changed_during_pass")
        report["status"] = "completed"
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["error"] = {"type": type(exc).__name__, "code": getattr(exc, "code", None),
                           "phase": phase, "current_frame": _location(frame),
                           "last_completed_frame": _location(last)}
        new_bank = None  # an incomplete pass must not become a reference
        loop_end = time.perf_counter_ns()
    finally:
        for name, owner in (("decoder", decoder), ("estimator", estimator)):
            report[f"{name}_cleanup"] = "not_created"
            if owner is not None:
                try:
                    owner.__exit__(None, None, None) if name == "decoder" else owner.close()
                    report[f"{name}_cleanup"] = "owner_released"
                except BaseException as exc:
                    report[f"{name}_cleanup"] = "failed"
                    report["cleanup_errors"].append({"owner": name, "type": type(exc).__name__})
                    if report["status"] == "completed":
                        report["status"] = "failed"
                    new_bank = None
    budget = 1000 / args.target_fps
    elapsed = (loop_end - loop_start) / 1e9 if loop_start is not None else None
    report.update({
        "all_frames": whole.summary(budget), "steady_after_initial_frames": steady.summary(budget),
        "windows_source_time": {k: v.summary(budget) for k, v in windows.items()},
        "loop_s": elapsed, "unpaced_loop_fps": whole.frames / elapsed
        if elapsed and whole.frames and report["status"] == "completed" else None,
        "timed_loop_includes_comparison": True, "preview": "none",
        "predictions_sha256": digest.hexdigest() if whole.frames else None,
        "hash_scope": "complete_pass" if report["status"] == "completed" else "partial_prefix",
        "provider_disagreement": {"kind": "CPU reference enrollment"} if new_bank is not None
        else comparisons.summary(),
        "live_60fps_verified": False, "accuracy_verified": False,
    })
    if new_bank is not None:
        new_bank.predictions_sha256 = report["predictions_sha256"]
    report["cpu_reference_hash_equal"] = (
        report["predictions_sha256"] == bank.predictions_sha256
        if bank is not None and report["status"] == "completed" else None
    )
    return report, new_bank


def native_control(args, probe):
    native_args = SimpleNamespace(
        mode="track", verify_pixels=False, verify_results=True, model_dir=args.native_model_dir,
        task_scheduling="parallel", input=args.input, preview="none", target_fps=args.target_fps,
        warmup_frames=args.warmup_frames, decode_threads=args.decode_threads,
    )
    partial = {}
    try:
        result = run_pass(native_args, probe, frame_limit=args.max_frames, failure_record=partial)
    except BaseException as exc:
        result = partial or {"status": "failed", "error": {"type": type(exc).__name__}}
    result.update({"kind": "native_mediapipe_control", "model": "mediapipe-native",
                   "provider": "cpu", "preview": "none",
                   "capabilities": {"body_points": 33, "hand_points_each": 21,
                                    "face_points": 478, "face_blendshapes": True,
                                    "monocular_world_landmarks": True, "metric_studio_3d": False},
                   "not_feature_equivalent_to_133point_candidates": True})
    return result


def make_plan(models, providers, abba: bool, native: bool):
    if not models or len(set(models)) != len(models) or any(m not in CANDIDATES for m in models):
        raise BenchmarkError("invalid_model_selection")
    if (not providers or len(set(providers)) != len(providers)
            or any(p not in PROVIDERS for p in providers)):
        raise BenchmarkError("invalid_provider_selection")
    if abba and (len(providers) != 2 or providers[0] != "cpu" or providers[1] == "cpu"):
        raise BenchmarkError("abba_requires_cpu_then_one_candidate_provider")
    if "cpu" in providers and providers[0] != "cpu":
        raise BenchmarkError("cpu_reference_must_be_first")
    plan = []
    if native:
        plan.append(("mediapipe-native", "cpu"))
    for model in models:
        for provider in ([*providers, *reversed(providers)] if abba else providers):
            plan.append((model, provider))
    if native:
        plan.append(("mediapipe-native", "cpu"))
    return plan


def execute(args):
    if args.output.exists():
        raise BenchmarkError("report_already_exists")
    if not args.research_only:
        raise BenchmarkError("explicit_research_terms_acknowledgement_required")
    if (args.max_frames < 0 or args.warmup_frames < 0 or not 1 <= args.ort_threads <= 64
            or not 0 <= args.decode_threads <= 64 or not math.isfinite(args.target_fps)
            or args.target_fps <= 0 or not math.isfinite(args.keypoint_threshold)
            or args.keypoint_threshold < 0):
        raise BenchmarkError("invalid_benchmark_limits")
    plan = make_plan(args.models, args.providers, args.abba, args.native_control)
    for i in range(1, len(plan) + 1):
        if args.output.with_name(args.output.stem + f".arm-{i:02d}.json").exists():
            raise BenchmarkError("checkpoint_already_exists")
    report = {
        "schema_version": 1, "experiment": "wholebody_onnx_screening", "status": "running",
        "source": None, "runs": [], "error": None,
        "runtime": {"platform": platform.platform(), "python": platform.python_version(),
                    "onnxruntime": installed("onnxruntime"), "onnx": installed("onnx"),
                    "opencv": cv2.__version__, "numpy": np.__version__,
                    "mediapipe": installed("mediapipe"), "source": _source_revision()},
        "configuration": {"plan": plan, "target_fps": args.target_fps,
                          "requested_prefix_frames": args.max_frames,
                          "ort_threads": args.ort_threads, "decode_threads": args.decode_threads,
                          "allow_cpu_partitions": args.allow_cpu_partitions,
                          "keypoint_threshold": args.keypoint_threshold,
                          "warmup_frames_included_in_all_frames": args.warmup_frames,
                          "preview": "none", "paced": False, "detector_cadence": "every_frame"},
        "privacy": {"frames_written": False, "landmarks_written": False,
                    "audio_processed": False, "reference_observations": "bounded_RAM_only"},
        "commercial_release_cleared": False, "catalog_sha256": catalog_digest(),
        "scope": ("all-frame 2D file service, NOT live 3D mocap "
                  "or feature-equivalent facial capture"),
    }
    banks = {}
    previous_model = None
    try:
        probe = inspect_recording(args.input, args.ffprobe)
        report["source"] = probe.summary()
        for model, provider in plan:
            if model != previous_model:
                banks.clear()  # Only one model's reference remains live in RAM.
                previous_model = model
            print(json.dumps({"status": "starting_arm", "model": model,
                              "provider": provider}), flush=True)
            if model == "mediapipe-native":
                arm = native_control(args, probe)
            else:
                arm, bank = candidate_pass(args, probe, model, provider, banks.get(model))
                if bank is not None:
                    banks[model] = bank
            report["runs"].append(arm)
            # Publish a unique per-arm checkpoint. A later process abort cannot erase it.
            checkpoint = args.output.with_name(args.output.stem +
                                                f".arm-{len(report['runs']):02d}.json")
            write_report(checkpoint, {**report, "status": "checkpoint_not_final"})
            if arm["status"] == "interrupted":
                break
        complete = len(report["runs"]) == len(plan) and all(
            r["status"] == "completed" for r in report["runs"])
        report["status"] = "completed" if complete else "partial_failed"
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["error"] = {"type": type(exc).__name__, "code": getattr(exc, "code", None)}
    report["performance_verdict"] = "requires_device_and_quality_review"
    report["accuracy_verified"] = False
    write_report(args.output, report)
    print(json.dumps({"status": report["status"], "arms_recorded": len(report["runs"])}))
    return 0 if report["status"] == "completed" else 2


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    commands.add_parser("catalog")
    fetch = commands.add_parser("fetch")
    fetch.add_argument("--models", nargs="+", choices=CANDIDATES, default=list(CANDIDATES))
    fetch.add_argument("--asset-dir", type=Path, default=Path("models/wholebody"))
    fetch.add_argument("--research-only", action="store_true")
    run = commands.add_parser("run")
    run.add_argument("input", type=Path)
    run.add_argument("--models", nargs="+", choices=CANDIDATES, default=list(CANDIDATES))
    run.add_argument("--providers", nargs="+", choices=PROVIDERS, default=["cpu"])
    run.add_argument("--abba", action="store_true")
    run.add_argument("--native-control", action="store_true")
    run.add_argument("--asset-dir", type=Path, default=Path("models/wholebody"))
    run.add_argument("--native-model-dir", type=Path, default=Path("models"))
    run.add_argument("--allow-cpu-partitions", action="store_true")
    run.add_argument("--research-only", action="store_true")
    run.add_argument("--max-frames", type=int, default=0)
    run.add_argument("--warmup-frames", type=int, default=60)
    run.add_argument("--decode-threads", type=int, default=0)
    run.add_argument("--ort-threads", type=int, default=4)
    run.add_argument("--keypoint-threshold", type=float, default=.3)
    run.add_argument("--target-fps", type=float, default=60)
    run.add_argument("--ffprobe", default="ffprobe")
    run.add_argument("--output", type=Path, required=True)
    return cli


def main():
    args = parser().parse_args()
    try:
        if args.command == "catalog":
            print(json.dumps({"assets": {k: asdict(v) for k, v in ASSETS.items()},
                              "providers": PROVIDERS,
                              "commercial_release_cleared": False}, indent=2))
            return 0
        if args.command == "fetch":
            receipts = fetch_assets(args.asset_dir, args.models, research_only=args.research_only)
            print(json.dumps({"status": "enrolled", "assets": receipts}, indent=2))
            return 0
        return execute(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "type": type(exc).__name__,
                          "code": getattr(exc, "code", None)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
