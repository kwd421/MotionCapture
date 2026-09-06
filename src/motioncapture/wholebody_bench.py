"""All-frame WholeBody133 candidate matrix; not feature-equivalent MediaPipe FPS."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import struct
import sys
import time
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from motioncapture.recording import RecordedDecoder, inspect_recording
from motioncapture.recording_bench import Samples, _source_revision, run_pass, write_report
from motioncapture.wholebody_assets import (
    ASSETS, DEFAULT_DIR, POSE_MODELS, digest_file, require_asset,
)
from motioncapture.wholebody_onnx import (
    CAPABILITIES, MODES, ORT_VERSION, PARTS, UnsupportedProvider, WholebodyONNX,
)


class SourceChanged(RuntimeError):
    pass


def failure_code(exc):
    """Stable hints without sharing native exception text or private paths."""
    if isinstance(exc, UnsupportedProvider):
        return "requested_provider_or_placement_unavailable"
    if isinstance(exc, SourceChanged):
        return "source_integrity_changed"
    text = str(exc).lower()
    for fragment, code in (
        ("checksum", "asset_checksum_failed"),
        ("coremlexecutionprovider", "coreml_session_failed"),
        ("cpu ep fallback", "cpu_graph_fallback_disallowed"),
        ("cpu fallback", "cpu_graph_fallback_disallowed"),
        ("not implemented", "unsupported_model_operator"),
        ("not supported", "unsupported_model_or_provider"),
        ("schema", "export_schema_mismatch"),
        ("input shape", "input_shape_mismatch"),
        ("nonfinite", "nonfinite_model_output"),
        ("onnxruntime", "onnxruntime_dependency_or_version"),
    ):
        if fragment in text:
            return code
    return "candidate_failed_inspect_phase_and_type"


def location(identity):
    return None if identity is None else {
        "sequence": identity.sequence, "pts": identity.pts, "time_base": str(identity.time_base),
    }


def numerical_summary(values):
    if not values:
        return {"samples": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    a = np.asarray(values, dtype=np.float64)
    return {"samples": len(a), "mean": float(a.mean()),
            **dict(zip(("p50", "p95", "p99", "max"),
                       map(float, np.quantile(a, [.5, .95, .99, 1]))))}


class CPUReference:
    """RAM only; at most one single-person prediction per selected source frame.

    Zero/multiple-person observations retain counts, not an arbitrary actor.
    The caller may cache one such bank per explicitly selected candidate model.
    """

    def __init__(self, frames: int):
        self.counts = np.full(frames, -1, dtype=np.int16)
        self.values = np.empty((frames, 133, 3), dtype=np.float32)
        self.pts = np.zeros(frames, dtype=np.int64)
        self.complete = False

    def observe(self, identity, xy, scores):
        i = identity.sequence
        if not 0 <= i < len(self.counts) or self.counts[i] != -1:
            raise ValueError("Reference sequence out of range or duplicate")
        self.counts[i], self.pts[i] = len(xy), identity.pts
        if len(xy) == 1:
            self.values[i, :, :2], self.values[i, :, 2] = xy[0], scores[0]


class Comparison:
    def __init__(self, reference: CPUReference | None, threshold: float):
        self.reference, self.threshold = reference, threshold
        self.frames = self.single = self.unmatched = 0
        self.parts = {name: {"jointly_confident_points": 0, "reference_only_points": 0,
                             "candidate_only_points": 0, "frame_mean": [], "frame_max": []}
                      for name in PARTS}

    def observe(self, identity, xy, scores):
        if self.reference is None or not self.reference.complete:
            return
        ref, i = self.reference, identity.sequence
        if not 0 <= i < len(ref.counts) or ref.pts[i] != identity.pts:
            raise ValueError("CPU reference PTS/sequence mismatch")
        self.frames += 1
        if ref.counts[i] != 1 or len(xy) != 1:
            self.unmatched += 1
            return
        self.single += 1
        for name, (start, end) in PARTS.items():
            a = ref.values[i, start:end]
            b, score = xy[0, start:end], scores[0, start:end]
            valid_a, valid_b = a[:, 2] > self.threshold, score > self.threshold
            both = valid_a & valid_b
            stats = self.parts[name]
            stats["jointly_confident_points"] += int(both.sum())
            stats["reference_only_points"] += int((valid_a & ~valid_b).sum())
            stats["candidate_only_points"] += int((valid_b & ~valid_a).sum())
            if both.any():
                distances = np.linalg.norm(a[both, :2] - b[both], axis=1)
                stats["frame_mean"].append(float(distances.mean()))
                stats["frame_max"].append(float(distances.max()))

    def summary(self):
        if self.reference is None or not self.reference.complete:
            return {"status": "unavailable", "reason": "no_completed_same_model_cpu_reference"}
        return {"status": "observed", "frames_compared": self.frames,
                "single_person_pairs": self.single,
                "unmatched_zero_or_multiple_person_frames": self.unmatched,
                "metric": "source-pixel displacement; per-frame distributions, NOT all-point AP",
                "correspondence": "single detected person in each frame; identity not verified",
                "accuracy_verified": False,
                "parts": {name: {k: v for k, v in values.items() if not k.startswith("frame_")}
                          | {"frame_mean_px": numerical_summary(values["frame_mean"]),
                             "frame_max_px": numerical_summary(values["frame_max"])}
                          for name, values in self.parts.items()}}


class FrameStatistics:
    def __init__(self):
        self.frames = 0
        self.person_counts = Counter()
        self.confident = Counter()
        self.stages = {}

    def observe(self, people, scores, threshold, durations):
        self.frames += 1
        self.person_counts[str(people)] += 1
        for name, (start, end) in PARTS.items():
            self.confident[name] += int((scores[:, start:end] > threshold).sum())
        for name, value in durations.items():
            self.stages.setdefault(name, Samples()).add(value)

    def summary(self, budget):
        instances = sum(int(n) * frames for n, frames in self.person_counts.items())
        return {"frames": self.frames, "person_instances": instances,
                "frames_by_person_count": dict(self.person_counts),
                "confidence_coverage_not_accuracy": {
                    name: {"above_threshold_points": self.confident[name],
                           "possible_points_in_detected_persons": instances * (end-start)}
                    for name, (start, end) in PARTS.items()},
                "stages": {name: values.summary(budget) for name, values in self.stages.items()}}


def execute_candidate(args, probe, model, mode, reference=None, *, engine_factory=WholebodyONNX,
                      asset_loader=require_asset):
    """Return even on caught failure, preserving partial work and primary error."""
    count = min(args.max_frames, len(probe.pts)) if args.max_frames else len(probe.pts)
    result = {"candidate": model, "pose_mode": mode, "status": "failed",
              "capabilities": CAPABILITIES, "requested_frames": count,
              "scope": "full_file" if count == len(probe.pts) else "explicit_prefix",
              "detector_policy": "fixed CPU; detect every frame; process ALL person boxes",
              "comparison_scope": "same-model CPU numeric disagreement, not ground truth",
              "frame_service_scope": "decode + detector + all person crops/pose + decode outputs",
              "loop_scope": "unpaced; includes hashing/comparison/statistics; no GUI",
              "error": None, "raw_point_validity": "score > configured threshold",
              "model_inference_executed": False}
    whole, steady, windows, workloads = FrameStatistics(), FrameStatistics(), {}, {}
    comparison = Comparison(reference, args.score_threshold)
    bank = CPUReference(count) if mode == "cpu" and reference is None else None
    digest = hashlib.sha256()
    phase, current, last = "asset_verification", None, None
    setup_start = time.perf_counter_ns()
    setup_ms = loop_start = loop_s = None
    engine = decoder = None
    cleanup = {"engine": "not_created", "decoder": "not_created"}
    primary = None
    try:
        if digest_file(args.input) != probe.sha256:
            raise SourceChanged("Input changed since initial inspection")
        detector_path, detector_meta = asset_loader(ASSETS["yolox-tiny"], args.model_dir)
        pose_path, pose_meta = asset_loader(ASSETS[model], args.model_dir)
        result["models"] = {"detector": detector_meta, "pose": pose_meta}
        phase = "session_setup"
        cleanup["engine"] = "unknown_after_open_failure"
        engine = engine_factory(detector_path, pose_path, mode, ASSETS[model].input_hw,
                                args.ort_threads)
        cleanup["engine"] = "open"
        phase = "placement_preflight"
        with RecordedDecoder(args.input, probe, threads=args.decode_threads) as probe_decoder:
            first = next(iter(probe_decoder))
            result["execution"] = engine.preflight(first.image_bgr)
            result["placement_probe_executed"] = True
            del first
        phase = "decode_setup"
        decoder = RecordedDecoder(args.input, probe, threads=args.decode_threads)
        decoder.__enter__()
        cleanup["decoder"] = "open"
        result["decode_threads_reported"] = decoder.actual_threads
        setup_ms = (time.perf_counter_ns() - setup_start) / 1e6
        loop_start = time.perf_counter_ns()
        iterator = iter(decoder)
        for _ in range(count):
            phase = "decode"
            frame = next(iterator)
            current = frame.identity
            phase = "inference"
            started = time.perf_counter_ns()
            boxes, xy, scores, durations = engine.process(frame.image_bgr)
            result["model_inference_executed"] = True
            durations = {**durations, "adapter_ms": (time.perf_counter_ns()-started)/1e6,
                         "decode_read_ms": frame.decode_ms}
            durations["frame_service_ms"] = frame.decode_ms + durations["adapter_ms"]
            if (xy.shape != (len(boxes), 133, 2) or scores.shape != (len(boxes), 133)
                    or not np.isfinite(xy).all() or not np.isfinite(scores).all()):
                raise ValueError("Invalid WholeBody result contract")
            phase = "verification"
            before = time.perf_counter_ns()
            digest.update(struct.pack("<qqI", current.sequence, current.pts, len(boxes)))
            digest.update(np.asarray(boxes, dtype="<f4").tobytes())
            digest.update(np.asarray(xy, dtype="<f4").tobytes())
            digest.update(np.asarray(scores, dtype="<f4").tobytes())
            if bank is not None:
                bank.observe(current, xy, scores)
            comparison.observe(current, xy, scores)
            durations["verification_ms"] = (time.perf_counter_ns()-before)/1e6
            whole.observe(len(boxes), scores, args.score_threshold, durations)
            if current.sequence >= args.warmup_frames:
                steady.observe(len(boxes), scores, args.score_threshold, durations)
            second = int((current.pts-probe.pts[0]) * probe.time_base) // 10 * 10
            windows.setdefault(f"{second}-{second+10}s", FrameStatistics()).observe(
                len(boxes), scores, args.score_threshold, durations)
            workloads.setdefault(str(len(boxes)), FrameStatistics()).observe(
                len(boxes), scores, args.score_threshold, durations)
            last = current
            if whole.frames % 300 == 0:
                print(f"{model}/{mode}: {whole.frames}/{count}", flush=True)
        phase = "eof_verification"
        if count == len(probe.pts):
            try:
                next(iterator)
            except StopIteration:
                pass
            else:
                raise ValueError("Unexpected extra source frame")
            if not decoder.complete:
                raise ValueError("Decoder did not acknowledge EOF")
        loop_s = (time.perf_counter_ns()-loop_start)/1e9
        phase = "source_integrity"
        if digest_file(args.input) != probe.sha256:
            raise SourceChanged("Input changed during candidate")
        result["status"] = "completed"
    except BaseException as exc:
        primary = exc
        result["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        # Avoid putting native exceptions containing local paths into shared JSON.
        result["error"] = {"type": type(exc).__name__, "phase": phase, "code": failure_code(exc),
                           "message_sha256": hashlib.sha256(str(exc).encode()).hexdigest()}
        if isinstance(exc, SourceChanged):
            result["fatal_source_change"] = True
        if loop_start is not None:
            loop_s = (time.perf_counter_ns()-loop_start)/1e9
    finally:
        for name, owner in (("decoder", decoder), ("engine", engine)):
            if owner is None:
                continue
            try:
                owner.__exit__(None, None, None) if name == "decoder" else owner.close()
            except BaseException as exc:
                cleanup[name] = "failed"
                if primary is None:
                    primary = exc
                    result["error"] = {"type": type(exc).__name__, "phase": f"{name}_cleanup"}
                    result["status"] = "failed"
                else:
                    result.setdefault("cleanup_errors", []).append(type(exc).__name__)
            else:
                cleanup[name] = "complete"
    ok = result["status"] == "completed"
    if bank is not None:
        bank.complete = ok and bool(np.all(bank.counts >= 0))
    budget = 1000 / args.target_fps
    result.update(setup_and_validation_ms=setup_ms, cleanup=cleanup,
                  completed_frames=whole.frames, last_completed=location(last),
                  current_frame=location(current), loop_s=loop_s,
                  unpaced_loop_fps=whole.frames/loop_s if ok and loop_s else None,
                  prediction_hash_scope="whole_pass" if ok else "completed_prefix",
                  predictions_sha256=digest.hexdigest() if whole.frames else None,
                  all_frames=whole.summary(budget),
                  steady_after_initial_frames=steady.summary(budget),
                  windows_source_time={k: v.summary(budget) for k, v in windows.items()},
                  frames_by_workload={k: v.summary(budget) for k, v in workloads.items()},
                  cpu_comparison=comparison.summary())
    return result, bank if bank is not None and bank.complete else None


def run_matrix(args, *, inspect=inspect_recording, candidate=execute_candidate):
    # Refuse an existing output directory. Intent records survive native aborts.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows, references = [], {}
    state = {"schema_version": 1, "status": "initializing", "runs": rows,
             "privacy": {"raw_frames_written": False, "coordinates_written": False,
                         "audio_processed": False, "network_during_inference": False},
             "accuracy_verified": False, "commercial_distribution_clearance": "not_audited"}
    def publish(row):
        file = f"run-{len(rows)+1:02d}.json"
        write_report(args.output_dir / file, row)
        rows.append({"file": file, **{key: row.get(key) for key in (
            "candidate", "pose_mode", "status", "completed_frames", "unpaced_loop_fps",
            "predictions_sha256", "error")}})
    try:
        probe = inspect(args.input)
        try:
            ort_version = version("onnxruntime")
        except PackageNotFoundError:
            ort_version = None
        state.update(source=probe.summary(), runtime={"python": platform.python_version(),
                     "platform": platform.platform(), "onnxruntime": ort_version,
                     "pinned_onnxruntime": ORT_VERSION, "source": _source_revision()},
                     configuration={"models": args.models, "providers": args.providers,
                     "repeats": args.repeats, "requested_prefix_frames": args.max_frames,
                     "target_fps": args.target_fps, "ort_threads": args.ort_threads,
                     "decode_threads": args.decode_threads, "score_threshold": args.score_threshold,
                     "warmup_frames": args.warmup_frames, "preview": "none",
                     "detector_score_threshold": .7, "detector_nms_iou": .45,
                     "person_crop_padding": 1.25, "model_color_order": "BGR",
                     "resampling": "detector416x416; pose256x192 crop; source pixels unchanged",
                     "capability_equivalent_to_mediapipe": False})
        write_report(args.output_dir / "manifest.json", state)
        cells = [(model, mode) for model in args.models for mode in args.providers]
        plan = [cell for repeat in range(args.repeats)
                for cell in (cells if repeat % 2 == 0 else list(reversed(cells)))]
        if args.include_mediapipe:
            plan = [("mediapipe", "cpu"), *plan, ("mediapipe", "cpu")]
        for model, mode in plan:
            intent = {"candidate": model, "pose_mode": mode, "status": "started",
                      "no_terminal_record_means": "interrupted process; NOT successful"}
            write_report(args.output_dir / f"run-{len(rows)+1:02d}.started.json", intent)
            print(f"START {model}/{mode}", flush=True)
            if model == "mediapipe":
                failure = {}
                native_args = SimpleNamespace(
                    input=args.input, mode="track", model_dir=args.mediapipe_model_dir,
                    task_scheduling="parallel", verify_pixels=False, verify_results=True,
                    preview="none", decode_threads=args.decode_threads,
                    warmup_frames=args.warmup_frames, target_fps=args.target_fps)
                try:
                    row = run_pass(native_args, probe, frame_limit=args.max_frames,
                                   failure_record=failure)
                except BaseException as exc:
                    row = {**failure, "status": "interrupted" if isinstance(exc, KeyboardInterrupt)
                           else "failed", "error": {"type": type(exc).__name__},
                           "unpaced_loop_fps": None}
                row.update(candidate=model, pose_mode=mode,
                           completed_frames=row.get("all_frames", {}).get("frames", 0),
                           capability_equivalent_to_wholebody133=False,
                           role="native_control_with_world_landmarks_and_face_blendshapes")
            else:
                row, bank = candidate(args, probe, model, mode, references.get(model))
                if bank is not None:
                    references[model] = bank
            publish(row)
            print(f"END {model}/{mode}: {row['status']}", flush=True)
            if row["status"] == "interrupted" or row.get("fatal_source_change"):
                break
        if digest_file(args.input) != probe.sha256:
            raise SourceChanged("Final source integrity mismatch")
        state["status"] = ("interrupted" if any(r["status"] == "interrupted" for r in rows)
                           else "completed" if len(rows) == len(plan)
                           and all(r["status"] == "completed" for r in rows)
                           else "completed_with_failures")
    except BaseException as exc:
        state["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        state["error"] = {"type": type(exc).__name__}
    finally:
        references.clear()
        write_report(args.output_dir / "summary.json", state)
    return 0 if state["status"] == "completed" else 2


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_DIR)
    p.add_argument("--mediapipe-model-dir", type=Path, default=Path("models"))
    p.add_argument("--models", nargs="+", choices=POSE_MODELS, default=list(POSE_MODELS))
    p.add_argument("--providers", nargs="+", choices=tuple(MODES), default=["cpu", "coreml-all"])
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=900,
                   help="explicit prefix; 0 processes full file")
    p.add_argument("--target-fps", type=float, default=60.0)
    p.add_argument("--ort-threads", type=int, default=4)
    p.add_argument("--decode-threads", type=int, default=0)
    p.add_argument("--warmup-frames", type=int, default=60)
    p.add_argument("--score-threshold", type=float, default=.3)
    p.add_argument("--include-mediapipe", action="store_true")
    p.add_argument("--output-dir", type=Path, required=True)
    return p


def main():
    args = parser().parse_args()
    if (not 1 <= args.repeats <= 4 or args.max_frames < 0 or args.warmup_frames < 0
            or not 1 <= args.ort_threads <= 32 or not 0 <= args.decode_threads <= 64
            or not math.isfinite(args.target_fps) or args.target_fps <= 0
            or not math.isfinite(args.score_threshold) or args.score_threshold < 0
            or len(set(args.models)) != len(args.models)
            or len(set(args.providers)) != len(args.providers)):
        raise ValueError("Invalid benchmark configuration")
    return run_matrix(args)


if __name__ == "__main__":
    sys.exit(main())
