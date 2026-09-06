"""Run boundary diagnostics and exact-input normalization/stage-overlap experiments.

Additive entry point: existing live and wholebody_bench commands are untouched.
Default plan: sequential/reference, sequential/LUT, overlap/LUT, reverse order.
Each pass has fresh sessions. No claim of camera latency, ground truth or 60Hz.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import cv2
import numpy as np

from motioncapture.recording import RecordedDecoder, inspect_recording
from motioncapture.recording_bench import Samples, _source_revision, write_report
from motioncapture.wholebody_boundary_probe import run_diagnostics
from motioncapture.wholebody_catalog import ASSETS, CANDIDATES, BenchmarkError, sha256, verify_asset
from motioncapture.wholebody_onnx import CAPABILITIES, PARTS, PROVIDERS, OrtModel
from motioncapture.wholebody_stage_comparison import StageDifference, StageReference, update_digest
from motioncapture.wholebody_stages import StagePipeline

MODES = {"sequential-reference": (False, False), "sequential-lut": (False, True),
         "overlap-lut": (True, True), "overlap-ready-lut": (True, True)}
DEFAULT_MODES = ("sequential-reference", "sequential-lut", "overlap-lut")


class Stats:
    def __init__(self):
        self.frames = self.people = 0
        self.counts, self.stages, self.by_count = {}, {}, {}
        self.valid_points = dict.fromkeys(PARTS, 0)

    def add(self, packet):
        people = packet.people
        if len(people) != len(packet.detected.boxes) or len(people) != len(packet.per_person_ms):
            raise BenchmarkError("pose_call_count_mismatch")
        if not math.isclose(sum(packet.per_person_ms), packet.times["pose_inference_ms"],
                            rel_tol=1e-10, abs_tol=1e-8):
            raise BenchmarkError("pose_call_time_mismatch")
        self.frames += 1
        self.people += len(people)
        n = str(len(people))
        self.counts[n] = self.counts.get(n, 0) + 1
        self.by_count.setdefault(n, {"frame_pose_ms": Samples(), "per_person_ms": Samples()})
        self.by_count[n]["frame_pose_ms"].add(packet.times["pose_inference_ms"])
        for value in packet.per_person_ms:
            self.by_count[n]["per_person_ms"].add(value)
        for name, value in packet.times.items():
            self.stages.setdefault(name, Samples()).add(value)
        for name, (start, end) in PARTS.items():
            self.valid_points[name] += sum(int(p.valid[start:end].sum()) for p in people)

    def summary(self):
        return {"frames": self.frames, "person_observations": self.people,
                "person_count_distribution": self.counts,
                "valid_point_observations": self.valid_points, "counts_are_accuracy": False,
                "stages": {k: v.summary(1000/60) for k, v in self.stages.items()},
                "pose_by_person_count": {n: {k: v.summary(1000/60) for k, v in group.items()}
                                         for n, group in self.by_count.items()}}


def run_pass(args, probe, mode, factories, reference=None, pipeline_factory=StagePipeline):
    count = min(args.max_frames or len(probe.pts), len(probe.pts))
    overlap, fast = MODES[mode]
    advance = mode == "overlap-ready-lut"
    report = {"mode": mode, "status": "running", "scope": "full_file" if count == len(probe.pts)
              else "explicit_prefix", "requested_frames": count, "error": None,
              "capabilities": CAPABILITIES, "live_60fps_verified": False,
              "ground_truth_accuracy_verified": False, "cleanup_errors": [],
              "same_provider_in_all_arms": True,
              "pose_handoff": "ready_before_verification" if advance else "after_verification",
              "timing_scope": {"loop": "all-frame unpaced service incl verification; no GUI",
                               "frame_work_ms": (
                                   "sum of host-measured stage wall times; NOT latency"),
                               "submit_to_pose_completion_ms": "host detector submit to pose end",
                               "source_to_photon_measured": False,
                               "initial_recipe_check_in_loop": fast,
                               "pose_submit_gap_ms": (
                                   "prior pose completion to next submission; N-1"),
                               "verified_output_interval_ms": (
                                   "successive validation completions; N-1")}}
    stats, steady = Stats(), Stats()
    digest, boxes_digest, pixels_digest = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    new_reference = StageReference(count) if reference is None else None
    comparison = StageDifference()
    pipeline = decoder = None
    phase, completed = "source_integrity", None
    loop_start = loop_end = None
    previous_verified_ns = None
    try:
        if sha256(args.input) != probe.sha256:
            raise BenchmarkError("source_changed_before_pass")
        phase = "session_setup"
        started = time.perf_counter_ns()
        pipeline = pipeline_factory(*factories, size=(ASSETS[args.model].shape[3],
                                                      ASSETS[args.model].shape[2]))
        with pipeline:
            report["backend"] = pipeline.metadata
            phase = "decoder_open"
            with RecordedDecoder(args.input, probe, threads=args.decode_threads) as decoder:
                report["decode_threads_reported"] = decoder.actual_threads
                report["setup_ms"] = (time.perf_counter_ns()-started)/1e6
                phase = "process"
                loop_start = time.perf_counter_ns()
                handoff_options = {"advance_pose": True} if advance else {}
                for packet in pipeline.packets(decoder, count, overlap=overlap, fast=fast,
                                               verify_eof=count == len(probe.pts),
                                               **handoff_options):
                    frame = packet.detected.frame
                    began = time.perf_counter_ns()
                    if new_reference is not None:
                        new_reference.store(frame, packet.people)
                    else:
                        comparison.add(reference, frame, packet.people)
                    update_digest(digest, frame, packet.people)
                    identity = f"{frame.identity.sequence}:{frame.identity.pts};".encode()
                    boxes_digest.update(identity)
                    boxes_digest.update(np.asarray(packet.detected.boxes, dtype="<f4").tobytes())
                    pixels_digest.update(identity)
                    pixels_digest.update(memoryview(np.ascontiguousarray(frame.image_bgr)))
                    verified_ns = time.perf_counter_ns()
                    packet.times["verification_ms"] = (verified_ns-began)/1e6
                    if previous_verified_ns is not None:
                        packet.times["verified_output_interval_ms"] = (
                            verified_ns-previous_verified_ns)/1e6
                    previous_verified_ns = verified_ns
                    stats.add(packet)
                    if frame.identity.sequence >= 60:
                        steady.add(packet)
                    completed = {"sequence": frame.identity.sequence, "pts": frame.identity.pts,
                                 "time_base": str(frame.identity.time_base)}
                    if stats.frames % 300 == 0:
                        print(f"{mode}: {stats.frames}/{count}", flush=True)
                    del packet, frame
                loop_end = time.perf_counter_ns()
                if stats.frames != count or (count == len(probe.pts) and not decoder.complete):
                    raise BenchmarkError("pipeline_frame_coverage_mismatch")
                phase = "decoder_cleanup"
            phase = "source_integrity"
            if sha256(args.input) != probe.sha256:
                raise BenchmarkError("source_changed_during_pass")
            phase = "stage_cleanup"
        report["status"] = "completed"
    except BaseException as exc:
        if loop_end is None:
            loop_end = time.perf_counter_ns()
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["error"] = {"phase": phase, "type": type(exc).__name__,
                           "code": getattr(exc, "code", None), "last_completed": completed}
        new_reference = None
    duration = (loop_end-loop_start)/1e9 if loop_start is not None else None
    ok = report["status"] == "completed"
    report.update(loop_s=duration, unpaced_loop_fps=count/duration if ok and duration else None,
                  all_frames=stats.summary(), steady_after_initial_frames=steady.summary(),
                  last_completed=completed,
                  pipeline=pipeline.snapshot() if pipeline is not None else None,
                  decoder_cleanup=("owner_released"
                                   if decoder is not None and decoder._capture is None
                                   else "unknown"),
                  predictions_sha256=digest.hexdigest() if stats.frames else None,
                  detector_predictions_sha256=boxes_digest.hexdigest() if stats.frames else None,
                  pixels_sha256=pixels_digest.hexdigest() if stats.frames else None,
                  hash_scope="complete_pass" if ok else "completed_prefix",
                  reference_hash_equal=(digest.hexdigest() == reference.predictions_sha256
                                        if reference is not None and ok else None))
    if new_reference is not None:
        new_reference.predictions_sha256 = digest.hexdigest()
    else:
        report["provider_disagreement"] = {**comparison.summary(),
            "reference": "same-provider sequential/reference first completed pass; NOT truth"}
    return report, new_reference


def checkpoint(path, suffix, data):
    target = path.with_name(path.stem + suffix + ".json")
    write_report(target, data)


def execute(args, *, inspector=inspect_recording, pass_runner=run_pass):
    if args.output.exists():
        raise FileExistsError(args.output)
    if any(args.output.parent.glob(args.output.stem + ".*.json")):
        raise BenchmarkError("checkpoint_prefix_already_exists")
    report = {"schema_version": 1, "experiment": "wholebody_exact_input_stage_overlap_v1",
              "status": "running", "runs": [], "error": None,
              "performance_verdict": "requires_device_and_quality_review",
              "live_60fps_verified": False, "accuracy_verified": False,
              "commercial_release_cleared": False,
              "privacy": {"frames_written": False, "coordinates_written": False,
                          "audio_processed": False, "observations": "bounded_RAM_only"}}
    phase = "inspect"
    try:
        if not args.input.is_file():
            raise BenchmarkError("input_video_missing")
        probe = inspector(args.input)
        report["source"] = probe.summary()
        report["runtime"] = {"python": platform.python_version(), "platform": platform.platform(),
                             "source": _source_revision(), "opencv": cv2.__version__}
        for package in ("onnxruntime", "onnx", "opencv-python", "numpy"):
            try:
                report["runtime"][package] = version(package)
            except PackageNotFoundError:
                report["runtime"][package] = None
        paths = {k: verify_asset(args.asset_dir, k) for k in ("yolox-tiny", args.model)}
        report["assets"] = {key: receipt for key, (_, receipt) in paths.items()}
        def factory(key, provider):
            return partial(OrtModel, paths[key][0], ASSETS[key].shape, provider,
                           allow_cpu=args.allow_cpu_partitions, threads=args.ort_threads)
        factories = (factory("yolox-tiny", args.detector_provider),
                     factory(args.model, args.pose_provider))
        plan = [*DEFAULT_MODES, *reversed(DEFAULT_MODES)]
        if args.suite == "handoff":
            plan = ["overlap-lut", "overlap-ready-lut", "overlap-ready-lut", "overlap-lut"]
        elif args.suite == "pipeline":
            plan = ["sequential-lut", "overlap-lut", "overlap-lut", "sequential-lut"]
        elif args.suite == "normalize":
            plan = ["sequential-reference", "sequential-lut",
                    "sequential-lut", "sequential-reference"]
        elif args.suite == "diagnostics":
            plan = []
        report["configuration"] = {"plan": plan, "model": args.model,
            "detector_provider": args.detector_provider, "pose_provider": args.pose_provider,
            "allow_cpu_partitions": args.allow_cpu_partitions, "preview": "none",
            "max_frames": args.max_frames, "target_fps": 60, "warmup_frames": 60,
            "decoder_every_original_pts": True, "detector_cadence": "every_source_frame",
            "maximum_people": 8, "max_pending_each_stage": 1,
            "detector_threshold": .5, "nms_threshold": .45, "keypoint_threshold": .3,
            "person_crop_padding": 1.25, "pose_input_hw": list(ASSETS[args.model].shape[2:]),
            "ort_threads": args.ort_threads, "decode_threads": args.decode_threads,
            "verification_pixel_hash_in_all_loops": True,
            "ready_handoff_policy": "only_if_next_detector_already_done; no extra source admission",
            "comparison_to_previous_loop_fps_requires_same_observer_work": True}
        checkpoint(args.output, ".manifest", report)
        if args.diagnose_from:
            phase = "boundary_probe"
            checkpoint(args.output, ".diagnostics.started", {"status": "started"})
            diagnostic = run_diagnostics(args.input, probe, args.diagnose_from,
                min(args.max_frames or len(probe.pts), len(probe.pts)),
                factory("yolox-tiny", "cpu"), factory("yolox-tiny", args.detector_provider),
                factories[1], size=(ASSETS[args.model].shape[3], ASSETS[args.model].shape[2]))
            report["boundary_diagnostics"] = diagnostic
            checkpoint(args.output, ".diagnostics", diagnostic)
            if diagnostic["status"] != "completed":
                report["status"] = diagnostic["status"]
                report["error"] = diagnostic["error"]
                return 2
        phase = "passes"
        bank = None
        for index, mode in enumerate(plan):
            checkpoint(args.output, f".arm-{index+1:02d}.started",
                       {"mode": mode, "status": "started"})
            row, enrolled = pass_runner(args, probe, mode, factories, bank)
            report["runs"].append(row)
            checkpoint(args.output, f".arm-{index+1:02d}", row)
            if enrolled is not None:
                bank = enrolled
            if row["status"] != "completed":
                report["status"] = row["status"]
                report["error"] = row["error"]
                break  # Do not silently continue with a missing experiment arm.
        else:
            report["status"] = "completed"
        phase = "final_integrity"
        if sha256(args.input) != probe.sha256:
            raise BenchmarkError("source_changed_at_end")
        report["all_pass_prediction_hashes_equal"] = (
            len(report["runs"]) == len(plan) and bool(plan)
            and all(r["status"] == "completed" for r in report["runs"])
            and len({r["predictions_sha256"] for r in report["runs"]}) == 1)
        for name, key in (("all_pass_pixel_hashes_equal", "pixels_sha256"),
                          ("all_pass_detector_hashes_equal", "detector_predictions_sha256")):
            report[name] = (len(report["runs"]) == len(plan) and bool(plan)
                            and all(r["status"] == "completed" for r in report["runs"])
                            and all(r.get(key) is not None for r in report["runs"])
                            and len({r[key] for r in report["runs"]}) == 1)
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["error"] = {"phase": phase, "type": type(exc).__name__,
                           "code": getattr(exc, "code", None)}
    finally:
        write_report(args.output, report)
    return 0 if report["status"] == "completed" else 2


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--asset-dir", type=Path, default=Path("models/wholebody"))
    p.add_argument("--model", choices=CANDIDATES, default="dwpose-m")
    p.add_argument("--detector-provider", choices=PROVIDERS, default="coreml-all")
    p.add_argument("--pose-provider", choices=PROVIDERS, default="coreml-all")
    p.add_argument("--allow-cpu-partitions", action="store_true")
    p.add_argument("--research-only", action="store_true", required=True)
    p.add_argument("--max-frames", type=int, default=900)
    p.add_argument("--decode-threads", type=int, default=0)
    p.add_argument("--ort-threads", type=int, default=4)
    p.add_argument("--suite", choices=("optimization", "pipeline", "normalize", "diagnostics",
                                       "handoff"),
                   default="optimization")
    p.add_argument("--diagnose-from", type=Path)
    p.add_argument("--output", type=Path, required=True)
    return p


def main():
    args = parser().parse_args()
    if (not 0 <= args.max_frames <= 120000 or not 0 <= args.decode_threads <= 64
            or not 1 <= args.ort_threads <= 64):
        raise BenchmarkError("invalid_benchmark_limits")
    if args.suite == "diagnostics" and args.diagnose_from is None:
        raise BenchmarkError("diagnostics_requires_source_report")
    return execute(args)


if __name__ == "__main__":
    try:
        exit_code = main()
    except KeyboardInterrupt:
        exit_code = 130
    except Exception as failure:
        print(json.dumps({"status": "failed", "type": type(failure).__name__,
                          "code": getattr(failure, "code", None)}))
        exit_code = 2
    raise SystemExit(exit_code)
