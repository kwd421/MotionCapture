"""Fixed-60Hz admission validation for the proven same-frame pose batch-2 candidate.

The source frames, original PTS, detector, model, provider and ready-only handoff
are unchanged. Only B arms release source sequence numbers at an exact host-side
60Hz schedule. This is a file capacity/latency test, not camera or display proof.
"""
from __future__ import annotations

import argparse
import json
import platform
from copy import copy
from functools import partial
from pathlib import Path

from motioncapture.recording import inspect_recording
from motioncapture.recording_bench import _source_revision, write_report
from motioncapture.wholebody_catalog import ASSETS, BenchmarkError, sha256, verify_asset
from motioncapture.wholebody_fixed_rate import FixedRatePacer
from motioncapture.wholebody_onnx import OrtModel
from motioncapture.wholebody_optimize_bench import run_pass
from motioncapture.wholebody_pose_batch import PoseBatchModels
from motioncapture.wholebody_pose_batch_pipeline_lab import (
    BatchStagePipeline,
    _fix_batch_timing_labels,
)
from motioncapture.wholebody_replay import SourcePacer

PLAN = ("source_pts_batch2", "fixed60_batch2", "fixed60_batch2", "source_pts_batch2")


def _fixed60_pipeline_factory(*factories, **kwargs):
    original = kwargs.get("pacer")
    if not isinstance(original, SourcePacer):
        raise BenchmarkError("fixed60_lab_expected_source_pacer_placeholder")
    kwargs["pacer"] = FixedRatePacer(original.limit, rate_hz=60)
    return BatchStagePipeline(*factories, **kwargs)


def _checkpoint(path: Path, suffix: str, value) -> None:
    write_report(path.with_name(path.stem + suffix + ".json"), value)


def _label_row(row: dict, label: str) -> None:
    _fix_batch_timing_labels(row)
    fixed = label == "fixed60_batch2"
    row["source_pacing"] = "fixed_60hz" if fixed else "original_pts"
    row["execution_arm"] = {
        "detector_provider": "coreml-all",
        "pose_provider": "coreml-all",
        "pose_execution": "same_frame_batch2",
        "same_frame_pose_batch_size": 2,
        "pose_intra_op_threads": 4,
        "release_schedule": "fixed_60hz" if fixed else "original_pts",
    }
    if fixed:
        row["timing_scope"]["loop"] = (
            "fixed-60Hz host release paced file incl verification; no GUI")
        row["replay_ages"]["scope"] = (
            "fixed-60Hz scheduled file release to validation complete; not photon latency")
        row["output_cadence"]["scope"] = (
            "host validation completions grouped by ORIGINAL source PTS; "
            "input release is fixed 60Hz")
        row["output_cadence"]["interval_budget_note"] = (
            "Original PTS are preserved for identity/reporting "
            "but do not drive this arm's release. "
            "Inspect replay source-age and fixed-rate pacer summary for backlog.")


def _strict_summary(rows: list[dict]) -> dict:
    fixed = [row for row in rows
             if row.get("execution_arm", {}).get("release_schedule") == "fixed_60hz"]
    completed = len(fixed) == 2 and all(row.get("status") == "completed" for row in fixed)
    full_file = completed and all(row.get("scope") == "full_file" for row in fixed)
    preserved = full_file and all(
        row["pipeline"]["read_frames"] == row["pipeline"]["emitted_frames"]
        == row["pipeline"]["pose_requests"]
        and row["pipeline"]["intentional_frame_skips"] == 0
        and row["pipeline"]["unemitted_read_frames"] == 0
        and row["pipeline"]["unemitted_pose_requests"] == 0
        and row["pipeline"]["source_pacing"]["frames_skipped"] == 0
        for row in fixed)
    schedules = [row["pipeline"]["source_pacing"] for row in fixed
                 if row.get("pipeline") is not None]
    rate_exact = completed and len(schedules) == 2 and all(
        schedule.get("configured_rate_hz") == 60.0
        and schedule.get("scheduled_source_rate_hz") is not None
        and abs(schedule["scheduled_source_rate_hz"] - 60.0) < 1e-5
        for schedule in schedules)
    return {
        "runs": len(fixed),
        "all_completed": completed,
        "all_full_file": full_file,
        "all_frames_preserved": preserved,
        "configured_release_schedule_60hz": rate_exact,
        "observed_release_rate_hz": [
            row["pipeline"]["source_pacing"].get("observed_release_rate_hz")
            if row.get("pipeline") is not None else None for row in fixed],
        "source_pts_rewritten": False,
        "hardware_camera_emulated": False,
        "source_age_mean_ms": [row["replay_ages"]["source_age_ms"]["mean_ms"] for row in fixed],
        "source_age_p95_ms": [row["replay_ages"]["source_age_ms"]["p95_ms"] for row in fixed],
        "source_age_p99_ms": [row["replay_ages"]["source_age_ms"]["p99_ms"] for row in fixed],
        "source_age_max_ms": [row["replay_ages"]["source_age_ms"]["max_ms"] for row in fixed],
        "last_source_age_ms": [row["replay_ages"]["last_source_age_ms"] for row in fixed],
        "over_100ms_frames": [
            row["replay_ages"]["source_age_ms"]["threshold_exceedances"]["100.0"]
            for row in fixed],
        "full_file_replayed_with_60hz_schedule": completed and preserved and rate_exact,
        "live_camera_60fps_verified": False,
        "display_latency_verified": False,
        "latency_acceptance_threshold_predeclared": False,
    }


def execute(args, *, inspector=inspect_recording, pass_runner=run_pass) -> int:
    if args.output.exists():
        raise FileExistsError(args.output)
    if any(args.output.parent.glob(args.output.stem + ".*.json")):
        raise BenchmarkError("checkpoint_prefix_already_exists")
    report = {
        "schema_version": 2,
        "experiment": "same_frame_batch2_fixed60_replay_v1",
        "status": "running",
        "runs": [],
        "error": None,
        "live_60fps_verified": False,
        "accuracy_verified": False,
        "commercial_release_cleared": False,
        "plan": list(PLAN),
        "comparison_scope": (
            "same batch2 detector/model/provider/PTS/ready-handoff; "
            "only host release schedule differs"),
        "fixed_rate_is_camera_emulation": False,
        "source_pts_rewritten": False,
        "privacy": {"frames_written": False, "coordinates_written": False,
                    "audio_processed": False},
    }
    phase = "inspect"
    try:
        if not args.input.is_file():
            raise BenchmarkError("input_video_missing")
        probe = inspector(args.input)
        report["source"] = probe.summary()
        report["runtime"] = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "source": _source_revision(),
        }
        paths = {key: verify_asset(args.asset_dir, key)
                 for key in ("yolox-tiny", args.model)}
        report["assets"] = {key: receipt for key, (_, receipt) in paths.items()}
        detector_factory = partial(
            OrtModel, paths["yolox-tiny"][0], ASSETS["yolox-tiny"].shape, "coreml-all",
            allow_cpu=True, threads=4)
        batch_pose_factory = partial(
            PoseBatchModels, paths[args.model][0], ASSETS[args.model].shape, "coreml-all",
            allow_cpu=True, threads=4)
        run_args = copy(args)
        run_args.suite = "pose-parallel"
        run_args.detector_provider = run_args.pose_provider = "coreml-all"
        run_args.pose_intra_op_threads = 4
        report["configuration"] = {
            "model": args.model,
            "detector_provider": "coreml-all",
            "pose_provider": "coreml-all",
            "pose_execution": "same_frame_batch2",
            "fixed_release_rate_hz": 60,
            "source_pts_span_fps": report["source"].get("pts_span_fps"),
            "source_pts_preserved": True,
            "future_frame_batching": False,
            "detector_every_source_frame": True,
            "frames_skipped": 0,
            "ready_handoff": True,
        }
        _checkpoint(args.output, ".manifest", report)
        phase = "passes"
        reference = None
        for index, label in enumerate(PLAN):
            _checkpoint(args.output, f".arm-{index + 1:02d}.started",
                        {"status": "started", "release_schedule": label})
            print(f"ARM {index + 1}/4: release_schedule={label}", flush=True)
            pipeline_factory = (_fixed60_pipeline_factory
                                if label == "fixed60_batch2" else BatchStagePipeline)
            row, enrolled = pass_runner(
                run_args, probe, "source-pts-ready-cvlut",
                (detector_factory, batch_pose_factory), reference,
                pipeline_factory=pipeline_factory)
            _label_row(row, label)
            report["runs"].append(row)
            _checkpoint(args.output, f".arm-{index + 1:02d}", row)
            if enrolled is not None:
                reference = enrolled
            if row["status"] != "completed":
                report["status"], report["error"] = row["status"], row["error"]
                break
        else:
            report["status"] = "completed"
        phase = "final_integrity"
        if sha256(args.input) != probe.sha256:
            raise BenchmarkError("source_changed_at_end")
        complete = len(report["runs"]) == 4 and all(
            row["status"] == "completed" for row in report["runs"])
        for name, key in (
            ("all_pass_pixel_hashes_equal", "pixels_sha256"),
            ("all_pass_detector_hashes_equal", "detector_predictions_sha256"),
            ("all_pass_prediction_hashes_equal", "predictions_sha256"),
        ):
            report[name] = complete and all(row.get(key) is not None for row in report["runs"]) \
                and len({row[key] for row in report["runs"]}) == 1
        report["fixed_60hz"] = _strict_summary(report["runs"])
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
    p.add_argument("--model", choices=("dwpose-m",), default="dwpose-m")
    p.add_argument("--detector-provider", choices=("coreml-all",), default="coreml-all")
    p.add_argument("--pose-provider", choices=("coreml-all",), default="coreml-all")
    p.add_argument("--allow-cpu-partitions", action="store_true")
    p.add_argument("--research-only", action="store_true", required=True)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--decode-threads", type=int, default=0)
    p.add_argument("--ort-threads", type=int, default=4)
    p.add_argument("--output", type=Path, required=True)
    return p


def main() -> int:
    args = parser().parse_args()
    if (not args.allow_cpu_partitions or args.ort_threads != 4
            or not 0 <= args.max_frames <= 120000
            or not 0 <= args.decode_threads <= 64):
        raise BenchmarkError("fixed60_batch_lab_requires_fixed_coreml_all_plan")
    return execute(args)


if __name__ == "__main__":
    try:
        code = main()
    except KeyboardInterrupt:
        code = 130
    except Exception as failure:
        print(json.dumps({"status": "failed", "type": type(failure).__name__,
                          "code": getattr(failure, "code", None)}))
        code = 2
    raise SystemExit(code)
