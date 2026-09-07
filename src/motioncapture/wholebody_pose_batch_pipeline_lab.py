"""Full-file A/B lab for same-frame pose batch-2; live/default paths stay untouched.

A uses the established ready-only StagePipeline with a batch-1 pose session.
B uses identical source admission, PTS pacing, detector and ready handoff, but one
pose owner routes adjacent detector slots through a fixed batch-2 specialization.
No future frames are batched, no person is dropped, and batch-2 failure is terminal.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from copy import copy
from functools import partial
from pathlib import Path

import numpy as np

from motioncapture.recording import inspect_recording
from motioncapture.recording_bench import _source_revision, write_report
from motioncapture.wholebody_catalog import ASSETS, BenchmarkError, sha256, verify_asset
from motioncapture.wholebody_fast_input import check_recipe, fast_pose_tensor
from motioncapture.wholebody_onnx import OrtModel, decode_pose, pose_tensor
from motioncapture.wholebody_optimize_bench import run_pass
from motioncapture.wholebody_pose_batch import PoseBatchModels
from motioncapture.wholebody_stages import Posed, StagePipeline, detect


def pose_batched(session, submitted, detected, size, threshold, fast, kernel="numpy"):
    spec = session.metadata.get("same_frame_batching", {})
    if (spec.get("maximum_batch_size") != 2 or spec.get("future_frame_batching") is not False
            or spec.get("parallel_model_calls") is not False):
        raise BenchmarkError("pose_batch_session_contract_mismatch")
    started = time.perf_counter_ns()
    pre = inference = post = 0.0
    people, per_person = [], []
    prepare = (lambda image, box, size: fast_pose_tensor(image, box, size, kernel=kernel)) \
        if fast else pose_tensor
    boxes = detected.boxes
    for begin in range(0, len(boxes), 2):
        group = boxes[begin:begin + 2]
        prep_started = time.perf_counter_ns()
        tensors, centers, scales = [], [], []
        for box in group:
            tensor, center, scale = prepare(detected.frame.image_bgr, box, size)
            tensors.append(tensor)
            centers.append(center)
            scales.append(scale)
        model_input = (tensors[0] if len(tensors) == 1 else
                       np.ascontiguousarray(np.concatenate(tensors, axis=0), dtype=np.float32))
        a = time.perf_counter_ns()
        outputs = session.run(model_input)
        b = time.perf_counter_ns()
        if len(group) == 1:
            decoded = [decode_pose(outputs, size, centers[0], scales[0], threshold)]
        else:
            if len(outputs) != 2 or any(v.shape[0] != 2 for v in outputs):
                raise BenchmarkError("invalid_batch2_pose_outputs")
            decoded = [decode_pose([v[i:i + 1] for v in outputs], size,
                                   centers[i], scales[i], threshold) for i in range(2)]
        c = time.perf_counter_ns()
        call_ms = (b - a) / 1e6
        pre += (a - prep_started) / 1e6
        inference += call_ms
        post += (c - b) / 1e6
        people.extend(decoded)
        # Accounting only: a batch call has no isolated per-person duration.
        per_person.extend([call_ms / len(group)] * len(group))
    ended = time.perf_counter_ns()
    times = {**detected.times, "pose_pre_ms": pre, "pose_inference_ms": inference,
             "pose_post_ms": post, "pose_stage_ms": (ended - started) / 1e6,
             "pose_queue_ms": (started - submitted) / 1e6,
             "detector_to_pose_wait_ms": (started - detected.completed_ns) / 1e6,
             "decode_read_ms": detected.frame.decode_ms,
             "submit_to_pose_completion_ms": (ended - detected.submitted_ns) / 1e6}
    times["frame_work_ms"] = (detected.frame.decode_ms + times["detector_stage_ms"]
                              + times["pose_stage_ms"])
    return Posed(detected, people, times, tuple(per_person), ended)


class BatchStagePipeline(StagePipeline):
    """Ready-only StagePipeline variant; only same-frame pose execution differs."""

    def __enter__(self):
        result = super().__enter__()
        spec = self.metadata.get("pose", {}).get("same_frame_batching", {})
        if (spec.get("maximum_batch_size") != 2 or spec.get("native_sessions") != 2
                or spec.get("future_frame_batching") is not False):
            raise BenchmarkError("pose_batch_session_contract_mismatch")
        self.metadata["pose_execution"] = {
            "native_sessions": 2,
            "assignment": "adjacent_same_frame_pairs; unpadded_single_remainder",
            "scope": "within_one_source_frame; detector slot order preserved",
            "parallel_model_calls": False,
            "future_frame_batching": False,
            "native_full_pipeline_speedup_verified": False,
        }
        return result

    def packets(self, frames, count: int, *, overlap: bool, fast: bool,
                verify_eof: bool = False, advance_pose: bool = False,
                defer_pose: bool = False):
        if (self.used or count <= 0 or not overlap or not fast or not advance_pose or defer_pose
                or self.pose_lanes != 1 or self.normalization_kernel != "opencv"):
            raise BenchmarkError("invalid_pose_batch_pipeline_run")
        self.used = True
        self.advance_pose = True
        self.dependency_handoff = False
        last_pose_end = None
        prestarted = None

        def submit_pose(detected, index):
            if detected.frame.identity.sequence != index:
                raise BenchmarkError("pipeline_detection_identity_mismatch")
            submitted = self.pose.submit(
                pose_batched, detected, self.size, self.threshold, True, self.normalization_kernel)
            self.pose_requests += 1
            return None if last_pose_end is None else (submitted - last_pose_end) / 1e6

        iterator = iter(frames)

        def read_frame():
            self.check()
            frame = next(iterator)
            if frame.identity.sequence != self.read_frames:
                raise BenchmarkError("pipeline_frame_sequence_mismatch")
            self.read_frames += 1
            return frame

        def submit_detector(frame):
            arguments = (frame,) if self.pacer is None else (frame, self.pacer)
            self.detector.submit(detect, *arguments)

        first = read_frame()
        h, w = first.image_bgr.shape[:2]
        check_recipe(first.image_bgr, np.array([0., 0., w, h], np.float32), self.size,
                     kernel=self.normalization_kernel)
        submit_detector(first)
        del first
        for index in range(count):
            if prestarted is None:
                detected = self.detector.receive()
                submit_gap = submit_pose(detected, index)
            else:
                detected, submit_gap = prestarted
                prestarted = None
            if index + 1 < count:
                upcoming = read_frame()
                submit_detector(upcoming)
                del upcoming
            packet = self.pose.receive()
            self.check()
            if packet.detected is not detected:
                raise BenchmarkError("pipeline_pose_identity_mismatch")
            last_pose_end = packet.completed_ns
            if submit_gap is not None:
                packet.times["pose_submit_gap_ms"] = submit_gap
            if index + 1 < count:
                ready, upcoming_detection = self.detector.take_ready()
                if ready:
                    next_gap = submit_pose(upcoming_detection, index + 1)
                    prestarted = (upcoming_detection, next_gap)
                    self.ready_handoffs += 1
                else:
                    self.unready_handoffs += 1
                del upcoming_detection
                self.check()
            packet.times["result_residence_ms"] = (
                time.perf_counter_ns() - packet.completed_ns) / 1e6
            self.emitted_frames += 1
            yield packet
            del packet, detected
        if verify_eof:
            try:
                next(iterator)
            except StopIteration:
                pass
            else:
                raise BenchmarkError("unexpected_extra_source_frame")
        self.check()

    def snapshot(self):
        result = super().snapshot()
        result.update({"same_frame_pose_batch_size": 2,
                       "native_pose_sessions": 2,
                       "future_frame_pose_batching": False,
                       "batch2_failure_fallback": False})
        return result


def _checkpoint(path: Path, suffix: str, value) -> None:
    write_report(path.with_name(path.stem + suffix + ".json"), value)


def _fix_batch_timing_labels(row: dict) -> None:
    row["timing_scope"]["pose_inference_ms"] = (
        "sum of native pose-call wall times; one batch2 call covers two same-frame slots")
    for key in ("all_frames", "steady_after_initial_frames"):
        semantics = row[key]["pose_count_timing_semantics"]
        semantics["frame_pose_ms"] = (
            "sum of native pose-call wall times; batch2 call covers two detector slots")
        semantics["per_person_ms"] = (
            "batch1 duration; batch2 duration divided equally for accounting only")


def execute(args) -> int:
    if args.output.exists():
        raise FileExistsError(args.output)
    if any(args.output.parent.glob(args.output.stem + ".*.json")):
        raise BenchmarkError("checkpoint_prefix_already_exists")
    report = {
        "schema_version": 1,
        "experiment": "same_frame_pose_batch2_full_pipeline_v1",
        "status": "running",
        "runs": [],
        "error": None,
        "live_60fps_verified": False,
        "accuracy_verified": False,
        "commercial_release_cleared": False,
        "privacy": {"frames_written": False, "coordinates_written": False,
                    "audio_processed": False},
        "plan": ["serial", "batch2", "batch2", "serial"],
        "comparison_scope": "same detector/provider/model/PTS/ready-handoff; pose execution only",
    }
    phase = "inspect"
    try:
        if not args.input.is_file():
            raise BenchmarkError("input_video_missing")
        probe = inspect_recording(args.input)
        report["source"] = probe.summary()
        report["runtime"] = {"python": platform.python_version(), "platform": platform.platform(),
                             "source": _source_revision()}
        paths = {key: verify_asset(args.asset_dir, key) for key in ("yolox-tiny", args.model)}
        report["assets"] = {key: receipt for key, (_, receipt) in paths.items()}
        detector_factory = partial(
            OrtModel, paths["yolox-tiny"][0], ASSETS["yolox-tiny"].shape, "coreml-all",
            allow_cpu=True, threads=4)
        serial_pose_factory = partial(
            OrtModel, paths[args.model][0], ASSETS[args.model].shape, "coreml-all",
            allow_cpu=True, threads=4)
        batch_pose_factory = partial(
            PoseBatchModels, paths[args.model][0], ASSETS[args.model].shape, "coreml-all",
            allow_cpu=True, threads=4)
        run_args = copy(args)
        # Reuse mature per-frame hashes from the existing research suite without
        # selecting its dual-session pose implementation.
        run_args.suite = "pose-parallel"
        run_args.detector_provider = run_args.pose_provider = "coreml-all"
        run_args.pose_intra_op_threads = 4
        _checkpoint(args.output, ".manifest", report)
        phase = "passes"
        reference = None
        for index, label in enumerate(report["plan"]):
            batch = label == "batch2"
            _checkpoint(args.output, f".arm-{index + 1:02d}.started",
                        {"status": "started", "pose_execution": label})
            print(f"ARM {index + 1}/4: pose_execution={label}", flush=True)
            row, enrolled = run_pass(
                run_args, probe, "source-pts-ready-cvlut",
                (detector_factory, batch_pose_factory if batch else serial_pose_factory),
                reference, pipeline_factory=BatchStagePipeline if batch else StagePipeline)
            row["execution_arm"] = {
                "detector_provider": "coreml-all", "pose_provider": "coreml-all",
                "pose_execution": label, "same_frame_pose_batch_size": 2 if batch else 1,
                "pose_intra_op_threads": 4,
            }
            if batch:
                _fix_batch_timing_labels(row)
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
        complete = len(report["runs"]) == 4 and all(r["status"] == "completed" for r in report["runs"])
        report["all_pass_pixel_hashes_equal"] = complete and len(
            {r["pixels_sha256"] for r in report["runs"]}) == 1
        report["all_pass_detector_hashes_equal"] = complete and len(
            {r["detector_predictions_sha256"] for r in report["runs"]}) == 1
        report["all_pass_prediction_hashes_equal"] = complete and len(
            {r["predictions_sha256"] for r in report["runs"]}) == 1
        batch_rows = [r for r in report["runs"] if r.get("execution_arm", {}).get("pose_execution") == "batch2"]
        comparisons = [r.get("provider_disagreement", {}).get("per_frame_predictions", {})
                       for r in batch_rows]
        box_comparisons = [r.get("provider_disagreement", {}).get("per_frame_detector_boxes", {})
                           for r in batch_rows]
        report["batch_changes_confined_to_multi_person_frames"] = bool(comparisons) and all(
            item.get("status") == "compared"
            and item.get("changed_frames") == item.get("multi_person_frames_changed")
            for item in comparisons)
        report["batch_detector_boxes_unchanged"] = bool(box_comparisons) and all(
            item.get("changed_frames") == 0 and item.get("count_mismatch_frames") == 0
            for item in box_comparisons)
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
            or not 0 <= args.max_frames <= 120000 or not 0 <= args.decode_threads <= 64):
        raise BenchmarkError("pose_batch_pipeline_requires_fixed_coreml_all_plan")
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
