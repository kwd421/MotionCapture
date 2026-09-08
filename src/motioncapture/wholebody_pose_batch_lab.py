"""Isolated same-frame pose batch feasibility lab; not a live or FPS benchmark.

The lab scans every original source frame with the selected detector, and when at
least two boxes are present compares two batch-1 pose calls with one batch-2 pose
call on the first two detector slots. Nothing is skipped in discovery and no
result is used by the live pipeline. A batch-2 failure is terminal, never retried
as batch-1 work.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import tempfile
import time
from pathlib import Path

import numpy as np

from motioncapture.recording import RecordedDecoder, inspect_recording
from motioncapture.recording_bench import Samples, _source_revision, write_report
from motioncapture.wholebody_catalog import ASSETS, BenchmarkError, sha256, verify_asset
from motioncapture.wholebody_fast_input import fast_pose_tensor
from motioncapture.wholebody_onnx import (
    PARTS,
    OrtModel,
    decode_people,
    decode_pose,
    detector_tensor,
    profile_placement,
    provider_plan,
)


def _point_deltas(reference, candidate):
    result = {}
    for name, (begin, end) in PARTS.items():
        a, b = reference.valid[begin:end], candidate.valid[begin:end]
        common = a & b
        values = np.linalg.norm(
            reference.xy[begin:end][common] - candidate.xy[begin:end][common], axis=1)
        result[name] = {
            "matched": int(len(values)),
            "sum_pixels": float(values.sum()) if len(values) else 0.0,
            "max_pixels": float(values.max()) if len(values) else 0.0,
            "above_1px": int((values > 1).sum()),
            "above_10px": int((values > 10).sum()),
            "reference_only_valid": int((a & ~b).sum()),
            "candidate_only_valid": int((b & ~a).sum()),
        }
    return result


def _merge_delta(total, row):
    for name, value in row.items():
        target = total.setdefault(name, {
            "matched": 0, "sum_pixels": 0.0, "max_pixels": 0.0,
            "above_1px": 0, "above_10px": 0,
            "reference_only_valid": 0, "candidate_only_valid": 0,
        })
        target["matched"] += value["matched"]
        target["sum_pixels"] += value["sum_pixels"]
        target["max_pixels"] = max(target["max_pixels"], value["max_pixels"])
        for key in ("above_1px", "above_10px", "reference_only_valid", "candidate_only_valid"):
            target[key] += value[key]


def _finish_delta(total):
    return {name: {**value, "mean_pixels": (
                value["sum_pixels"] / value["matched"] if value["matched"] else None)}
            for name, value in total.items()}


class BatchOrtModel:
    """Batch-2 specialization of the same ONNX pose model and provider contract."""

    def __init__(self, path: Path, shape: tuple[int, ...], provider: str, *,
                 allow_cpu: bool, threads: int):
        if shape[0] != 2 or len(shape) != 4 or shape[1] != 3:
            raise BenchmarkError("batch_lab_requires_batch2_nchw")
        import onnx
        import onnxruntime as ort
        if ort.__version__ != "1.22.1":
            raise BenchmarkError("requires_onnxruntime_1_22_1")
        providers, expected = provider_plan(provider, ort.get_available_providers(),
                                             allow_cpu=allow_cpu)
        graph = onnx.load(str(path), load_external_data=False)
        inputs = [i for i in graph.graph.input
                  if i.name not in {t.name for t in graph.graph.initializer}]
        if len(inputs) != 1 or inputs[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise BenchmarkError("expected_single_float32_input")
        dims = inputs[0].type.tensor_type.shape.dim
        if len(dims) != 4 or not dims[0].dim_param:
            raise BenchmarkError("batch_lab_requires_symbolic_batch")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        if provider != "cpu" and not allow_cpu:
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        bindings = {}
        for dim, value in zip(dims, shape, strict=True):
            if dim.dim_param:
                if dim.dim_param in bindings and bindings[dim.dim_param] != value:
                    raise BenchmarkError("conflicting_symbolic_dimensions")
                bindings[dim.dim_param] = value
                options.add_free_dimension_override_by_name(dim.dim_param, value)
            elif dim.dim_value != value:
                raise BenchmarkError("model_input_shape_mismatch")
        self.session = None
        start = time.perf_counter_ns()
        with tempfile.TemporaryDirectory(prefix="mocap-batch2-profile-") as directory:
            options.enable_profiling = True
            options.profile_file_prefix = str(Path(directory) / "placement")
            try:
                self.session = ort.InferenceSession(str(path), sess_options=options,
                                                     providers=providers)
                self.session.disable_fallback()
                effective = self.session.get_providers()
                if expected not in effective:
                    raise BenchmarkError("provider_substitution_rejected")
                self.input_name = self.session.get_inputs()[0].name
                self.output_names = [o.name for o in self.session.get_outputs()]
                output = self.run(np.zeros(shape, np.float32))
                wanted = sorted([(2, 133, shape[3] * 2), (2, 133, shape[2] * 2)])
                if sorted(v.shape for v in output) != wanted or any(
                    not np.isfinite(v).all() for v in output
                ):
                    raise BenchmarkError("batch2_preflight_output_schema_mismatch")
                profile = self.session.end_profiling()
                placement = profile_placement(
                    json.loads(Path(profile).read_text()), expected, allow_cpu)
                self.metadata = {
                    "requested": provider, "requested_providers": providers,
                    "registered_providers": effective,
                    "reported_options": self.session.get_provider_options(),
                    "ort_cpu_partitions_explicitly_allowed": allow_cpu,
                    "dynamic_dimension_bindings": bindings,
                    "placement": placement, "input_shape": shape,
                    "intra_op_threads": threads,
                    "preflight_ms": (time.perf_counter_ns() - start) / 1e6,
                    "preflight_source": "synthetic batch2 zero tensor; not tracking",
                }
            except BaseException:
                self.close()
                raise

    def run(self, tensor):
        if (self.session is None or tensor.dtype != np.float32
                or tensor.ndim != 4 or tensor.shape[0] != 2):
            raise BenchmarkError("invalid_batch2_pose_input")
        return self.session.run(self.output_names, {self.input_name: tensor})

    def close(self):
        self.session = None
        gc.collect()


def _decode_batch(outputs, size, centers, scales, threshold):
    if len(outputs) != 2 or any(v.shape[0] != 2 for v in outputs):
        raise BenchmarkError("invalid_batch2_pose_outputs")
    return [decode_pose([v[i:i+1] for v in outputs], size, centers[i], scales[i], threshold)
            for i in range(2)]


def execute(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    report = {"schema_version": 1, "experiment": "same_frame_pose_batch2_feasibility_v1",
              "status": "running", "error": None, "live_60fps_verified": False,
              "accuracy_verified": False, "commercial_release_cleared": False,
              "scope": "first_two_detected_slots_on_multi-person_frames; isolated pose kernel",
              "privacy": "no frames/tensors/coordinates written"}
    detector = serial = batch = None
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
        size = (ASSETS[args.model].shape[3], ASSETS[args.model].shape[2])
        phase = "session_setup"
        detector = OrtModel(paths["yolox-tiny"][0], ASSETS["yolox-tiny"].shape,
                            args.detector_provider, allow_cpu=args.allow_cpu_partitions,
                            threads=args.ort_threads)
        serial = OrtModel(paths[args.model][0], ASSETS[args.model].shape,
                          args.pose_provider, allow_cpu=args.allow_cpu_partitions,
                          threads=args.ort_threads)
        batch = BatchOrtModel(paths[args.model][0], (2, *ASSETS[args.model].shape[1:]),
                              args.pose_provider, allow_cpu=args.allow_cpu_partitions,
                              threads=args.ort_threads)
        report["backend"] = {"detector": detector.metadata, "serial_pose": serial.metadata,
                             "batch2_pose": batch.metadata}
        serial_ms, batch_ms = Samples(), Samples()
        compared, multi_frames, pair_observations = {}, 0, 0
        serial_hash, batch_hash = hashlib.sha256(), hashlib.sha256()
        phase = "scan_and_compare"
        with RecordedDecoder(args.input, probe, threads=args.decode_threads) as decoder:
            for frame in decoder:
                tensor, ratio = detector_tensor(frame.image_bgr)
                raw = detector.run(tensor)
                if len(raw) != 1:
                    raise BenchmarkError("invalid_detector_outputs")
                boxes = decode_people(raw[0], ratio)
                if len(boxes) > 8:
                    raise BenchmarkError("person_capacity_exceeded")
                if len(boxes) >= 2:
                    multi_frames += 1
                    tensors, centers, scales = [], [], []
                    for box in boxes[:2]:
                        value, center, scale = fast_pose_tensor(frame.image_bgr, box, size,
                                                                kernel="opencv")
                        tensors.append(value)
                        centers.append(center)
                        scales.append(scale)
                    pair = np.ascontiguousarray(np.concatenate(tensors, axis=0), dtype=np.float32)
                    reference_people = None
                    for mode in ("serial", "batch", "batch", "serial"):
                        began = time.perf_counter_ns()
                        if mode == "serial":
                            outputs = [serial.run(value) for value in tensors]
                            people = [decode_pose(out, size, centers[i], scales[i], .3)
                                      for i, out in enumerate(outputs)]
                            serial_ms.add((time.perf_counter_ns() - began) / 1e6)
                        else:
                            outputs = batch.run(pair)
                            people = _decode_batch(outputs, size, centers, scales, .3)
                            batch_ms.add((time.perf_counter_ns() - began) / 1e6)
                        if reference_people is None:
                            reference_people = people
                        elif mode == "batch":
                            for a, b in zip(reference_people, people, strict=True):
                                _merge_delta(compared, _point_deltas(a, b))
                        digest = serial_hash if mode == "serial" else batch_hash
                        digest.update(frame.identity.sequence.to_bytes(8, "little"))
                        for person in people:
                            digest.update(np.asarray(person.xy, dtype="<f8").tobytes())
                            digest.update(np.asarray(person.scores, dtype="<f8").tobytes())
                            digest.update(person.valid.tobytes())
                    pair_observations += 1
                if (frame.identity.sequence + 1) % 1000 == 0:
                    print(f"pose-batch-lab: {frame.identity.sequence + 1}/{len(probe.pts)}",
                          flush=True)
        if sha256(args.input) != probe.sha256:
            raise BenchmarkError("source_changed_during_lab")
        serial_summary, batch_summary = serial_ms.summary(1000/60), batch_ms.summary(1000/60)
        report["result"] = {
            "frames_scanned": len(probe.pts), "multi_person_frames": multi_frames,
            "pair_observations": pair_observations,
            "abba_runs_per_pair": 4, "serial_pair_ms": serial_summary,
            "batch2_pair_ms": batch_summary,
            "mean_pair_speedup": (serial_summary["mean_ms"] / batch_summary["mean_ms"]
                                  if batch_summary["mean_ms"] else None),
            "coordinate_disagreement": _finish_delta(compared),
            "serial_output_sha256": serial_hash.hexdigest(),
            "batch_output_sha256": batch_hash.hexdigest(),
            "hashes_directly_comparable": False,
            "hash_note": "serial and batch digests include different repeated-call ordering",
            "detector_every_source_frame": True,
            "future_frame_batching": False,
            "ground_truth_accuracy_verified": False,
        }
        report["status"] = "completed"
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["error"] = {"phase": phase, "type": type(exc).__name__,
                           "code": getattr(exc, "code", None)}
    finally:
        for model in (batch, serial, detector):
            if model is not None:
                try:
                    model.close()
                except BaseException as cleanup:
                    report.setdefault("cleanup_errors", []).append(type(cleanup).__name__)
                    if report["status"] == "completed":
                        report["status"] = (
                            "interrupted" if isinstance(cleanup, KeyboardInterrupt) else "failed")
                        report["error"] = {"phase": "model_cleanup",
                                           "type": type(cleanup).__name__,
                                           "code": getattr(cleanup, "code", None)}
        write_report(args.output, report)
    if report["status"] == "interrupted":
        return 130
    return 0 if report["status"] == "completed" else 2


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--asset-dir", type=Path, default=Path("models/wholebody"))
    p.add_argument("--model", default="dwpose-m", choices=("dwpose-m",))
    p.add_argument("--detector-provider", default="coreml-all", choices=("coreml-all",))
    p.add_argument("--pose-provider", default="coreml-all", choices=("coreml-all",))
    p.add_argument("--allow-cpu-partitions", action="store_true")
    p.add_argument("--research-only", action="store_true", required=True)
    p.add_argument("--decode-threads", type=int, default=0)
    p.add_argument("--ort-threads", type=int, default=4)
    p.add_argument("--output", required=True, type=Path)
    return p


def main():
    args = parser().parse_args()
    if not args.allow_cpu_partitions:
        raise BenchmarkError("batch_lab_requires_explicit_cpu_partitions")
    if not 0 <= args.decode_threads <= 64 or not 1 <= args.ort_threads <= 64:
        raise BenchmarkError("invalid_benchmark_limits")
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
