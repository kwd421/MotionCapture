"""Explicit same-frame pose batching for symbolic-batch SimCC ONNX models.

Two static CoreML/ORT specializations are owned together: batch 1 and batch 2.
Routing is selected by the caller; batch-2 failure is terminal and never retried
as batch-1 work. No future-frame batching or missing-person padding is performed.
"""
from __future__ import annotations

import gc
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from motioncapture.wholebody_catalog import BenchmarkError
from motioncapture.wholebody_onnx import OrtModel, profile_placement, provider_plan


class Batch2OrtModel:
    """Batch-2 specialization of the same symbolic-batch ONNX pose model."""

    def __init__(self, path: Path, shape: tuple[int, ...], provider: str, *,
                 allow_cpu: bool = False, threads: int = 4):
        if len(shape) != 4 or shape[0] != 2 or shape[1] != 3 or min(shape[2:]) <= 0:
            raise BenchmarkError("batch2_requires_nchw_shape")
        if not 1 <= threads <= 64:
            raise BenchmarkError("invalid_ort_thread_count")
        import onnx
        import onnxruntime as ort

        if ort.__version__ != "1.22.1":
            raise BenchmarkError("requires_onnxruntime_1_22_1")
        providers, expected = provider_plan(provider, ort.get_available_providers(),
                                             allow_cpu=allow_cpu)
        graph = onnx.load(str(path), load_external_data=False)

        def check_tensors(message):
            if (isinstance(message, onnx.TensorProto)
                    and message.data_location == onnx.TensorProto.EXTERNAL):
                raise BenchmarkError("external_tensor_assets_unsupported")
            for field, value in message.ListFields():
                if field.type == field.TYPE_MESSAGE:
                    children = value if field.is_repeated else (value,)
                    for child in children:
                        check_tensors(child)

        check_tensors(graph)
        inputs = [i for i in graph.graph.input
                  if i.name not in {t.name for t in graph.graph.initializer}]
        if len(inputs) != 1 or inputs[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise BenchmarkError("expected_single_float32_input")
        dims = inputs[0].type.tensor_type.shape.dim
        if len(dims) != 4 or not dims[0].dim_param:
            raise BenchmarkError("batch2_requires_symbolic_batch")
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
        del graph
        self.session = None
        self.shape = shape
        started = time.perf_counter_ns()
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
                outputs = self.run(np.zeros(shape, np.float32))
                expected_shapes = sorted([
                    (2, 133, shape[3] * 2), (2, 133, shape[2] * 2),
                ])
                if sorted(v.shape for v in outputs) != expected_shapes:
                    raise BenchmarkError("batch2_preflight_output_schema_mismatch")
                if any(not np.isfinite(v).all() for v in outputs):
                    raise BenchmarkError("preflight_nonfinite_outputs")
                profile = self.session.end_profiling()
                placement = profile_placement(json.loads(Path(profile).read_text()),
                                                expected, allow_cpu)
                self.metadata = {
                    "requested": provider, "requested_providers": providers,
                    "registered_providers": effective,
                    "reported_options": self.session.get_provider_options(),
                    "ort_cpu_partitions_explicitly_allowed": allow_cpu,
                    "dynamic_dimension_bindings": bindings, "placement": placement,
                    "input_shape": shape, "intra_op_threads": threads,
                    "preflight_ms": (time.perf_counter_ns() - started) / 1e6,
                    "preflight_source": "synthetic batch2 zero tensor; not measured tracking",
                    "profiling_during_timed_pass": False,
                }
            except BaseException:
                self.close()
                raise

    def run(self, tensor: np.ndarray) -> list[np.ndarray]:
        if (self.session is None or not isinstance(tensor, np.ndarray)
                or tensor.dtype != np.float32 or tensor.shape != self.shape
                or not tensor.flags.c_contiguous):
            raise BenchmarkError("invalid_batch2_pose_input")
        return self.session.run(self.output_names, {self.input_name: tensor})

    def close(self) -> None:
        self.session = None
        gc.collect()


class PoseBatchModels:
    """Own batch-1 and batch-2 specializations on one pose worker."""

    def __init__(self, path: Path, shape: tuple[int, ...], provider: str, *,
                 allow_cpu: bool = False, threads: int = 4,
                 single_factory=OrtModel, pair_factory=Batch2OrtModel):
        if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
            raise BenchmarkError("pose_batch_requires_batch1_base_shape")
        self.shape = shape
        self.models = {}
        self.metadata = {}
        try:
            single = single_factory(path, shape, provider, allow_cpu=allow_cpu, threads=threads)
            self.models[1] = single
            pair = pair_factory(path, (2, *shape[1:]), provider,
                                allow_cpu=allow_cpu, threads=threads)
            self.models[2] = pair
            if (single.metadata.get("requested") != provider
                    or pair.metadata.get("requested") != provider
                    or tuple(single.metadata.get("input_shape", ())) != shape
                    or tuple(pair.metadata.get("input_shape", ())) != (2, *shape[1:])
                    or single.metadata.get("intra_op_threads") != threads
                    or pair.metadata.get("intra_op_threads") != threads):
                raise BenchmarkError("pose_batch_session_metadata_mismatch")
            self.metadata = {
                **single.metadata,
                "same_frame_batching": {
                    "maximum_batch_size": 2,
                    "native_sessions": 2,
                    "specializations": {
                        "1": single.metadata,
                        "2": pair.metadata,
                    },
                    "same_model_file": True,
                    "model_file_modified": False,
                    "routing": "adjacent same-frame pairs; unpadded single remainder",
                    "parallel_model_calls": False,
                    "future_frame_batching": False,
                    "physical_dispatch_verified": False,
                },
            }
        except BaseException as primary:
            try:
                self.close()
            except BaseException as cleanup:
                primary.add_note(f"Pose batch cleanup failed: {type(cleanup).__name__}")
            raise

    def run(self, tensor: np.ndarray) -> list[np.ndarray]:
        if (not isinstance(tensor, np.ndarray) or tensor.dtype != np.float32
                or tensor.ndim != 4 or tensor.shape[1:] != self.shape[1:]
                or tensor.shape[0] not in (1, 2) or not tensor.flags.c_contiguous):
            raise BenchmarkError("invalid_same_frame_pose_batch")
        if not self.models:
            raise BenchmarkError("pose_batch_sessions_not_open")
        return self.models[tensor.shape[0]].run(tensor)

    def close(self) -> None:
        models, self.models = self.models, {}
        errors = []
        for model in reversed(list(models.values())):
            try:
                model.close()
            except BaseException as error:
                errors.append(error)
        if errors:
            for additional in errors[1:]:
                errors[0].add_note(f"Additional pose batch cleanup: {type(additional).__name__}")
            raise errors[0]
