"""In-process ONNX whole-body observations; no MediaPipe-shaped fabricated outputs.

The small model-specific codecs are adapted from Apache-2.0 RTMLib/MMPose/YOLOX.
See third_party/WHOLEBODY_NOTICE.md. OpenCV owns affine warping and NMS.
"""
from __future__ import annotations

import gc
import json
import platform
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from motioncapture.wholebody_catalog import ASSETS, BenchmarkError, verify_asset

PROVIDERS = ("cpu", "coreml-gpu", "coreml-ane", "coreml-all")
PARTS = {"body": (0, 17), "feet": (17, 23), "face": (23, 91),
         "left_hand": (91, 112), "right_hand": (112, 133)}
CAPABILITIES = {
    "schema": "COCO-WholeBody-133-2D", "body_points": 17, "foot_points": 6,
    "face_points": 68, "hand_points_each": 21, "face_blendshapes": False,
    "metric_3d": False, "monocular_world_landmarks": False, "bone_rotations": False,
    "persistent_actor_identity": False,
}


def _finite(a: np.ndarray, shape: tuple[int, ...], code: str) -> None:
    if not isinstance(a, np.ndarray) or a.shape != shape or not np.isfinite(a).all():
        raise BenchmarkError(code)


@dataclass(frozen=True)
class Person2D:
    xy: np.ndarray
    scores: np.ndarray
    valid: np.ndarray

    def __post_init__(self):
        _finite(self.xy, (133, 2), "invalid_keypoint_coordinates")
        _finite(self.scores, (133,), "invalid_keypoint_scores")
        if self.valid.shape != (133,) or self.valid.dtype != np.bool_:
            raise BenchmarkError("invalid_keypoint_mask")
        # A score is an uncalibrated SimCC response, not a [0,1] probability.
        for value in (self.xy, self.scores, self.valid):
            value.flags.writeable = False


def pose_tensor(image: np.ndarray, bbox: np.ndarray,
                size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RTMLib's BGR/mean/std contract, 1.25 padding and aspect-preserving affine crop."""
    _finite(bbox, (4,), "invalid_person_box")
    if np.any(bbox[2:] <= bbox[:2]):
        raise BenchmarkError("invalid_person_box")
    width, height = size
    center = (bbox[:2] + bbox[2:]) * .5
    scale = (bbox[2:] - bbox[:2]) * 1.25
    scale = np.array((max(scale[0], scale[1] * width / height),
                      max(scale[1], scale[0] * height / width)))
    # Same three-point construction as RTMLib get_warp_matrix with rot=0.
    src = np.float32([center, center + [0, -scale[0] / 2],
                      center + [-scale[0] / 2, -scale[0] / 2]])
    dst = np.float32([[width / 2, height / 2], [width / 2, (height - width) / 2],
                      [0, (height - width) / 2]])
    matrix = cv2.getAffineTransform(src, dst)
    crop = cv2.warpAffine(image, matrix, size, flags=cv2.INTER_LINEAR)
    # Deliberately no extra BGR->RGB swap: matches this SDK's RTMLib recipe.
    normalized = (crop - np.array([123.675, 116.28, 103.53])) / np.array(
        [58.395, 57.12, 57.375])
    tensor = np.ascontiguousarray(normalized.transpose(2, 0, 1)[None], dtype=np.float32)
    return tensor, center, scale


def decode_pose(outputs: list[np.ndarray], size: tuple[int, int],
                center: np.ndarray, scale: np.ndarray, threshold: float) -> Person2D:
    width, height = size
    if len(outputs) != 2:
        raise BenchmarkError("expected_two_simcc_heads")
    # Use actual axis dimensions, not unverified output names or exported ordering.
    x = [v for v in outputs if v.shape == (1, 133, width * 2)]
    y = [v for v in outputs if v.shape == (1, 133, height * 2)]
    if len(x) != 1 or len(y) != 1:
        raise BenchmarkError("invalid_simcc_schema")
    x, y = x[0], y[0]
    _finite(x, (1, 133, width * 2), "nonfinite_simcc")
    _finite(y, (1, 133, height * 2), "nonfinite_simcc")
    # Preserve RTMLib 0.0.16's 2D mean-axis score semantics; do not clip to 1.
    scores = ((x.max(axis=2) + y.max(axis=2)) * .5)[0]
    locs = np.stack([x.argmax(axis=2), y.argmax(axis=2)], axis=-1)[0].astype(np.float32)
    locs[scores <= 0] = -1
    xy = locs / 2.0 / np.array(size) * scale + center - scale / 2
    return Person2D(xy, scores.copy(), (scores > 0) & (scores >= threshold))


def detector_tensor(image: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    ratio = min(416 / height, 416 / width)
    resized = cv2.resize(image, (int(width * ratio), int(height * ratio)),
                         interpolation=cv2.INTER_LINEAR)
    padded = np.full((416, 416, 3), 114, dtype=np.uint8)
    padded[:resized.shape[0], :resized.shape[1]] = resized
    return np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32), ratio


def decode_people(output: np.ndarray, ratio: float, threshold: float = .5,
                  nms_threshold: float = .45) -> np.ndarray:
    """Official raw YOLOX-tiny COCO output; filter person class before NMS."""
    _finite(output, (1, 3549, 85), "invalid_yolox_schema")
    predictions = output[0]
    confidence = predictions[:, 4] * predictions[:, 5]  # COCO person, class 0 only
    selected = confidence > threshold
    if not selected.any():
        return np.empty((0, 4), np.float32)
    grids, scales = [], []
    for stride in (8, 16, 32):
        yy, xx = np.mgrid[:416 // stride, :416 // stride]
        grids.append(np.stack([xx, yy], axis=-1).reshape(-1, 2))
        scales.append(np.full((xx.size, 1), stride))
    grid, strides = np.concatenate(grids)[selected], np.concatenate(scales)[selected]
    prediction = predictions[selected]
    center = (prediction[:, :2] + grid) * strides / ratio
    with np.errstate(over="ignore"):
        extent = np.exp(prediction[:, 2:4]) * strides / ratio
    if not np.isfinite(extent).all() or np.any(extent <= 0):
        raise BenchmarkError("invalid_yolox_box_extent")
    boxes = np.concatenate([center - extent / 2, extent], axis=1)
    # Mature OpenCV NMS (continuous box IoU); recorded as part of this adapter recipe.
    keep = cv2.dnn.NMSBoxes(boxes.tolist(), confidence[selected].tolist(),
                            threshold, nms_threshold)
    if len(keep) == 0:
        return np.empty((0, 4), np.float32)
    boxes = boxes[np.asarray(keep).reshape(-1)]
    return np.concatenate([boxes[:, :2], boxes[:, :2] + boxes[:, 2:]], axis=1).astype(
        np.float32)


RESEARCH_ORT_VERSIONS = frozenset({"1.22.1", "1.29.0"})


def provider_plan(name: str, available: list[str], *, allow_cpu: bool) -> tuple[list, str]:
    if name not in PROVIDERS:
        raise BenchmarkError("unknown_provider")
    if name == "cpu":
        if "CPUExecutionProvider" not in available:
            raise BenchmarkError("cpu_provider_unavailable")
        return ["CPUExecutionProvider"], "CPUExecutionProvider"
    if platform.system() != "Darwin" or platform.machine() not in {"arm64", "aarch64"}:
        raise BenchmarkError("coreml_requires_apple_silicon_macos")
    if "CoreMLExecutionProvider" not in available:
        raise BenchmarkError("coreml_provider_unavailable")
    units = {"coreml-gpu": "CPUAndGPU", "coreml-ane": "CPUAndNeuralEngine",
             "coreml-all": "ALL"}[name]
    providers = [("CoreMLExecutionProvider", {
        "ModelFormat": "MLProgram", "MLComputeUnits": units,
        "RequireStaticInputShapes": "1", "EnableOnSubgraphs": "0",
    })]
    if allow_cpu:
        providers.append("CPUExecutionProvider")
    return providers, "CoreMLExecutionProvider"


def profile_placement(events: list[dict], expected: str, allow_cpu: bool) -> dict:
    counts: Counter = Counter()
    for event in events:
        if event.get("cat") == "Node" and event.get("args", {}).get("provider"):
            counts[event["args"]["provider"]] += 1
    if counts[expected] == 0:
        raise BenchmarkError("requested_provider_has_no_profiled_nodes")
    if set(counts) - {expected, "CPUExecutionProvider"}:
        raise BenchmarkError("unexpected_execution_provider")
    if expected != "CPUExecutionProvider" and not allow_cpu and counts["CPUExecutionProvider"]:
        raise BenchmarkError("forbidden_cpu_partitions")
    return {"node_execution_events": dict(counts), "scope": "one_zero_tensor_preflight",
            "coreml_internal_hardware_verified": False, "ane_dispatch_verified": False}


class OrtModel:
    """Own one session, explicit provider and bounded placement preflight."""

    def __init__(self, path: Path, shape: tuple[int, ...], provider: str,
                 *, allow_cpu: bool = False, threads: int = 4):
        self.session = None
        self.metadata = {}
        self.shape = shape
        if not 1 <= threads <= 64:
            raise BenchmarkError("invalid_ort_thread_count")
        import onnx
        import onnxruntime as ort

        if ort.__version__ not in RESEARCH_ORT_VERSIONS:
            raise BenchmarkError("unsupported_research_onnxruntime_version")
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
        if len(dims) != 4:
            raise BenchmarkError("expected_nchw_input")
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
        start = time.perf_counter_ns()
        with tempfile.TemporaryDirectory(prefix="mocap-ort-profile-") as directory:
            options.enable_profiling = True
            options.profile_file_prefix = str(Path(directory) / "placement")
            try:
                self.session = ort.InferenceSession(str(path), sess_options=options,
                                                     providers=providers)
                # The public Python constructor may try a substitute at initialization.
                # Reject it BEFORE any inference. Disable its run-time retry path too.
                self.session.disable_fallback()
                effective = self.session.get_providers()
                if expected not in effective:
                    raise BenchmarkError("provider_substitution_rejected")
                self.input_name = self.session.get_inputs()[0].name
                self.output_names = [o.name for o in self.session.get_outputs()]
                preflight_outputs = self.run(np.zeros(shape, dtype=np.float32))
                expected_shapes = ([(1, 3549, 85)] if shape == (1, 3, 416, 416)
                                   else [(1, 133, shape[3] * 2), (1, 133, shape[2] * 2)])
                if sorted(v.shape for v in preflight_outputs) != sorted(expected_shapes):
                    raise BenchmarkError("preflight_output_schema_mismatch")
                if any(not np.isfinite(v).all() for v in preflight_outputs):
                    raise BenchmarkError("preflight_nonfinite_outputs")
                profile = self.session.end_profiling()
                placement = profile_placement(json.loads(Path(profile).read_text()),
                                                expected, allow_cpu)
                self.metadata = {
                    "onnxruntime_version": ort.__version__,
                    "requested": provider, "requested_providers": providers,
                    "registered_providers": effective,
                    "reported_options": self.session.get_provider_options(),
                    "ort_cpu_partitions_explicitly_allowed": allow_cpu,
                    "dynamic_dimension_bindings": bindings, "placement": placement,
                    "input_shape": shape, "intra_op_threads": threads,
                    "preflight_ms": (time.perf_counter_ns() - start) / 1e6,
                    "preflight_source": "synthetic zero tensor; not measured tracking",
                    "profiling_during_timed_pass": False,
                }
            except BaseException:
                self.close()
                raise

    def run(self, tensor: np.ndarray) -> list[np.ndarray]:
        if self.session is None:
            raise BenchmarkError("session_not_open")
        if tensor.dtype != np.float32 or tensor.shape != self.shape:
            raise BenchmarkError("invalid_input_tensor")
        return self.session.run(self.output_names, {self.input_name: tensor})

    def close(self):
        self.session = None
        gc.collect()  # releasing Python ownership is not proof of driver-memory reclamation


class WholebodyEstimator:
    def __init__(self, root: Path, model: str, provider: str, *, detector_provider: str = "cpu",
                 allow_cpu: bool = False, threads: int = 4, keypoint_threshold: float = .3,
                 max_people: int = 8, session_factory=OrtModel):
        self.detector = self.pose = None
        self.metadata = {}
        self.last_detector_boxes = None
        self.last_pose_person_inference_ms = None
        self.keypoint_threshold, self.max_people = keypoint_threshold, max_people
        if not np.isfinite(keypoint_threshold) or keypoint_threshold < 0 or max_people < 1:
            raise BenchmarkError("invalid_estimator_limits")
        if model not in ASSETS or ASSETS[model].kind != "simcc133":
            raise BenchmarkError("unknown_pose_model")
        if provider not in PROVIDERS:
            raise BenchmarkError("unknown_provider")
        if detector_provider not in PROVIDERS:
            raise BenchmarkError("unknown_detector_provider")
        paths = {key: verify_asset(root, key) for key in ("yolox-tiny", model)}
        try:
            detector_kwargs = {"threads": threads}
            if detector_provider != "cpu":
                detector_kwargs["allow_cpu"] = allow_cpu
            self.detector = session_factory(
                paths["yolox-tiny"][0], ASSETS["yolox-tiny"].shape, detector_provider,
                **detector_kwargs)
            self.pose = session_factory(paths[model][0], ASSETS[model].shape, provider,
                                        threads=threads, allow_cpu=allow_cpu)
            self.size = (ASSETS[model].shape[3], ASSETS[model].shape[2])
            self.metadata = {
                "detector_provider": detector_provider,
                "pose_provider": provider,
                "detector": self.detector.metadata, "pose": self.pose.metadata,
                "assets": {k: receipt for k, (_, receipt) in paths.items()},
                "recipe": "yolox-raw-coco-person/opencv-nms/bgr-rtmlib-simcc-mean-v1",
                "detector_threshold": .5, "nms_threshold": .45,
                "keypoint_threshold": keypoint_threshold,
                "score_interpretation": "uncalibrated SimCC response, not probability",
                "detector_cadence": "every_source_frame", "max_people": max_people,
            }
        except BaseException as exc:
            try:
                self.close()
            except BaseException:
                exc.add_note("Additional estimator cleanup failure")
            raise

    def process(self, image: np.ndarray) -> tuple[list[Person2D], dict[str, float]]:
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 2):
            raise BenchmarkError("invalid_source_image")
        if self.detector is None or self.pose is None:
            raise BenchmarkError("estimator_not_open")
        start = time.perf_counter_ns()
        tensor, ratio = detector_tensor(image)
        t1 = time.perf_counter_ns()
        outputs = self.detector.run(tensor)
        t2 = time.perf_counter_ns()
        if len(outputs) != 1:
            raise BenchmarkError("invalid_detector_outputs")
        boxes = decode_people(outputs[0], ratio)
        t3 = time.perf_counter_ns()
        if len(boxes) > self.max_people:
            raise BenchmarkError("person_capacity_exceeded")
        times = {"detector_pre_ms": (t1 - start) / 1e6,
                 "detector_inference_ms": (t2 - t1) / 1e6,
                 "detector_post_ms": (t3 - t2) / 1e6,
                 "pose_pre_ms": 0., "pose_inference_ms": 0., "pose_post_ms": 0.}
        people = []
        per_person_inference_ms = []
        for bbox in boxes:
            began = time.perf_counter_ns()
            tensor, center, scale = pose_tensor(image, bbox, self.size)
            a = time.perf_counter_ns()
            outputs = self.pose.run(tensor)
            b = time.perf_counter_ns()
            people.append(decode_pose(outputs, self.size, center, scale, self.keypoint_threshold))
            c = time.perf_counter_ns()
            times["pose_pre_ms"] += (a - began) / 1e6
            times["pose_inference_ms"] += (b - a) / 1e6
            times["pose_post_ms"] += (c - b) / 1e6
            per_person_inference_ms.append((b - a) / 1e6)
        # No person means no pose call, NEVER a synthetic full-image bbox.
        times["wholebody_service_ms"] = (time.perf_counter_ns() - start) / 1e6
        # Copies prevent a caller from observing mutable runtime-owned arrays.  The
        # benchmark reads these diagnostics immediately after this sequential call;
        # they stay outside the numeric timing map so existing callers retain its
        # float-only contract.
        self.last_detector_boxes = np.asarray(boxes, dtype=np.float32).copy()
        self.last_pose_person_inference_ms = tuple(per_person_inference_ms)
        return people, times

    def close(self):
        errors = []
        for name in ("pose", "detector"):
            session = getattr(self, name)
            if session is not None:
                try:
                    session.close()
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    setattr(self, name, None)
        if errors:
            raise BenchmarkError("estimator_cleanup_failed") from errors[0]
