"""Strict in-process ONNX WholeBody adapter, not MediaPipe topology emulation.

Model pre/post conventions follow RTMLib (Apache-2.0); see THIRD_PARTY_WHOLEBODY.md.
The current slice detects on every frame using a fixed CPU YOLOX-tiny. All
returned person boxes are processed; identity tracking and 3D are not implemented.
"""
from __future__ import annotations

import json
import math
import tempfile
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ORT_VERSION = "1.23.2"
MODES = {"cpu": None, "coreml-gpu": "CPUAndGPU",
         "coreml-ane": "CPUAndNeuralEngine", "coreml-all": "ALL"}
PARTS = {"body": (0, 17), "feet": (17, 23), "face": (23, 91),
         "left_hand": (91, 112), "right_hand": (112, 133)}
CAPABILITIES = {"schema": "coco_wholebody_133_2d", "image_xy": "source_pixels",
                "world_xyz": "unsupported", "face_blendshapes": "unsupported",
                "bone_rotations": "unsupported", "actor_id": "unsupported",
                "point_score": "raw SimCC peak score; not a calibrated probability"}


class UnsupportedProvider(RuntimeError):
    pass


def tensor(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(image.transpose(2, 0, 1)[None], dtype=np.float32)


def pose_input(image: np.ndarray, bbox: np.ndarray,
               hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    box = np.asarray(bbox, dtype=np.float32)
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
        raise ValueError("Invalid person box")
    h, w = hw
    center = (box[:2] + box[2:]) * .5
    scale = (box[2:] - box[:2]) * 1.25
    if scale[0] > scale[1] * w / h:
        scale[1] = scale[0] * h / w
    else:
        scale[0] = scale[1] * w / h
    # Zero-rotation top-down affine, using OpenCV rather than a custom resampler.
    src = np.array([center, center + [0, -scale[0] / 2],
                    center + [-scale[0] / 2, -scale[0] / 2]], np.float32)
    dst = np.array([[w / 2, h / 2], [w / 2, h / 2 - w / 2],
                    [0, h / 2 - w / 2]], np.float32)
    crop = cv2.warpAffine(image, cv2.getAffineTransform(src, dst), (w, h),
                         flags=cv2.INTER_LINEAR)
    # These exported SDK graphs use the RTMLib BGR normalization convention.
    crop = (crop.astype(np.float32) - np.array([123.675, 116.28, 103.53], np.float32))
    crop /= np.array([58.395, 57.12, 57.375], np.float32)
    return tensor(crop), center, scale


def decode_pose(outputs: list[np.ndarray], center: np.ndarray, scale: np.ndarray,
                hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = hw
    if (len(outputs) != 2 or outputs[0].shape != (1, 133, w * 2)
            or outputs[1].shape != (1, 133, h * 2)):
        raise ValueError("Unsupported pose outputs; require two 133-point SimCC distributions")
    x, y = outputs
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite SimCC output")
    score = np.minimum(x.max(axis=-1), y.max(axis=-1))[0]
    xy = np.stack([x.argmax(axis=-1), y.argmax(axis=-1)], axis=-1)[0] / 2.0
    xy = xy / np.array([w, h]) * scale + center - scale / 2
    # Invalid peaks are retained as invalid scores, never called valid observations.
    return xy.astype(np.float32), score.astype(np.float32)


def detector_input(image: np.ndarray, size: int = 416) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    ratio = min(size / h, size / w)
    fitted = cv2.resize(image, (max(1, int(w * ratio)), max(1, int(h * ratio))),
                        interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[:fitted.shape[0], :fitted.shape[1]] = fitted
    return tensor(canvas), ratio


def decode_boxes(outputs: list[np.ndarray], ratio: float,
                 threshold: float = .7, nms: float = .45) -> np.ndarray:
    if not outputs or not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("Invalid detector output/scale")
    raw = outputs[0]
    if raw.ndim != 3 or raw.shape[0] != 1 or not np.isfinite(raw).all():
        raise ValueError("Unsupported/nonfinite detector output")
    if raw.shape[-1] == 5:
        # MMDeploy NMS exports return source-sized boxes plus score, and may
        # also return labels. This asset is a human-only detector.
        boxes, scores = raw[0, :, :4].copy(), raw[0, :, 4]
        if len(outputs) == 2:
            labels = outputs[1]
            if labels.shape != raw.shape[:2] or not np.isfinite(labels).all():
                raise ValueError("Invalid detector labels")
            if np.any(labels[0, scores > threshold] != 0):
                raise ValueError("Unexpected non-person class in human-only model")
        elif len(outputs) != 1:
            raise ValueError("Unexpected detector output count")
        boxes = boxes[scores > threshold] / ratio
    elif raw.shape[-1] == 6 and len(outputs) == 1:
        # Raw one-class YOLOX export, fixed strides 8/16/32.
        grids, strides = [], []
        for stride in (8, 16, 32):
            x, y = np.meshgrid(np.arange(416 // stride), np.arange(416 // stride))
            grids.append(np.stack([x, y], -1).reshape(-1, 2))
            strides.append(np.full((x.size, 1), stride))
        grid, step = np.concatenate(grids), np.concatenate(strides)
        if raw.shape[1] != len(grid):
            raise ValueError("Unexpected raw YOLOX grid")
        prediction = raw[0]
        scores = prediction[:, 4] * prediction[:, 5]
        selected = scores > threshold
        p, scores = prediction[selected], scores[selected]
        with np.errstate(over="raise", invalid="raise"):
            centers = (p[:, :2] + grid[selected]) * step[selected]
            wh = np.exp(p[:, 2:4]) * step[selected]
        boxes = np.concatenate([centers - wh / 2, centers + wh / 2], axis=1) / ratio
        if len(boxes):
            xywh = np.concatenate([boxes[:, :2], boxes[:, 2:] - boxes[:, :2]], axis=1)
            keep = np.asarray(cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), threshold, nms))
            boxes = boxes[keep.reshape(-1).astype(int)]
    else:
        raise ValueError("Unknown detector export schema; no guessed decoding")
    if (len(boxes) > 32 or not np.isfinite(boxes).all()
            or (len(boxes) and np.any(boxes[:, 2:] <= boxes[:, :2]))):
        raise ValueError("Invalid or excessive person boxes")
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


class StrictSession:
    """No ORT CPU graph fallback or runtime retry for a selected CoreML mode."""

    def __init__(self, path: Path, mode: str, hw: tuple[int, int], threads: int = 4,
                 ort_module=None):
        if mode not in MODES or not 1 <= threads <= 32:
            raise ValueError("Invalid execution mode/thread count")
        if ort_module is None:
            import onnxruntime as ort_module
        ort = ort_module
        if ort.__version__ != ORT_VERSION:
            raise ValueError(f"Benchmark pins onnxruntime=={ORT_VERSION}")
        ep = "CPUExecutionProvider" if mode == "cpu" else "CoreMLExecutionProvider"
        if ep not in ort.get_available_providers():
            raise UnsupportedProvider(f"{ep} unavailable")
        self.mode, self.ep, self.hw = mode, ep, hw
        self.session = None
        self.profile = tempfile.TemporaryDirectory(prefix="mocap-ort-")
        self.placement = None
        self.profiling_ended = False
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.enable_profiling = True
        options.profile_file_prefix = str(Path(self.profile.name) / "placement")
        options.log_severity_level = 3
        providers = [ep]
        self.provider_options = {}
        if mode != "cpu":
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
            self.provider_options = {"ModelFormat": "MLProgram", "MLComputeUnits": MODES[mode],
                                     "RequireStaticInputShapes": "0", "EnableOnSubgraphs": "0"}
            providers = [(ep, self.provider_options)]
        try:
            self.session = ort.InferenceSession(
                str(path), sess_options=options, providers=providers)
            self.session.disable_fallback()
            if ep not in self.session.get_providers():
                raise UnsupportedProvider("Requested provider not retained by runtime")
            ins = self.session.get_inputs()
            expected = (1, 3, *hw)
            if len(ins) != 1 or ins[0].type != "tensor(float)" or len(ins[0].shape) != 4:
                raise ValueError("Unsupported ONNX input schema")
            if any(isinstance(a, int) and a != b for a, b in zip(ins[0].shape, expected)):
                raise ValueError("ONNX input shape differs from candidate manifest")
            self.input_name = ins[0].name
            outs = self.session.get_outputs()
            self.output_names = [out.name for out in outs]
            self.input_schema = {"name": ins[0].name, "shape": ins[0].shape, "type": ins[0].type}
        except BaseException as primary:
            try:
                self.close()
            except BaseException as cleanup_error:
                primary.add_note(f"Session cleanup: {type(cleanup_error).__name__}")
            raise

    def run(self, image: np.ndarray) -> list[np.ndarray]:
        if self.session is None or image.shape != (1, 3, *self.hw) or image.dtype != np.float32:
            raise ValueError("Invalid session/input")
        return self.session.run(self.output_names, {self.input_name: image})

    def audit_probe(self, image: np.ndarray) -> list[np.ndarray]:
        """One real-input placement probe; end tracing BEFORE measured frames."""
        if self.placement is not None:
            raise ValueError("Placement probe already executed")
        outputs = self.run(image)
        path = self.session.end_profiling()
        self.profiling_ended = True
        events = json.loads(Path(path).read_text())
        counts = Counter(e.get("args", {}).get("provider") for e in events
                         if e.get("cat") == "Node" and e.get("args", {}).get("provider"))
        self.placement = {"observed_kernel_events_by_provider": dict(counts),
                          "scope": "single real-input placement probe outside pass timings",
                          "internal_coreml_hardware_verified": False}
        Path(path).unlink()
        if not counts or set(counts) != {self.ep}:
            raise UnsupportedProvider("Missing placement evidence or unexpected provider execution")
        return outputs

    def metadata(self) -> dict:
        return {"requested_mode": self.mode, "requested_provider": self.ep,
                "provider_options": self.provider_options, "placement": self.placement,
                "input_schema": self.input_schema, "output_names": self.output_names,
                "coreml_internal_cpu_is_permitted": self.mode != "cpu",
                "ane_execution_verified": False, "runtime_fallback_enabled": False}

    def close(self) -> None:
        session, self.session = self.session, None
        try:
            if session is not None and not self.profiling_ended:
                session.end_profiling()
        finally:
            del session
            self.profile.cleanup()


class WholebodyONNX:
    def __init__(self, detector: Path, pose: Path, mode: str, hw=(256, 192), threads=4,
                 session_factory=StrictSession):
        self.detector = self.pose = None
        self.hw = hw
        try:
            self.detector = session_factory(detector, "cpu", (416, 416), threads)
            self.pose = session_factory(pose, mode, hw, threads)
        except BaseException as primary:
            try:
                self.close()
            except BaseException as cleanup_error:
                primary.add_note(f"Session cleanup: {type(cleanup_error).__name__}")
            raise

    def preflight(self, image: np.ndarray) -> dict:
        x, ratio = detector_input(image)
        decode_boxes(self.detector.audit_probe(x), ratio)
        # Full-frame ROI is a labelled PLACEMENT PROBE, never a substitute for
        # a missing person box and never included in benchmark observations.
        h, w = image.shape[:2]
        x, center, scale = pose_input(image, np.array([0, 0, w, h]), self.hw)
        decode_pose(self.pose.audit_probe(x), center, scale, self.hw)
        return {"detector": self.detector.metadata(), "pose": self.pose.metadata(),
                "pose_probe_roi": "full_frame_only_for_runtime_validation",
                "temporal_model_state": "stateless_per_frame"}

    def process(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        stages = {}
        start = time.perf_counter_ns()
        x, ratio = detector_input(image)
        prep = time.perf_counter_ns()
        outputs = self.detector.run(x)
        inferred = time.perf_counter_ns()
        boxes = decode_boxes(outputs, ratio)
        finished = time.perf_counter_ns()
        stages.update(detector_pre_ms=(prep-start)/1e6, detector_infer_ms=(inferred-prep)/1e6,
                      detector_post_ms=(finished-inferred)/1e6)
        xy, scores = [], []
        pre_ms = infer_ms = post_ms = 0.0
        for box in boxes:
            start = time.perf_counter_ns()
            x, center, scale = pose_input(image, box, self.hw)
            prep = time.perf_counter_ns()
            outputs = self.pose.run(x)
            inferred = time.perf_counter_ns()
            points, score = decode_pose(outputs, center, scale, self.hw)
            finished = time.perf_counter_ns()
            pre_ms += (prep-start)/1e6
            infer_ms += (inferred-prep)/1e6
            post_ms += (finished-inferred)/1e6
            xy.append(points)
            scores.append(score)
        stages.update(pose_pre_ms=pre_ms, pose_infer_ms=infer_ms, pose_post_ms=post_ms)
        return (boxes, np.asarray(xy, np.float32).reshape(-1, 133, 2),
                np.asarray(scores, np.float32).reshape(-1, 133), stages)

    def close(self) -> None:
        errors = []
        for name in ("pose", "detector"):
            owner = getattr(self, name)
            setattr(self, name, None)
            if owner is not None:
                try:
                    owner.close()
                except BaseException as exc:
                    errors.append(exc)
        if errors:
            raise RuntimeError("ONNX session cleanup failed") from errors[0]
