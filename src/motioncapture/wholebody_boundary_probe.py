"""Counterfactual input/SimCC diagnostics, deliberately NOT an FPS benchmark.

Select a bounded set of error frames from a previous report, decode sequentially,
and keep all pixel/tensor/landmark values in RAM. The output stores hashes,
aggregate deltas and source frame identifiers only. No numerical "fix" is applied.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from motioncapture.recording import RecordedDecoder
from motioncapture.wholebody_catalog import BenchmarkError
from motioncapture.wholebody_fast_input import array_hash, crop_geometry, normalize
from motioncapture.wholebody_onnx import (
    PARTS,
    decode_people,
    decode_pose,
    detector_tensor,
    pose_tensor,
)


def select_frames(path: Path, probe, limit: int) -> tuple[list[int], dict]:
    if path.stat().st_size > 20 * 1024 * 1024:
        raise BenchmarkError("diagnostic_report_too_large")
    data = json.loads(path.read_text())
    if data.get("source", {}).get("sha256") != probe.sha256:
        raise BenchmarkError("diagnostic_source_hash_mismatch")
    errors = {}
    for run in data.get("runs", []):
        for item in run.get("provider_disagreement", {}).get("worst_frames", []):
            sequence, pts = item.get("sequence"), item.get("pts")
            delta = item.get("max_pixel_disagreement")
            if (type(sequence) is not int or not 0 <= sequence < len(probe.pts)
                    or type(pts) is not int or pts != probe.pts[sequence]
                    or not isinstance(delta, (int, float)) or not np.isfinite(delta) or delta < 0):
                raise BenchmarkError("invalid_diagnostic_frame_reference")
            if sequence < limit and delta > 0:
                errors[sequence] = max(errors.get(sequence, 0.), float(delta))
    ranked = sorted(errors, key=errors.get, reverse=True)[:8]
    if not ranked:
        raise BenchmarkError("no_positive_disagreement_frames_in_prefix")
    selected = sorted({neighbor for index in ranked for neighbor in (index-1, index, index+1)
                       if 0 <= neighbor < limit})
    return selected, {"selection": "top_8_positive_errors_plus_immediate_neighbors",
                      "selected_frames": selected, "maximum_selected_frames": 24,
                      "report_source_revision": data.get("runtime", {}).get("source")}


def delta(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        raise BenchmarkError("diagnostic_tensor_schema_mismatch")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise BenchmarkError("diagnostic_nonfinite_tensor")
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {"shape": list(a.shape), "reference_sha256": array_hash(a),
            "candidate_sha256": array_hash(b), "byte_equal": a.tobytes() == b.tobytes(),
            "elements": int(a.size), "changed_elements": int(np.count_nonzero(d)),
            "mean_abs": float(d.mean()), "max_abs": float(d.max())}


def axes(outputs, size):
    width, height = size
    results = []
    for length in (width*2, height*2):
        matches = [x for x in outputs if x.shape == (1, 133, length)]
        if len(matches) != 1 or not np.isfinite(matches[0]).all():
            raise BenchmarkError("diagnostic_simcc_schema_mismatch")
        results.append(matches[0])
    return results


def response_delta(a, b, size):
    result = {}
    for axis, reference, candidate in zip(("x", "y"), axes(a, size), axes(b, size), strict=True):
        top_a, top_b = reference.argmax(-1), candidate.argmax(-1)
        changed = top_a != top_b
        top2 = np.partition(reference, -2, axis=-1)[..., -2:]
        gaps = top2[..., 1] - top2[..., 0]
        result[axis] = {**delta(reference, candidate),
                        "argmax_changed_joints": int(changed.sum()),
                        "max_argmax_bin_delta": int(np.abs(top_a-top_b).max()),
                        "reference_top2_min_gap_at_changed_joints": (
                            float(gaps[changed].min()) if changed.any() else None)}
    return result


def point_delta(a, b):
    result = {}
    for name, (begin, end) in PARTS.items():
        both = a.valid[begin:end] & b.valid[begin:end]
        d = np.linalg.norm(a.xy[begin:end][both] - b.xy[begin:end][both], axis=-1)
        result[name] = {"compared_points": int(len(d)),
                        "max_source_pixels": float(d.max()) if len(d) else None,
                        "above_1px": int((d > 1).sum()), "above_10px": int((d > 10).sum()),
                        "validity_changes": int((a.valid[begin:end] != b.valid[begin:end]).sum())}
    return result


def observe_frame(frame, detector_a, detector_b, pose, size, threshold):
    source_before = array_hash(frame.image_bgr)
    image, ratio = detector_tensor(frame.image_bgr)
    input_before = array_hash(image)
    outputs_a = [x.copy() for x in detector_a.run(image)]
    outputs_b = [x.copy() for x in detector_b.run(image)]
    if array_hash(image) != input_before:
        raise BenchmarkError("detector_mutated_input")
    if len(outputs_a) != 1 or len(outputs_b) != 1:
        raise BenchmarkError("invalid_detector_outputs")
    boxes_a, boxes_b = decode_people(outputs_a[0], ratio), decode_people(outputs_b[0], ratio)
    result = {"sequence": frame.identity.sequence, "pts": frame.identity.pts,
              "time_base": str(frame.identity.time_base), "source_pixels_sha256": source_before,
              "detector_input_sha256": input_before,
              "detector_people": [len(boxes_a), len(boxes_b)],
              "detector_output": delta(outputs_a[0], outputs_b[0])}
    if len(boxes_a) != 1 or len(boxes_b) != 1:
        result["status"] = "unmatched_zero_or_multiple_people"
        return result
    crop_a, center_a, scale_a, matrix_a = crop_geometry(frame.image_bgr, boxes_a[0], size)
    crop_b, center_b, scale_b, matrix_b = crop_geometry(frame.image_bgr, boxes_b[0], size)
    tensor_a, tensor_b = normalize(crop_a), normalize(crop_b)
    # Validate the new fast implementation against the actual imported owner for
    # EVERY diagnostic crop. This catches local uncommitted recipe changes.
    for box, value, center, scale in ((boxes_a[0], tensor_a, center_a, scale_a),
                                      (boxes_b[0], tensor_b, center_b, scale_b)):
        originals = pose_tensor(frame.image_bgr, box, size)
        if any(a.dtype != b.dtype or a.tobytes() != b.tobytes()
               for a, b in zip(originals, (value, center, scale), strict=True)):
            raise BenchmarkError("fast_preprocessing_recipe_mismatch")
    def run(value):
        before = array_hash(value)
        output = [x.copy() for x in pose.run(value)]
        if array_hash(value) != before:
            raise BenchmarkError("pose_mutated_input")
        return output
    # A/B/B/A through ONE fixed pose session. Copy native output before the next call.
    aa, ab, bb, ba = run(tensor_a), run(tensor_b), run(tensor_b), run(tensor_a)
    pa = decode_pose(aa, size, center_a, scale_a, threshold)
    pb = decode_pose(ab, size, center_b, scale_b, threshold)
    geometry_only = decode_pose(aa, size, center_b, scale_b, threshold)
    if array_hash(frame.image_bgr) != source_before:
        raise BenchmarkError("diagnostic_source_pixels_mutated")
    result.update(status="compared", box_edges=delta(boxes_a, boxes_b),
                  affine=delta(matrix_a, matrix_b), crop_pixels=delta(crop_a, crop_b),
                  pose_input=delta(tensor_a, tensor_b), simcc=response_delta(aa, ab, size),
                  same_tensor_a_repeat=response_delta(aa, ba, size),
                  same_tensor_b_repeat=response_delta(ab, bb, size),
                  full_downstream=point_delta(pa, pb),
                  coordinate_transform_only=point_delta(pa, geometry_only),
                  pose_session_policy="one_fixed_session; input_A_B_B_A",
                  ground_truth_accuracy_verified=False)
    return result


def run_diagnostics(path, probe, report_path, limit, detector_a_factory, detector_b_factory,
                    pose_factory, size=(192, 256), threshold=.3):
    rows, owners = [], []
    result = {"status": "running", "kind": "input_boundary_counterfactual_not_performance",
              "frames": rows, "cleanup": [], "error": None,
              "privacy": "hashes_deltas_identifiers_only; pixels_and_coordinates_RAM_only"}
    phase = "select_frames"
    try:
        selected, selection = select_frames(report_path, probe, limit)
        result.update(selection)
        phase = "session_setup"
        for factory in (detector_a_factory, detector_b_factory, pose_factory):
            owners.append(factory())
        result["execution"] = [x.metadata for x in owners]
        phase = "decode_and_probe"
        with RecordedDecoder(path, probe) as decoder:
            for frame in decoder:
                if frame.identity.sequence in selected:
                    rows.append(observe_frame(frame, *owners, size, threshold))
                if frame.identity.sequence >= selected[-1]:
                    break
        if len(rows) != len(selected):
            raise BenchmarkError("incomplete_diagnostic_coverage")
        result["status"] = "completed"
    except BaseException as exc:
        result["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        result["error"] = {"phase": phase, "type": type(exc).__name__,
                           "code": getattr(exc, "code", None)}
    finally:
        for owner in reversed(owners):
            try:
                owner.close()
            except BaseException as exc:
                result["cleanup"].append({"status": "failed", "type": type(exc).__name__})
                if result["status"] == "completed":
                    result["status"] = "failed"
            else:
                result["cleanup"].append({"status": "owner_released"})
    return result
