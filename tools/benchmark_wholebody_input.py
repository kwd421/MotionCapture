"""Recorded pixels, synthetic ROI: preprocessing microbenchmark, never model FPS."""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from motioncapture.recording_bench import write_report
from motioncapture.wholebody_catalog import sha256
from motioncapture.wholebody_fast_input import fast_pose_tensor
from motioncapture.wholebody_onnx import pose_tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--frames", type=int, default=900)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.rounds <= 10 or not 1 <= args.frames <= 120000:
        raise ValueError("Invalid limits or existing output")
    input_hash = sha256(args.input)
    target = args.frames
    result = {"scope": "preprocessing_only; synthetic_fixed_ROI; NOT tracking inference",
              "source_sha256": input_hash, "selected_frames_per_round": target,
              "pts_used_for_model": False, "model_calls": 0,
              "roi": "fixed central source rectangle, NOT detector output",
              "runtime": {"platform": platform.platform(), "python": platform.python_version(),
                          "opencv": cv2.__version__, "numpy": np.__version__,
                          "opencv_threads": cv2.getNumThreads()},
              "actual_model_inference": False, "rounds": [], "byte_equal_samples": 0}
    for iteration in range(args.rounds):
        values = {"reference": [], "lookup": []}
        seen = 0
        cap = cv2.VideoCapture(str(args.input), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
        try:
            if not cap.isOpened():
                raise ValueError("FFmpeg decoder unavailable")
            for index in range(target):
                ok, image = cap.read()
                if not ok or image is None:
                    raise ValueError("Decoder stopped before requested prefix")
                height, width = image.shape[:2]
                box = np.array([width*.2, height*.015, width*.8, height*.985], np.float32)
                outputs = {}
                pairs = [("reference", pose_tensor), ("lookup", fast_pose_tensor)]
                if (index+iteration) % 2:
                    pairs.reverse()
                for name, method in pairs:
                    began = time.perf_counter_ns()
                    outputs[name] = method(image, box, (192, 256))
                    values[name].append((time.perf_counter_ns()-began)/1e6)
                if any(a.dtype != b.dtype or a.tobytes() != b.tobytes()
                       for a, b in zip(outputs["reference"], outputs["lookup"], strict=True)):
                    raise ValueError("Preprocessing equality failed")
                seen += 1
                result["byte_equal_samples"] += 1
                if seen == target:
                    break
        finally:
            cap.release()
        if seen != target:
            raise ValueError("Frame count mismatch")
        result["rounds"].append({k: {"mean_ms": statistics.mean(v),
                                    "p95_ms": float(np.quantile(v, .95))}
                                for k, v in values.items()})
    medians = {k: statistics.median(r[k]["mean_ms"] for r in result["rounds"])
               for k in ("reference", "lookup")}
    result["median_round_mean_ms"] = medians
    result["kernel_reduction_percent"] = 100*(1-medians["lookup"]/medians["reference"])
    if sha256(args.input) != input_hash:
        raise ValueError("Source changed during microbenchmark")
    write_report(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
