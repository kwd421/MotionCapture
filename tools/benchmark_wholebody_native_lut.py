"""Synthetic ROI preprocessing only; no detector, neural inference or user footage."""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

import cv2
import numpy as np

from motioncapture.wholebody_fast_input import fast_pose_tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--rounds", type=int, default=6)
    args = parser.parse_args()
    if args.iterations < 1 or args.rounds < 2:
        raise ValueError("Positive iterations and at least two rounds required")
    cv2.setNumThreads(4)
    rng = np.random.default_rng(621)
    image = rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    boxes = [np.array([i*7.25-25, i*3.125-10, 500+i*11.75, 890+i*2.5], np.float32)
             for i in range(32)]
    matches = 0
    for box in boxes:
        a = fast_pose_tensor(image, box, (192, 256), kernel="numpy")
        b = fast_pose_tensor(image, box, (192, 256), kernel="opencv")
        if not all(x.dtype == y.dtype and x.tobytes() == y.tobytes()
                   for x, y in zip(a, b, strict=True)):
            raise AssertionError("Input tensors differ")
        matches += 1
    timings = {"numpy": [], "opencv": []}
    for repeat in range(args.rounds):
        for kernel in (["numpy", "opencv"] if repeat % 2 == 0 else ["opencv", "numpy"]):
            started = time.perf_counter_ns()
            for i in range(args.iterations):
                fast_pose_tensor(image, boxes[i % len(boxes)], (192, 256), kernel=kernel)
            timings[kernel].append((time.perf_counter_ns()-started)/1e6/args.iterations)
    medians = {k: statistics.median(v) for k, v in timings.items()}
    print(json.dumps({
        "scope": "synthetic ROI affine+lookup+NCHW only; no native model or M5 claim",
        "platform": platform.platform(), "python": platform.python_version(),
        "opencv": cv2.__version__, "numpy": np.__version__,
        "opencv_threads": cv2.getNumThreads(), "exact_roi_comparisons": matches,
        "rounds": args.rounds, "iterations_per_round": args.iterations,
        "round_mean_ms": timings, "median_round_mean_ms": medians,
        "median_time_reduction_percent": 100*(1-medians["opencv"]/medians["numpy"]),
        "native_inference_measured": False,
    }, indent=2))


if __name__ == "__main__":
    main()
