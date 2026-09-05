"""Synthetic OpenCV camera+overlay microbenchmark; NOT a mocap/FPS benchmark.

The reference copies the baseline's draw/copy/flip/resize order. The display
kernel uses the production preview topology/drawing code. Both use the SAME
synthetic points/edges. Panel text, MediaPipe, camera IO and HighGUI are excluded.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

import cv2
import numpy as np

from motioncapture.contracts import Landmark
from motioncapture.preview import _Topology, _draw


def fixture(count, seed):
    rng = np.random.default_rng(seed)
    return tuple(Landmark(float(x), float(y), 0) for x, y in rng.uniform(0.15, 0.85, (count, 2)))


def reference_draw(image, points, edges, color, thickness, circles):
    height, width = image.shape[:2]

    def visible(point):
        presence = getattr(point, "presence", None)
        visibility = getattr(point, "visibility", None)
        return (presence is None or presence >= 0.2) and (visibility is None or visibility >= 0.2)

    def pixel(point):
        return round(point.x * width), round(point.y * height)

    for start, end in edges:
        if visible(points[start]) and visible(points[end]):
            cv2.line(image, pixel(points[start]), pixel(points[end]), color, thickness)
    if circles:
        for point in points:
            if visible(point):
                cv2.circle(image, pixel(point), 2, color, -1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--rounds", type=int, default=6)
    args = parser.parse_args()
    if args.iterations < 1 or args.rounds < 1:
        raise ValueError("iterations and rounds must be positive")
    source = np.random.default_rng(11).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    groups = []
    for count, used, color, thickness, circles in (
        (33, 33, (68, 231, 255), 3, True),
        (21, 21, (108, 255, 128), 3, True),
        (21, 21, (255, 155, 82), 3, True),
        (478, 124, (255, 120, 214), 1, False),
    ):
        edges = tuple((i, i + 1) for i in range(used - 1))
        groups.append((fixture(count, count), edges, color, thickness, circles,
                       _Topology.create(edges, count, circles)))

    def original():
        annotated = source.copy()
        for points, edges, color, thickness, circles, _ in groups:
            reference_draw(annotated, points, edges, color, thickness, circles)
        annotated = cv2.flip(annotated, 1)
        return cv2.resize(annotated, (960, 540), interpolation=cv2.INTER_AREA)

    def display():
        output = np.empty((540, 1360, 3), dtype=np.uint8)
        image = output[:, :960]
        cv2.resize(source, (960, 540), dst=image, interpolation=cv2.INTER_AREA)
        for points, _, color, thickness, _, topology in groups:
            _draw(image, points, topology, color, thickness)
        cv2.flip(image, 1, dst=image)
        return output

    for _ in range(40):
        original()
        display()
    results = {"baseline_kernel": [], "display_kernel": []}
    for repeat in range(args.rounds):
        order = [("baseline_kernel", original), ("display_kernel", display)]
        if repeat % 2:
            order.reverse()
        for name, method in order:
            started = time.perf_counter_ns()
            for _ in range(args.iterations):
                method()
            results[name].append((time.perf_counter_ns() - started) / 1e6 / args.iterations)
    medians = {name: statistics.median(values) for name, values in results.items()}
    print(json.dumps({
        "scope": "synthetic camera+overlay kernel only; excludes panel, camera IO, MediaPipe, GUI",
        "source_size": [1280, 720], "preview_size": [960, 540],
        "platform": platform.platform(), "python": platform.python_version(),
        "opencv": cv2.__version__, "numpy": np.__version__,
        "opencv_threads": cv2.getNumThreads(), "rounds": args.rounds,
        "iterations_per_round": args.iterations, "round_mean_ms": results,
        "median_round_mean_ms": medians,
        "median_reduction_percent": 100 * (
            1 - medians["display_kernel"] / medians["baseline_kernel"]
        ),
        "native_tracking_tested": False, "hardware_camera_tested": False,
    }, indent=2))


if __name__ == "__main__":
    main()
