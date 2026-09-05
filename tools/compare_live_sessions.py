"""Compare two metadata-only sessions without treating detection totals as accuracy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def get(data, *keys):
    for key in keys:
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    return data


def fmt(value):
    return "unknown" if value is None else f"{value:.3f}"


def accounting(data):
    capture = data.get("capture") or {}
    names = ("captured", "delivered", "replaced", "discarded", "pending")
    if not all(isinstance(capture.get(name), int) for name in names):
        return "unknown"
    return "ok" if capture["captured"] == sum(capture[name] for name in names[1:]) else "INVALID"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())
    print("metric | baseline | candidate")
    print("--- | ---: | ---:")
    for label, path in (
        ("capture FPS", ("capture", "mean_capture_fps")),
        ("processing FPS (receive-span)", ("mean_processing_fps",)),
        ("capture queue mean ms", ("latency", "capture_queue", "mean_ms")),
        ("tracker mean ms", ("latency", "stages", "tracker_total", "mean_ms")),
        ("inference wall mean ms", ("latency", "stages", "inference_wall", "mean_ms")),
        ("preview mean ms", ("latency", "stages", "preview_composition", "mean_ms")),
        ("receive-to-preview mean ms", ("latency", "stages", "host_post_receive_total", "mean_ms")),
        ("receive-to-preview maximum ms", ("latency", "stages", "host_post_receive_total", "maximum_ms")),
        ("event pump mean ms", ("performance", "additional_stages", "event_pump", "mean_ms")),
        ("result residence mean ms", ("performance", "additional_stages", "result_residence", "mean_ms")),
        ("receive-to-preview p95 upper ms", ("performance", "stage_distributions", "host_post_receive_total", "p95_upper_ms")),
    ):
        print(f"{label} | {fmt(get(baseline, *path))} | {fmt(get(candidate, *path))}")
    warnings = []
    for name, data in (("baseline", baseline), ("candidate", candidate)):
        capture = data.get("capture") or {}
        print(f"\n{name}: status={data.get('terminal_status')}, capture_accounting={accounting(data)}, cleanup={capture.get('cleanup_complete')}")
        count = capture.get("captured")
        replaced = capture.get("replaced")
        if count and replaced is not None:
            print(f"capture replacement fraction: {100 * replaced / count:.3f}%")
        if data.get("terminal_status") != "completed" or not capture.get("cleanup_complete"):
            warnings.append(f"{name}: failed/incomplete run is not a performance success")
        if accounting(data) == "INVALID":
            warnings.append(f"{name}: capture accounting is invalid")
    for path in (
        ("models",), ("runtime", "platform"), ("runtime", "inference_provider"),
        ("runtime", "opencv"), ("runtime", "mediapipe"), ("camera_observation",),
        ("mode", "task_scheduling"), ("mode", "headless"),
    ):
        left, right = get(baseline, *path), get(candidate, *path)
        if left is None or right is None:
            warnings.append(f"Comparison metadata unknown: {'.'.join(path)}")
        elif left != right:
            warnings.append(f"Comparison settings differ: {'.'.join(path)}")
    print("\nNo automatic accuracy or sensor-to-display latency claim. Live gestures, visibility, temperature and framing must be controlled separately.")
    for warning in warnings:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
