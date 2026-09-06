"""Native CPU vs local-browser hand backends; full Pose/Hands/Face on every frame."""
from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

import cv2

from motioncapture import recording_bench as baseline
from motioncapture.browser_hands import (
    BrowserHandError,
    BrowserHandLab,
    BrowserHandTask,
    collect_assets,
)
from motioncapture.hand_comparison import HandReference
from motioncapture.recording import RecordedDecoder, inspect_recording


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--browser-sdk", type=Path, default=Path(
        "tools/browser_hands/node_modules/@mediapipe/tasks-vision"))
    parser.add_argument("--include-web-cpu", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Explicit prefix smoke test; zero processes the entire file")
    parser.add_argument("--target-fps", type=float, default=60)
    parser.add_argument("--preview", choices=("display", "none"), default="display")
    parser.add_argument("--decode-threads", type=int, default=0)
    parser.add_argument("--warmup-frames", type=int, default=60)
    parser.add_argument("--browser-timeout", type=float, default=120)
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser


def run(args):
    if (args.max_frames < 0 or args.warmup_frames < 0 or not 0 <= args.decode_threads <= 64
            or not math.isfinite(args.target_fps) or args.target_fps <= 0
            or not math.isfinite(args.browser_timeout) or not 0 < args.browser_timeout <= 300):
        raise ValueError("Invalid benchmark limits")
    if args.output.exists():
        raise FileExistsError("Output exists; choose a new report filename")
    plan = (["native", "web_cpu", "web_gpu", "web_gpu", "web_cpu", "native"]
            if args.include_web_cpu else ["native", "web_gpu", "web_gpu", "native"])
    lab = None
    report = {"schema_version": 2, "status": "running", "stage": "preflight", "runs": [],
              "experiment": "hybrid_browser_hands", "source": None,
              "runtime": {"platform": platform.platform(), "python": platform.python_version(),
                          "mediapipe": baseline._installed("mediapipe"),
                          "opencv": cv2.__version__, "source": baseline._source_revision()},
              "configuration": {"plan": plan, "requested_prefix_frames": args.max_frames,
                                "target_fps": args.target_fps, "preview": args.preview,
                                "task_scheduling": "parallel", "resized": False,
                                "paced": False, "result_verification": True,
                                "frame_observer_overhead_in_loop_fps": True,
                                "browser_diagnostics": "sampled_1Hz_page_heartbeat",
                                "browser_timeout_s": args.browser_timeout},
              "scope": ("full three-task unpaced service including local browser transport; "
                        "not live"),
              "privacy": {"raw_frames_written": False, "landmarks_written": False,
                          "audio_processed": False, "network": "127.0.0.1 only",
                          "reference_hands": "bounded RAM only"},
              "performance_verdict": "pending_target_review", "accuracy_verified": False,
              "error": None}
    try:
        if baseline._installed("mediapipe") != "0.10.31":
            raise RuntimeError("Pinned MediaPipe 0.10.31 required")
        from motioncapture.landmarkers import MediaPipeLandmarkTracker
        from motioncapture.model_assets import MODEL_ASSETS, require_models
        import mediapipe as mp

        models = require_models(args.model_dir)
        report["models"] = [{"key": x.key, "sha256": x.sha256} for x in MODEL_ASSETS]
        web = Path(__file__).resolve().parents[2] / "tools/browser_hands"
        assets, manifest = collect_assets(args.browser_sdk, models["hands"], web)
        report["browser_assets"] = manifest
        report["stage"] = "probe"
        probe = inspect_recording(args.input, args.ffprobe)
        report["source"] = probe.summary()
        n = min(args.max_frames, len(probe.pts)) if args.max_frames else len(probe.pts)
        reference = HandReference(probe.pts[:n], probe.width, probe.height)
        report["configuration"]["scope"] = "full_file" if n == len(probe.pts) else "explicit_prefix"
        with BrowserHandLab(assets, timeout=args.browser_timeout) as lab:
            # URL is an ephemeral local capability, intentionally NOT in the JSON report.
            print("Open this local URL in Chrome/Edge and keep the tab visible:", flush=True)
            print(lab.url, flush=True)
            report["stage"] = "browser_connect"
            lab.wait_ready()
            report["stage"] = "gpu_preflight"
            # Fail unavailable GPU/image paths before the expensive native baseline pass.
            test = BrowserHandTask(lab, "GPU")
            try:
                with RecordedDecoder(args.input, probe, threads=args.decode_threads) as decoder:
                    frame = next(iter(decoder))
                    image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                     data=cv2.cvtColor(frame.image_bgr, cv2.COLOR_BGR2RGB))
                    test.detect_for_video(image, 0)
            finally:
                test.close()
            report["gpu_preflight"] = {"frames": 1, "outside_pass_timings": True,
                                       **test.summary(1000 / args.target_fps)}
            report["stage"] = "passes"
            for pass_index, backend in enumerate(plan):
                browser_tasks = []
                comparison = reference.comparator() if pass_index else None
                hidden_start = lab.hidden_events

                def make_tracker(
                    model_dir, scheduling, hand_backend=backend, task_records=browser_tasks,
                ):
                    def make_hand(path):
                        if path.resolve() != models["hands"].resolve():
                            raise ValueError("Unexpected hand model")
                        delegate = "GPU" if hand_backend == "web_gpu" else "CPU"
                        task = BrowserHandTask(lab, delegate)
                        task_records.append(task)
                        return task
                    return MediaPipeLandmarkTracker(model_dir, task_scheduling=scheduling,
                        hand_task_factory=make_hand if hand_backend != "native" else None)

                def observe(
                    frame, result, pair=comparison, current_pass=pass_index, current_backend=backend,
                ):
                    if pair is None:
                        reference.record(frame, result)
                    else:
                        pair.observe(frame, result)
                    if frame.identity.sequence % 300 == 0:
                        print(json.dumps({"pass": current_pass + 1, "backend": current_backend,
                                          "frame": frame.identity.sequence, "total": n}), flush=True)

                pass_args = argparse.Namespace(
                    **vars(args), mode="track", task_scheduling="parallel",
                    verify_results=True, verify_pixels=False,
                )
                begin = time.perf_counter_ns()
                result = {}
                try:
                    result = baseline.run_pass(pass_args, probe, tracker_factory=make_tracker,
                        result_observer=observe, frame_limit=args.max_frames,
                        failure_record=result)
                except BaseException:
                    result["status"] = "failed"
                    result["performance_comparable"] = False
                    raise
                finally:
                    # A failed fourth pass must not disappear from the final JSON.
                    result["backend"] = backend
                    result["pass_index"] = pass_index + 1
                    result["pass_wall_including_setup_cleanup_hash_s"] = (
                        time.perf_counter_ns() - begin
                    ) / 1e9
                    result["hand_comparison"] = (
                        comparison.summary(allow_partial=True) if comparison
                        else {"kind": "reference", "frames_recorded": reference.count}
                    )
                    result["hand_backend"] = (
                        browser_tasks[0].summary(1000 / args.target_fps) if browser_tasks
                        else {"delegate": "native CPU"} if backend == "native"
                        else {"delegate_requested": "GPU" if backend == "web_gpu" else "CPU",
                              "initialization_complete": False, "actual_delegate": None,
                              "model_cleanup_acknowledged": False}
                    )
                    result["tab_hidden_events"] = lab.hidden_events - hidden_start
                    result["performance_comparable"] = (
                        result.get("status") == "completed"
                        and result["tab_hidden_events"] == 0 and lab.visible
                    )
                    report["runs"].append(result)
                print(json.dumps({"pass": pass_index + 1, "backend": backend,
                                  "fps": result["unpaced_loop_fps"]}), flush=True)
            natives = [r["predictions_sha256"] for r in report["runs"]
                       if r["backend"] == "native"]
            report["native_reference_repeat_equal"] = len(natives) >= 2 and len(set(natives)) == 1
            report["performance_comparable"] = all(r["performance_comparable"] for r in report["runs"])
        report["status"], report["stage"] = "completed", "finished"
        report["browser_diagnostics"] = lab.snapshot()
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["performance_verdict"] = "failed_incomplete_experiment"
        report["performance_comparable"] = False
        report["error"] = {"type": type(exc).__name__, "stage": report["stage"]}
        if lab is not None:
            report["browser_failure"] = lab.failure_snapshot or lab.snapshot()
        cause = exc
        while cause is not None:
            if isinstance(cause, BrowserHandError):
                report["error"]["code"] = str(cause)
                break
            cause = cause.__cause__
        baseline.write_report(args.output, report)
        print(json.dumps({"status": report["status"], "error": report["error"]}), flush=True)
        return 130 if isinstance(exc, KeyboardInterrupt) else 2
    baseline.write_report(args.output, report)
    print(json.dumps({"status": "completed", "passes": len(report["runs"])}), flush=True)
    return 0


def main():
    return run(_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
