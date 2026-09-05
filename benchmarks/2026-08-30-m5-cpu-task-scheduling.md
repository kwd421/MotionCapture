# M5 CPU task-scheduling benchmark

## Purpose

Measure the verified serial Pose/Hands/Face bottleneck before changing task
scheduling, then verify the same pipeline with three independently owned worker
threads. This benchmark does not compare model accuracy.

## Environment and invariants

- Host: Apple M5, macOS, Python 3.12.10
- Runtime: MediaPipe 0.10.31 CPU/XNNPACK
- Models: pinned Pose Landmarker Full, Hand Landmarker, Face Landmarker assets
- Camera: AVFoundation index 0, observed 1280x720 at nominal 30 FPS
- Frames: 120 real camera frames per run
- Preview composition executed in headless mode; presentation alone was omitted
- No raw frames or landmarks were retained
- Pose, hands, face, drawing, and manifest stages remained enabled

## Result

| Metric | Serial | Parallel |
|---|---:|---:|
| Observed processing FPS | 7.85 | 14.88 |
| Tracker total mean | 96.75 ms | 36.37 ms |
| Inference wall mean | not separately recorded | 35.33 ms |
| Preview composition mean | 8.74 ms | 8.42 ms |
| Host post-receive total mean | 105.71 ms | 44.99 ms |
| Pose detections | 120/120 | 120/120 |
| Face detections | 120/120 | 120/120 |

Serial manifest: `sessions/20260830T083821Z-f80ea320.json`

Parallel manifest: `sessions/20260830T084013Z-7d885a59.json`

## Decision

Use parallel task scheduling as the prototype default. Preserve serial as an
explicit benchmark mode. Do not retry serial after a parallel worker failure.

The observed FPS remains below the nominal camera rate even though host
post-receive processing improved. Camera-read scheduling and capture/inference
decoupling are the next measured boundary; no sensor-to-display latency claim is
made. Hand-detection totals differed because the performer's visible hand pose
was not controlled, so they are not evidence of a quality change.
