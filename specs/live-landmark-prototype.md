# Live landmark prototype

## Definition

Open the explicitly selected local camera and display current body, both hands
including fingers, face contours, and facial blendshape diagnostics in one live
window.

## Current contract

- Input is one explicitly selected camera index. The initial verified target is
  the MacBook built-in front camera at index `0`.
- Inference is MediaPipe Tasks 0.10.31 using three explicitly selected CPU
  tasks: Pose Landmarker Full, Hand Landmarker, and Face Landmarker. Each
  float16 v1 model asset is separately pinned by SHA-256.
- Task scheduling is explicitly selected as `serial` or `parallel`. Parallel
  scheduling owns one worker per independent landmarker and never retries in
  serial mode after a worker failure.
- Results may contain 33 pose landmarks, 21 landmarks for each visible hand,
  478 face landmarks, and face blendshape coefficients.
- Frame time authority is host monotonic receive time captured immediately after
  a successful camera read. It is not hardware capture time and must be labeled
  accordingly.
- Preview mirroring is display-only. Inference runs on the unmirrored camera
  frame.
- No raw frame is persisted. A metadata-only session manifest records the model,
  camera format, timing metrics, detection counts, and terminal status.
- Timing diagnostics separately report input conversion, pose inference, hand
  inference, face inference, result assembly, preview composition, and total
  host post-receive processing. These are host processing measurements; they do
  not claim camera sensor-to-display latency.
- Missing or checksum-invalid models, unavailable cameras, failed frame reads,
  and inference failures are terminal errors. No alternative model, camera,
  provider, or prerecorded input is selected automatically.

## Non-goals

- Metric or calibrated 3D reconstruction
- Camera/room position measurement or sweet-spot recommendations
- Bone local rotations, IK, retargeting, BVH, VRM, or VMC
- Multi-camera synchronization
- Multi-person association
- Recording images, video, landmarks, or biometric data
- Performance claims for CoreML, Windows ML, DirectML, or AMD GPUs

## Acceptance criteria

- All three model assets can be downloaded separately and their pinned SHA-256
  values verified.
- Self-check validates the model and initializes the landmarker without opening
  a camera.
- Camera index `0` opens without selecting another source and reports its actual
  delivered frame dimensions and FPS metadata.
- Headless camera verification still executes the full landmark drawing and
  diagnostics-panel composition path; it only omits window presentation.
- The live window clearly labels the mode as 2D and 3D/retargeting as inactive.
- When visible, pose, face, left-hand, and right-hand results are drawn from the
  current inference result; both hands show articulated 21-point connections.
- The diagnostics panel shows processing FPS, inference latency, detection
  status/counts, provider, per-task latency, and top facial blendshape scores.
- The metadata-only manifest records mean and maximum latency for every measured
  host-processing stage. Processing FPS is derived from first/last successful
  host frame-receive timestamps, not session initialization duration.
- Serial and parallel scheduling can be selected from the CLI and the active
  scheduling mode is visible in the provider label and session manifest.
- `Q` or `Esc` closes the camera, landmarker, and window cleanly.
- A metadata-only manifest is written on success, interruption, or a terminal
  runtime failure.

## Verification

```bash
uv run ruff check .
uv run pytest
uv run motioncapture-demo --self-check
uv run motioncapture-demo --camera-index 0 --headless-frames 30
uv run motioncapture-demo --camera-index 0
```

The final command requires visually checking the real MacBook camera path and
then quitting through the documented key control.

## Verified runtime decision

MediaPipe 1.0.1 was rejected on this Apple M5/macOS 27 host. Both the aggregate
Holistic graph and each separate task aborted during initialization inside
`DrishtiMetalHelper` with `graph_service.h:139 Service is unavailable` when the
CPU delegate was explicitly selected. The GPU delegate also failed to bind a
Metal tensor. These were native process aborts rather than recoverable Python
errors.

An isolated compatibility check confirmed that MediaPipe 0.10.31 initializes
the same official Pose, Hands, and Face task models successfully on this host.
The prototype therefore pins 0.10.31 and exposes the three-task CPU pipeline as
its only operating mode; it does not silently retry another provider or model.
