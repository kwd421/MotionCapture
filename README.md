# MotionCapture prototype

The first vertical slice is a local MacBook camera demo. It displays real-time
MediaPipe Pose, Hands, and Face task landmarks plus facial blendshape scores.

This prototype is explicitly **2D/monocular landmark inference**. It does not
perform calibrated 3D reconstruction, bone rotation solving, retargeting, room
measurement, or VMC export.

## Run

```bash
uv sync
uv run motioncapture-models fetch
uv run motioncapture-avatars fetch
uv run motioncapture-demo --self-check
uv run motioncapture-demo --camera-index 0
```

Press `Q` or `Esc` to stop. The app writes a metadata-only session manifest to
`sessions/`; it does not record camera frames.

## Capture/runtime boundary

The camera now has one owning thread, independent of inference and preview.
Delivery is explicitly **latest-only, one pending frame**: a slow consumer skips
older unprocessed frames instead of accumulating latency. This does not control
the camera driver's internal buffering and is not multi-camera synchronization.

The preview shows host-observed capture FPS, replacement count and capture-queue
residence time. Schema-v2 session manifests distinguish `capture.captured`,
`capture.delivered`, `capture.replaced`, `capture.discarded`, `capture.pending`
and successfully processed `frames`. They also retain stream/frame identity,
original host receive time, capture cleanup status and queue timing. These are
**not sensor-to-display latency or sensor frame-drop measurements**.

`--headless-frames 120` processes exactly 120 selected real frames, including
preview composition but without window presentation. Capture may read more than
120 frames. `--capture-timeout 10` sets the maximum wait per camera startup,
frame read and shutdown; increase it explicitly for an initial permission prompt.

```bash
uv run ruff check .
uv run pytest
uv run motioncapture-demo --camera-index 0 --headless-frames 120
uv run motioncapture-demo --camera-index 0 --no-mirror
```

A blocked native camera read cannot be killed as a Python thread. A shutdown
timeout is reported as failed with cleanup incomplete; it is not silently
restarted. Native inference crash isolation is not implemented in this slice.
Real Mac/Windows camera tests and before/after performance measurements remain
pending. See [capture/runtime contract](specs/capture-runtime-boundary.md).

## Explicit operating mode

The prototype explicitly selects MediaPipe 0.10.31 CPU inference with three
pinned official model assets. It never silently changes models, camera sources,
or inference providers. A missing model, unavailable camera, or inference error
terminates with an explicit error.

Pose, hand, and face tasks run in the explicitly reported `parallel` scheduling
mode by default. Use `--task-scheduling serial` only when intentionally running
the measured serial comparison path; a parallel failure never falls back to it.
Capture threading is separate from this inference-task scheduling choice.

Model outputs are copied to owned numeric records. Empty detections stay
`missing`; malformed coordinates, scores and topology fail explicitly. The
original host timestamp survives conversion to the model's strictly increasing
millisecond time. A new stream requires a fresh tracker lifecycle.

## Local test avatars

`motioncapture-avatars fetch` installs and verifies two pinned official sample
avatars under `assets/avatars/`:

- Live2D Cubism `Haru`, with eye-blink and lip-sync parameter groups.
- VRM 1.0 `Seed-san`, with 51 humanoid bones including articulated fingers and
  standard face, gaze, and viseme expression presets.

The downloaded third-party files are intentionally git-ignored and retain their
own license terms. Run `uv run motioncapture-avatars verify` to check the local
files without downloading or replacing anything. These assets are ready for a
future renderer/retargeter, but the current landmark overlay does not drive them.
