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

## Explicit operating mode

The prototype explicitly selects MediaPipe 0.10.31 CPU inference with three
pinned official model assets. It never silently changes models, camera sources,
or inference providers. A missing model, unavailable camera, or inference error
terminates with an explicit error.

Pose, hand, and face tasks run in the explicitly reported `parallel` scheduling
mode by default. Use `--task-scheduling serial` only when intentionally running
the measured serial comparison path; a parallel failure never falls back to it.

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
