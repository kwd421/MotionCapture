# Capture/runtime boundary

## Definition and status

Keep the real one-person landmark demo while separating continuous camera reads
from inference, with explicit latest-frame delivery and owned numeric results.

Implemented on `feat/capture-runtime-boundary`, based on
`9a2608ab2e10144f5b0ee78dfc75d777b52a3fa8`. This contract owns the new runtime and
observation boundary; the existing live-landmark specification continues to own
model selection, inference task scheduling, display-only mirroring and non-goals.
Its original frame-loop scheduling is superseded here, not its product scope.

Automated contract/runtime/session tests were run in the implementation container
using Python 3.13.5 on Linux. The project remains pinned to Python 3.12 and the
existing dependencies. The full pinned environment, MediaPipe model execution,
real camera input, Windows/macOS behavior and performance are **not verified** by
those unit tests. No quality or FPS improvement is claimed yet.

## Current contract

- `LocalCamera` remains the explicit AVFoundation/Media Foundation adapter.
- `CaptureRuntime` owns open/read/close on one producer thread. The caller owns
  the MediaPipe tracker and preview. There is one producer and one consumer.
- A `threading.Condition` protects a single pending frame, lifecycle, counters
  and terminal error. The runtime never creates an unbounded video queue.
- Replacing a pending frame increments `replaced`. Frames discarded on stop or
  failure increment `discarded`. `delivered` counts dequeue, not successful
  inference. The manifest's `frames` counts completed inference/preview cycles.
- At every snapshot:
  `captured == delivered + replaced + discarded + pending`.
  Only validated successful reads enter `captured`; sensor loss remains unknown.
- `source_id` is a session-local backend/index identifier, **not** a persistent
  hardware serial number. Each runtime has a fresh `stream_id`.
- `FrameIdentity` retains the original sequence and host-monotonic receive time.
  Strictly increasing sequence and time are checked; no synthetic camera time is
  created. `CapturedFrame` omits image pixels from its representation.
- The inference adapter owns conversion to model milliseconds relative to the
  first frame of the stream. Duplicate/non-increasing millisecond times or a
  different stream fail rather than silently nudging time or reusing state.
- `LandmarkResult` contains numeric immutable landmark/blendshape records, not
  framework-owned result objects. Model initialization/processing errors remain
  `InferenceError`; malformed results are not converted to missing detections.
- The selected topology remains pose 33, each hand 21 and face 478. Zero results
  mean missing detections. This is not a general multi-actor skeleton schema.
- A zero blendshape score is valid. Missing score, nonfinite data, missing/duplicate
  blendshape name or unexpected landmark count is invalid and fails the slice.
- Image x/y are normalized; image z is model-relative. Pose world coordinates
  remain estimated hip-relative meters, hand world coordinates estimated
  hand-relative meters. None of these is calibrated studio-space 3D.
- Preview composition uses the exact image associated with the returned result,
  never a newer camera image overlaid with old landmarks. Mirroring is display-only.
- `Q`/`Esc`, Ctrl-C and terminal errors request capture stop; shutdown is bounded.
  Waiting readers are awakened on stop/failure. Failure takes precedence over a
  pending frame. An already selected frame is not republished after a noticed
  capture failure.
- No frames, landmarks or facial coefficients are persisted by this slice.

## Ownership and failures

`contracts.py` owns the numeric boundary. `runtime.py` owns camera lifecycle and
slot accounting. `landmarkers.py` owns model instances, model timestamp mapping,
framework conversion and parallel/serial task scheduling. `app.py` owns ordering,
preview and final cleanup. `session.py` owns the metadata manifest.

Runtime states: `new -> starting -> running -> stopping -> stopped`, with
terminal `failed` reachable from any acquired lifecycle phase. Opening a runtime
twice is invalid; a new session constructs new owners. A failure cannot become a
successful stop merely because later cleanup completes. The first failure is
preserved; later cleanup failures are attached to it.

The producer uses a daemon thread only to avoid preventing interpreter exit when
a native camera call is wedged. Normal stop requests still join the owner. If
join times out, state is failed and `cleanup_complete` is false. Python cannot
safely kill that call; there is no claim of clean release, automatic retry or
process isolation. Capture/GUI subprocess isolation is a separate future slice.

An inference failure does not switch provider/model or capture mode. A manifest
write failure on an otherwise successful path is an error. When another error is
already in flight, the write failure is also reported without hiding the primary
failure.

## Session schema 2 and timing

Existing diagnostic fields remain; additions are `capture`, first/last processed
frame identities, delivery mode and capture-queue latency. Unobserved capture or
queue timing is null, not zero. Dependency versions can be read without importing
or initializing the native model runtime.

`capture.mean_capture_fps` is calculated from host receive times of successful
reads. `mean_processing_fps` remains the rate of selected processed-frame receive
times. Neither is camera exposure rate nor monitor refresh rate.

`capture_queue` measures receive to dequeue using `monotonic_ns`.
`host_post_receive_total` now explicitly includes this queue time through preview
composition, but still excludes the camera's hidden buffering, sensor exposure,
window presentation and monitor scanout. Model/preview compute durations use
`perf_counter_ns`. Do not directly label any of these motion-to-photon latency.

## Non-goals

No new model or provider, GPU conversion, GUI framework, native crash isolation,
recording/replay, IK/retargeting, VMC/VTS output, phone app, synchronization,
calibration, multiple cameras or multiple people. No new dependencies or lockfile
changes. The existing `main` and unrelated code/assets are not changed.

## Acceptance and user verification

The deterministic tests use controlled camera events, not sleeps, to exercise
slow consumers, replacement accounting, errors, startup failure, monotonicity,
stop during an in-flight read, waiting readers and stream changes. Actual timeout
behavior has dedicated tests. Numeric tests cover missing vs zero, invalid values,
owned copies and topology. Session tests cover provenance, unknown state and I/O
failure. Test doubles exist only under `tests/`.

On the target machine:

```bash
uv sync
uv run ruff check .
uv run pytest
uv run motioncapture-demo --self-check
uv run motioncapture-demo --camera-index 0 --headless-frames 120
uv run motioncapture-demo --camera-index 0
uv run motioncapture-demo --camera-index 0 --no-mirror
uv run motioncapture-demo --camera-index 0 --task-scheduling serial --headless-frames 120
```

Explicitly fetch models first only if they are not already installed. For the
initial OS permission dialog, `--capture-timeout 30` may be selected deliberately.

Check correct body/left-hand/right-hand/face overlays; no stale overlay on a newer
image; successful `Q`/`Esc` and Ctrl-C; and camera release followed by a new run.
Check invalid index/permission denial produce errors without another camera.
On a removable webcam, check removal stops with an error rather than old frames.
The final manifest should have pending zero and complete cleanup on a normal run.
`frames` must be 120 in the headless run; `captured` may be larger. If a shutdown
times out, treat it as a failure and restart the process before another test.

Compare against `main` using the same camera, dimensions, task-scheduling mode,
lighting and pose. Record capture FPS, processing FPS, replacements, queue age,
compute durations, OS/version and thermal conditions. Runtime correctness and a
speed improvement are separate outcomes; a higher capture FPS is not proof of
higher inference FPS or lower sensor-to-display latency.
