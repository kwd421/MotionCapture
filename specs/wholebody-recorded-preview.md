# Whole-body recorded preview

Definition: replay an explicitly selected file through the existing same-frame
batch-2 DWPose pipeline and display each corresponding 2D observation.

## Contract and owners

- `RecordedDecoder` owns decoding and original PTS/identity validation.
- `SourcePacer` releases at original PTS from a fixed epoch. No skips or rebases.
- `BatchStagePipeline` owns the existing every-frame detector, OpenCV input
  normalization, ready-only handoff and same-frame batch-2 pose execution.
- `run_pass` retains inference verification, hashes and cleanup. An optional
  caller-thread consumer receives every validated packet; consumer failures are
  terminal and retain partial inference counts separately from display counts.
- `wholebody_recorded_preview` owns an OpenCV window and session report. Rendering
  uses a separate resized image, valid in-frame points, body/hand connections,
  and colored foot/face points. Detector slots are not persistent actor IDs.
- The inference loop runs on one worker and passes validated packets through a
  one-slot FIFO to the main-thread window. Slow display applies backpressure;
  it never overwrites a packet. One packet may be in the UI and one waiting in
  the queue, in addition to the existing bounded inference work. Cancellation
  unblocks enqueue and joins the inference owner before window cleanup.
- CoreML ALL with CPU partitions requires explicit flags; no provider retry.
- File mode, 2D scope, active provider, PTS, source rate, UI submission rate,
  point validity counts and result age are visible. UI-return age is measured
  separately from inference-validation age and never called photon latency.
- Escape/Q/window close interrupts the run. Model/decoder/UI failure cannot be
  reported as completed. Existing reports cannot be overwritten.
- Images and coordinates are transient by default; no video, audio or pose recording.
  Explicit `--snapshot-frame N` saves that frame's annotated preview next to the
  report, reports this privacy choice and measures its disk cost separately.

## Non-goals

Live capture, fixed-60 input emulation, actor association, calibrated 3D,
retargeting, external output, facial blendshapes, tracking accuracy certification.

## Acceptance and verification

Both local benchmark videos must finish with decoded = validated = UI-submitted
frames, preserved source timestamps, zero intentional skips and released owners.
Inspect the real window during both runs. Record inference age, UI-return age,
output intervals, rendering cost and process CPU/memory. Slower-than-source
processing remains an observed limitation, never a success implied by completion.
Exercise interruption/consumer failure and ensure cleanup and partial reports.

```sh
uv run --with-requirements tools/requirements-wholebody.txt \
  python -m motioncapture.wholebody_recorded_preview INPUT.mp4 \
  --provider coreml-all --allow-cpu-partitions --research-only \
  --output sessions/UNIQUE-preview.json
```

## Explicit display experiment (2026-09-08)

Compare the existing `opencv` display with explicitly selected `sdl` via
pygame-ce. The same composed BGR canvas, source frames, model outputs, source
PTS, one-slot FIFO, verification hashes and every-frame submissions are retained.
SDL owns only the main-thread window, synchronous canvas blit, event pump and
shutdown. It must reject headless/dummy display drivers. There is no automatic
backend substitution. Missing pygame-ce is an explicit setup failure.
The report records backend/library/driver, measured submit-and-event-pump cost,
OpenCV thread budget and unchanged physical-display verification limitation.
No intentional delay, frame replacement, resolution change or inference omission
is part of this experiment. Vsync and photon timing are not asserted.

Acceptance: run the same phone prefix in A/B/B/A order, inspect exact inference
hash agreement, frame coverage and cleanup; then run any promising candidate on
both full local videos. Exercise the real SDL pixel transfer and quit event.
Speed benefit remains unproven until measured; default stays OpenCV during trials.

`--opencv-threads 1..64` explicitly sets the process-wide OpenCV CPU budget before
workers start; omission preserves the runtime default. Requested and observed
values are recorded. Schema 2 renames the OpenCV-specific UI cost field to
`display_submit_and_event_pump_ms` so the same measurement applies to both backends.

An isolated 1.29.0 runtime candidate uses the unchanged original ONNX assets.
`--expected-ort-version` defaults to 1.22.1; a mismatched environment is terminal
before source/model access. Both supported research versions are exact matches
from the shared adapter allowlist and are recorded in native session metadata.
The actual selected version is shown in the preview. No automatic upgrade.
