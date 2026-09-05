# Live performance candidate: evidence and limits

## Source and baseline

Code baseline: `932e8e8d90a0b86629d4e8d35ea165ba87edc094` on
`feat/capture-runtime-boundary`. Original runtime, contracts, capture, session,
landmarkers, renderer, entry point and dependency lock are not modified.

The user supplied a schema-v2 metadata-only log, SHA-256
`37ff5202b1cf53a352171d125650bfc492bfd291d304f91dc73f172a44bb7e8a`. No original capture/face data
or device UUIDs are included in this bundle.

| Observed item | Value |
| --- | ---: |
| Total session duration | 135.32 s |
| Host-observed capture FPS | 29.9947 |
| Successfully processed FPS, receive-span | 28.8896 |
| Captured / delivered / replaced / discarded | 3996 / 3847 / 148 / 1 |
| Capture queue, mean / max | 6.1167 / 34.3086 ms |
| Input conversion mean | 0.3773 ms |
| Pose / hands / face mean | 12.9770 / 13.9971 / 6.1500 ms |
| Parallel inference wall, mean / max | 15.1565 / 50.0589 ms |
| Result assembly mean | 0.4793 ms |
| Total tracker mean | 16.0132 ms |
| Preview composition mean | 2.8386 ms |
| Host receive to preview, mean / max | 25.0426 / 74.7538 ms |

Capture accounting balances. Replacements were 3.704% of captured frames.
A nominal 30 FPS input limits throughput headroom to approximately
3.84% for this run. That is not a limit on latency reductions.
Independent task times are concurrent and must not be added as wall time.

**Confirmed:** preview and HighGUI are serialized after inference in baseline;
GUI submit/event processing is outside the supplied latency window. The code
reprojects vertices repeatedly and draws on 1280x720 before reducing to 960x540.

**Not established:** HighGUI is the actual cause of the 148 replacements; native
hand-model tail latency cause; thermal state; CPU/GPU memory costs; visual
accuracy. Low hand-detection counts cannot show failure without visibility data.
The previous 14.88 FPS repository benchmark used different live inputs and is
not a controlled speedup reference for this log.

## Implemented changes and invariant costs

- Opt-in `motioncapture.live_app` leaves `motioncapture-demo` untouched. A single
  worker owns construction/open/process/close of the same pinned tracker.
- Request at most one next frame before main-thread composition/presentation.
  Memory cannot grow with queued inference futures. One extra completed source
  image may be retained, and its age is measured. Slower GUI can still backpressure
  this bounded pipeline; an improvement on the user's host is not assumed.
- Resize preview first, batch line segments, project vertices once, project only
  referenced face-contour vertices, cache static panel labels, and render into one
  newly owned output buffer. Inference still receives the original image.
- The display overlay rasterization changes slightly (display-pixel line widths).
  Source pixels, observations, models, detection thresholds, and handedness do not.
  `--preview-mode native` selects the unchanged old renderer explicitly.
- Bounded 0.25-ms latency histograms add p50/p95/p99 upper bounds, budget exceedance
  counts, result residence, camera wait, GUI submit and event-pump timings. This
  adds small CPU/memory overhead; percentile collection never retains frame traces.
- A GUI submit timestamp is not a physical presentation timestamp. True
  sensor-to-photon latency remains unmeasured. Camera hardware timestamps and
  camera-driver buffering are not changed.

## Measured local verification

47 targeted tests passed in Linux / Python 3.13.5, with real OpenCV 4.13.0 pixel
operations and explicit synthetic camera/tracker doubles. Tests cover native
camera ownership boundaries through the real CaptureRuntime, one-ahead bounds,
identity, source immutability, main-thread GUI calls, exact-N headless operation,
loss/failure/timeout handling, shutdown accounting, and bounded histogram memory.
No production fallback or fake input source is added.

The synthetic camera+overlay **kernel** benchmark used 1280x720 input, six
alternating-order rounds of 250 iterations and four OpenCV threads. It excludes
panel text, native MediaPipe, camera IO, and GUI. Round-mean medians were:

- Baseline-order reference kernel: 4.0582 ms.
- Display-resolution production kernel: 2.6743 ms.
- Reduction of this synthetic kernel's time: 34.10%.

This is **not** a measured M5 FPS gain, full preview gain, or native tracker gain.
See `2026-09-06-preview-kernel.json` for raw timing summaries and environment.

Python 3.12 syntax parsing passed. Python 3.12 execution, Ruff, the complete
original repository suite, actual MediaPipe/Metal initialization, Mac/Windows
camera and GUI tests, and long-session thermal validation remain **unverified**.

## Device acceptance

Run the original baseline and candidate with the same resolution, lighting,
framing, gestures (including visible hands) and session duration. Run both
headless and live; the old log lacks GUI timing and headless metadata. Compare
queue age, receive-to-preview mean/tail, event pump, replacement fraction,
completion FPS and visual hand/face quality. A gain in FPS alone does not accept a
latency/quality regression. Use 2-minute runs first, then a longer thermal run.

If overlap increases result residence or tail latency, use the explicitly
selected sequential mode and retain both reports. Do not automatically switch
modes after an error. Do not merge this candidate on synthetic timing alone.
