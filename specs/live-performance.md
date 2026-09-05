# Live performance slice

Baseline: `932e8e8d90a0b86629d4e8d35ea165ba87edc094`.

Definition: retain full pinned CPU Pose/Hands/Face inference while overlapping
one next-frame inference with main-thread preview work and reducing preview-only
work. The original `motioncapture-demo` is unchanged for an exact A/B baseline.

## Contract

- New entry point: `uv run python -m motioncapture.live_app`.
- Same 1280x720/30 request, models, thresholds, VIDEO tracking, numeric validation,
  handedness, face/hand capabilities and metadata-only privacy defaults.
- A single worker owns tracker construction/open/process/close. Original
  CaptureRuntime retains ownership of the native camera. UI calls stay on main.
- At most one next-frame request or completed packet is outstanding. Never queue
  an unbounded set of futures. Each packet keeps its own source frame and identity.
- `overlap` requests the next frame before composing/presenting the current frame;
  `sequential` requests after it. These are selected modes, not error fallbacks.
- Headless N mode infers exactly N frames; it never prefetches N+1. Interactive
  quitting may abandon one in-flight result, which is separately counted.
- A failure remains terminal and pending results are not published after it is
  observed. Native hangs are not forcibly killed; timeout means failure with
  incomplete cleanup. No native-process crash isolation is claimed.
- Preview `display` resizes to 960x540 before drawing; this changes only overlay
  rasterization, not inference or observation values. `native` preserves the old
  composition path for comparisons. No dynamic pose or face results are cached.
- Bounded timing histograms report percentile upper bounds, not exact percentiles;
  GUI submit/event-pump timings do not claim sensor-to-photon latency.

## Acceptance and verification

Deterministic event-controlled tests cover one-ahead bounds, sequential tracker
ownership, identity, failure/cancellation, cleanup, and exact-N headless capture.
Real OpenCV tests cover pixel immutability, mirroring, detection loss and bounded
preview/template state. Tests use synthetic inputs only, not recorded users.

Local microbenchmarks are not native MediaPipe or Mac/Windows camera benchmarks.
Real camera, thermal, jitter, confidence, 2-minute/30-minute sustained A/B and full
Python 3.12/Ruff validation remain gates on the user's device. Do not claim a
speedup in the native tracker from faster preview code.
