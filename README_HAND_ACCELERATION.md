# Hand backend R&D: native CPU / Web CPU / Web GPU

Experimental candidate based on `d77c332e08d281d80d50087c846f1dc6452fb836`.
**This is not a proven speedup or a 60 FPS release. Live defaults are unchanged.**

The existing Python MediaPipe CPU path still owns Pose and Face. Only Hands can
be supplied by a localhost browser worker. All three tasks run on every source
frame, with the same pinned `.task` files, full-size pixels, original video PTS,
two-hand configuration and 0.5 confidence thresholds. No resizing, JPEG transport,
frame skipping, interpolation, new model weights or automatic CPU retry is used.

The browser runtime is `@mediapipe/tasks-vision@0.10.32`; Python stays at 0.10.31.
The weights match, but runtime/API/GPU floating-point behavior may differ. CPU
reference predictions are not ground truth. Browser GPU means the WebGL GPU
delegate is requested on an identified, non-software WebGL2 context; it does NOT
prove dispatch of every operator, Metal internals or Neural Engine usage.

## Setup

Use the `feat/hand-gpu-lab` branch. Keep local changes; do not reset your tree.
From the repository root, with Node/npm, uv, ffprobe and Chrome or Edge available:

```bash
uv sync
npm install --prefix tools/browser_hands --ignore-scripts
uv run motioncapture-models verify
ffprobe -version
```

Npm downloads the fixed SDK once. The experiment serves a closed asset whitelist
from local files, not a CDN. No video or predictions are uploaded to a remote
service. The ephemeral URL token is printed only in the terminal, not the report.

## First: explicit 900-frame prefix smoke test

```bash
uv run python -m motioncapture.hand_acceleration_bench \
  "$HOME/Downloads/20260906_030954.mp4" \
  --max-frames 900 --include-web-cpu \
  --output sessions/hand-backends-smoke.json
```

After file inspection, open the printed `http://127.0.0.1:.../#...` URL in a
hardware-accelerated Chrome/Edge window. Keep that tab visible and do not reload
it. Use a separate terminal window beside it to see progress. Connection timeout
is 120 seconds; `--browser-timeout` may explicitly set up to 300 seconds.

The GPU graph is initialized and processes one real frame before the native
baseline to detect unsupported GPU/image paths early. That preflight and task
setup are reported separately and excluded from per-frame pass timings.

The smoke test processes only the first 900 original frames in each pass. It is
reported as `explicit_prefix`, not full-file coverage. Do not compare its average
FPS directly with a complete clip or treat it as long-duration acceptance.

## Full file comparison

```bash
uv run python -m motioncapture.hand_acceleration_bench \
  "$HOME/Downloads/20260906_030954.mp4" \
  --include-web-cpu \
  --output sessions/hand-backends-full.json
```

Six passes: **native / web_cpu / web_gpu / web_gpu / web_cpu / native**.
Omitting `--include-web-cpu` selects four passes: native / web_gpu / web_gpu /
native. Every pass resets model state and preserves every original frame/PTS.
The existing `recording_bench` and `live_app` commands remain available unchanged.
Output files are never overwritten; choose a fresh name for each run.

## Read the result without mistaking a faster model call for a faster pipeline

- `all_frames.stages.hands_ms` includes Python RGB packing, HTTP transfer, browser
  preparation, hand inference and result-return overhead. It is the useful
  replacement-boundary cost, not just time spent in a neural network.
- `hand_backend.stages.browser_detect_ms` isolates the browser task call;
  `browser_pixels_ms` measures RGB-to-RGBA preparation. `rpc_roundtrip_ms` includes
  their transport envelope. Timings from different clocks are not subtracted to
  manufacture an end-to-end latency.
- `frame_service_ms`, `total_ms` and `unpaced_loop_fps` measure different scopes.
  The latter includes result-hashing and comparison bookkeeping. This file runner
  is unpaced and serializes decode/frame service; it is not the live display path.
- `hand_comparison` compares matching handedness slots against the first native
  run: 2D displacement in source-image pixels, hand-relative world displacement
  in mm, and reference-only/candidate-only detections. Unmatched points are not
  silently assigned zero error. These numbers measure disagreement, not accuracy.
- Full numeric output hashes are recorded. GPU equality is not required, but a
  difference requires quality review. `native_reference_repeat_equal` checks
  whether the two native controls themselves stayed numerically identical.
- Hidden-tab events mark performance comparison as contaminated. A visible tab
  does not establish matched temperature, clock, power, browser or system load.

**The localhost RGB bridge might cost more than GPU acceleration saves.** A low
`browser_detect_ms` with a high `hands_ms` is an important negative result, not a
reason to ignore transfer cost. This bridge is a measurable R&D boundary, not a
final broadcast transport architecture. Do not promote a candidate from FPS alone.

## Failure and privacy

No GPU-to-CPU fallback. Missing assets, wrong SDK version, software/unknown GPU,
malformed/duplicate RPC, changed timestamps, decoder failure, a closed tab or a
request timeout terminate the experiment. A failed report is still useful; send
that report and the browser's visible error, not the secret local URL.

One frame/RPC is outstanding. The source video remains on disk; full RGB frames
are transient on loopback. Reference hand predictions are bounded in RAM only;
reports never write frames, audio, individual landmarks or private input paths.
The browser must close its task before the next pass; abrupt disconnect cleanup
is not reported as a clean GPU session. Closing the page releases browser state.

## Verified scope of this patch

Targeted Python tests run on Linux/Python 3.13.5, plus Node protocol tests and
Python 3.12 syntax parsing. Test doubles are explicitly synthetic. The real
Chromium worker tests were attempted but localhost navigation is blocked by this
execution environment (`ERR_BLOCKED_BY_ADMINISTRATOR`); they are optional, not
reported as passing. Pinned native MediaPipe and GPU inference were not available
here. Ruff, the full repository suite and Mac/Windows runtime acceptance remain
unverified. See `benchmarks/2026-09-06-hand-backends.md` for research and commands.
