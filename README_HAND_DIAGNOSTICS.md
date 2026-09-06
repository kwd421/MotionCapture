# Browser hand failure diagnostics (schema 2)

Based on `feat/hand-gpu-lab` at `f14afcb824def4ef1898a1cc8deef200a7f10e9d`.
This patch fixes lost failure evidence and error propagation. It does NOT establish
that the reported GPU timeout is fixed, improve inference, or claim 60 FPS.

The supplied smoke finished six 900-frame passes. Native ran at 51.36/50.00 FPS,
web CPU at 12.66/12.58, and web GPU at 35.60/35.21. In the full run, only native
(44.54), web CPU (10.33), and the first web GPU (26.83) completed 6,442 frames.
The user observed the next GPU pass stop around progress frame 3300. The schema-1
report omitted that pass entirely and retained only `browser_request_timeout`.
Its exact failing sequence and cause cannot be recovered from those aggregates.

The current HTTP/RGB browser candidate is not accepted as a performance upgrade.
Do not treat Web GPU versus Web CPU as a speedup over the native baseline. The
full GPU task call averaged 22.04 ms and the complete hand boundary 32.98 ms.
Transfer/preparation removal alone is not demonstrated to meet the 16.67 ms budget.
Native controls after the failed pass never ran, so the full comparison is incomplete.

## What changed

- Each failed pass now remains in `runs`, with completed-prefix statistics,
  last completed/current source frame and PTS, failure phase and cleanup states.
  A timeout is not included as a completed frame or disguised as successful FPS.
- Cleanup errors no longer replace the primary processing exception.
- The bridge freezes request ID/op/model timestamp, fetch/write/result state,
  and latest sampled browser phase before clearing an outstanding request.
- A main-page heartbeat, independent of the inference worker, samples the last
  worker phase every second. Worker errors, message errors, page close and GPU
  context loss report fixed fault codes. No exception text/tokens/images are sent.
- Top 16 per-frame disagreement maxima include only source sequence/PTS and
  aggregate pixel distance, not coordinates. Partial comparisons are labelled.

No model, thresholds, two-hand setting, full-resolution input, RGB transport,
PTS, inference scheduling, request timeout or live defaults were changed. There
is no retry/restart/CPU fallback after a failure. Diagnostics add overhead; compare
modes inside the same new run, not sub-millisecond changes across report versions.
A deliberate context close is detached from the context-loss fault handler first.

`browser_failure.browser_progress` is sampled evidence, not an exact instruction
trace. A fresh page heartbeat with an old `detect` phase localizes a stall but
does not prove a GPU driver bug. Missing heartbeats can mean a frozen page, network
failure, suspension or other causes. Host and browser absolute clocks are not
subtracted. Hidden events for completed passes do not prove the failed pass stayed visible.

## Next diagnostic run

Use the same clip with the existing SDK installed. Omit `--include-web-cpu` to
avoid repeating the known-slow CPU-browser control. Test the first 4000 frames,
covering the previously reported progress position without seeking or resetting
tracking within a pass:

```bash
uv run python -m motioncapture.hand_acceleration_bench \
  "$HOME/Downloads/20260906_030954.mp4" \
  --max-frames 4000 \
  --output sessions/hand-backends-diagnostic-v2.json
```

Open the new printed localhost URL in Chrome/Edge and keep its tab visible.
Plan: native / web_gpu / web_gpu / native. This is an explicit prefix, not full
or sustained acceptance. Omitting the long Web CPU pass changes prior load and
may change reproducibility; success does not prove the original fault fixed.
Never increase the timeout as a substitute for finding the blocked stage.
Reports do not overwrite an existing path. Close the old experiment tab.

## Verification scope

Linux/Python 3.13.5: 60 targeted Python tests pass, including real loopback HTTP,
real FFmpeg/OpenCV VFR decoding and deterministic injected fourth-pass failure.
Node 22: 8 protocol/page-lifecycle tests pass using explicit worker/DOM doubles.
Python 3.12 syntax and JS syntax checks pass. Actual Chromium tests were attempted
and fail before navigation with `ERR_BLOCKED_BY_ADMINISTRATOR`; neither browser
GPU nor real MediaPipe inference was executed here. Full repository/Ruff, actual
Python 3.12, macOS/Windows hardware and long-duration acceptance remain unverified.

```bash
uv run pytest tests/test_browser_hands.py tests/test_hand_comparison.py \
  tests/test_acceleration_runner.py tests/test_recording_bench.py \
  tests/test_recording.py tests/test_recorded_tracker_clock.py \
  tests/test_hand_backend_diagnostics.py
node --test tools/browser_hands/protocol.test.mjs tools/browser_hands/page.test.mjs
# Optional real browser protocol test, synthetic SDK, not native model accuracy:
MOCAP_BROWSER_E2E=1 uv run pytest tests/test_browser_worker.py
```
