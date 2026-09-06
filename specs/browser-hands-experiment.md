# Browser hand acceleration experiment

Base: d77c332e08d281d80d50087c846f1dc6452fb836.

Measure whether moving only HandLandmarker off the Python CPU improves the real
three-task pipeline. Native Pose/Face and their defaults remain unchanged.
Explicit modes: native CPU, browser WASM CPU control, browser WebGL GPU candidate.
Same pinned hand .task, two hands, VIDEO, all confidence thresholds 0.5, full RGB
pixels, original file PTS, no detector cadence change, no resizing or frame drop.
Web SDK is 0.10.32; Python is 0.10.31. Same weights does NOT mean identical runtime.

One authenticated, loopback-only browser worker consumes one outstanding RPC at
a time. The Python owner waits for completion before releasing/reusing pixels.
No external CDN is loaded during capture; npm installation is a separate explicit
step. Allowed static assets are a closed set; paths and arbitrary repo files are
not served. Token, raw images and per-frame predictions never enter reports.
Image transfer is lossless RGB, not JPEG. HTTP/RGB packing/unpacking costs stay in
hands_ms; browser task time is an additional diagnostic, not a substitute for it.

No GPU-to-CPU retry. Known software renderers and unavailable/unidentified GPU
contexts fail GPU initialization. Renderer identifies a context, NOT dispatch of
every model operator. No ANE/Metal-specific claim or per-operator verification.
Browser tab visibility changes invalidate performance comparability.

Compare full-pass results using ABBA native/GPU/GPU/native. Optional six-pass
native/web-CPU/GPU/GPU/web-CPU/native separates bridge/runtime from GPU benefit.
Fixed prefix is explicit, never labelled a full-file run. Full mode consumes EOF.
Per-frame reference hands are kept in bounded RAM only; reports contain aggregate
matching-slot displacement, detection disagreement and full-output hashes.
CPU/GPU float hashes need not match: mismatch cannot imply success or failure of
accuracy without a quality review. Reference CPU is not ground truth.

Live defaults and VMC/retargeting are out of scope. Acceptance: actual target-Mac
all-frame comparison plus time/quality gates. Synthetic protocol/browser tests
verify transport and failure states only, never native inference speed.

## Failure evidence extension (schema 2)

A pass must not disappear when processing/cleanup fails. The recording owner
retains completed-prefix groups and last completed/current source sequence/PTS;
failed samples never become valid frames or a successful full-pass FPS. Cleanup
acknowledgements and original failure type remain separate.

The bridge owns one immutable first-failure snapshot before request disposal.
It contains only request lifecycle fields and latest authenticated page heartbeat
(last worker phase/ID, phase age, host receive age, visibility). Heartbeats do
not reset request deadlines and are not proof of GPU dispatch or root cause.
Worker faults and unexpected WebGL context loss are terminal; deliberate close
must not trigger a false context-loss failure. No model restart or CPU fallback.

Disagreement locations are bounded to 16 frame/PTS/max-distance records and
contain no raw coordinates. Partial quality summaries are explicitly incomplete.
Acceptance includes injected fourth-pass failure, original exception surviving
cleanup, bounded sampled telemetry and absence of private frame/path/token data.
See README_HAND_DIAGNOSTICS.md for current negative results and target run.
