# Hand backend experiment — research, patch and verification

Date: 2026-09-06. Base: `d77c332e08d281d80d50087c846f1dc6452fb836`.
Status: implemented experimental adapter; native hardware performance **unknown**.

## Prior evidence and decision

User-supplied serial log: two-hand + face workload hand API mean 16.7214 ms,
p95 17.1241 ms. Entire serial file loop 29.9219 FPS. Earlier parallel controls
47.2219 / 40.8919 FPS with repeat drift. A full library hand call, not pure network
compute, is the measured bottleneck. Staggering did not establish a robust win.
These are the user's M5 measurements, not measurements on this execution host.

Moving hand inference to a different compute backend is the chosen next axis.
Three alternatives were examined:

1. Simply select Python `Delegate.GPU`: the official Python BaseOptions reference
   documents Ubuntu-limited GPU support. Not a verified macOS switch for the
   installed 0.10.31 wheel. Do not silently convert failure back to CPU.
2. Core ML / ONNX: viable later backend candidates, but the `.task` is a complete
   detector/ROI/tracking/landmark pipeline, not a converted model with a proven
   equivalent end-to-end adapter. No conversion or ANE speedup is claimed here.
3. MediaPipe Web GPU: selected controlled experiment. Its official API accepts
   the same task bundle and GPU selection. The bridge lets native Pose/Face stay
   fixed, while a Web CPU control separates web/runtime/transport costs from a
   GPU effect. Web 0.10.32 vs native Python 0.10.31 is an explicit confounder.

Sources inspected (primary API/source or the published package artifact):
- https://developers.google.com/edge/api/mediapipe/python/mp/tasks/BaseOptions
- https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/web_js
- https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html
- https://app.unpkg.com/@mediapipe/tasks-vision@0.10.32/files/vision.d.ts
- https://app.unpkg.com/@mediapipe/tasks-vision@0.10.32/files/vision_bundle.mjs
- https://app.unpkg.com/@mediapipe/tasks-vision@0.10.32/files/wasm

The pinned SDK loader calls `importScripts` to load its WASM glue. A module worker
would not provide that operation; the patch deliberately uses a classic worker
with dynamic module imports. No SDK monkey patch or network-loader fallback was
added. Result fields, ImageData support, canvas option and the four WASM asset
filenames were checked against the pinned package rather than assumed.

## Patch invariants and trade-offs

Only an explicit hand-task factory changes the existing tracker; its native
constructor/defaults stay unchanged. All .task files, thresholds, two-hand limit,
full input dimensions and original PTS remain unchanged. Native task cleanup now
attempts all three resources even if one close fails. All old benchmark modes
remain available. Full-file behavior remains EOF/count/checksum verified.

Local HTTP serves a closed whitelist of SDK/app/model bytes. RPC needs a random
capability token, one client ID, correct Host and same Origin. No directory serving,
CDN, external frame transfer, raw-image logging or persistent per-frame biometrics.
One synchronous outstanding frame. TCP_NODELAY prevents transport packet batching;
its performance effect is not benchmarked separately. Transfer costs are included
in hand call and pipeline service measurements, not hidden behind browser-only ms.

Only known non-software identified WebGL2 contexts are allowed for GPU mode.
That is not proof of per-operator dispatch or ANE/Metal-specific utilization.
Failed initialization/transport is terminal. No GPU-to-CPU fallback. Web CPU is
an explicitly selected experimental control, not a recovery mode.

Performance reports include reference-only/candidate-only hands and matching-slot
pixel/world displacement. The CPU output is a reference, not annotated truth.
A backend difference can change detections/handedness/float outputs; quality needs
review. Tab visibility is measured; thermals/power/other processes remain unknown.

## Executed verification

Environment: Linux x86_64, Python 3.13.5, Node 22.16.0. No real MediaPipe execution.

```bash
PYTHONPATH=src pytest -q \
  tests/test_acceleration_runner.py tests/test_browser_hands.py \
  tests/test_hand_comparison.py tests/test_recording.py \
  tests/test_recorded_tracker_clock.py tests/test_recording_bench.py \
  tests/test_scheduling_comparison.py tests/test_staggered_scheduling.py \
  tests/test_live_preview.py tests/test_browser_worker.py
node --test tools/browser_hands/protocol.test.mjs
```

Final counts are recorded in the accompanying verification manifest. These tests
exercise real Python localhost HTTP, binary RGB, actual encoded VFR decoding with
FFprobe/OpenCV, timestamp/count/error boundaries, model-lifecycle injection,
reference comparisons, and unchanged CPU scheduling/preview tests. Model calls
in tests are synthetic, not disguised native inference.

Two real Chromium-worker tests were attempted with an explicitly synthetic SDK.
Both were blocked at localhost navigation by environment policy before a worker
ran (`net::ERR_BLOCKED_BY_ADMINISTRATOR`). No policy bypass was attempted. They
are opt-in via MOCAP_BROWSER_E2E=1 and skipped in the default targeted suite.
Thus actual browser SDK loading, GPU execution and real browser transport remain
unverified. Neither test result nor source inspection establishes a speedup.

Python 3.12 AST syntax and Node syntax are checked, not actual Python 3.12 runtime.
Ruff and the complete repository suite were not run in the partial local checkout.

## Next target-host gate

Run the explicit prefix smoke test first, then the all-frame six-pass comparison
from README_HAND_ACCELERATION.md. Review browser errors/provider identity, source
and model hashes, hands_ms including transfer, frame_service/loop FPS, hand
comparison metrics, native-control repeat consistency and visibility. No candidate
is automatically selected. Continuous live 60Hz capture and display still require
a separate hardware-camera/GUI/thermal test after a useful backend is identified.
