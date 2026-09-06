# Exact-input preprocessing and bounded stage overlap

Additive experiment against `dfb7b12fdee10352fc3f7b409c6acaf4d1a3bdbd`.
No existing file, live default, dependency lock, model or user capture is modified.
The user's uncommitted detector-provider patch is NOT reconstructed, overwritten,
or claimed as verified here. This entry point uses shared primitive codecs and
session ownership, not the estimator or comparison helpers' changing signatures.

## Run

Keep the existing model enrollment and pinned optional dependencies. From the repo:

```bash
git fetch origin
git switch feat/wholebody-onnx-lab
git pull --ff-only origin feat/wholebody-onnx-lab

uv run --with-requirements tools/requirements-wholebody.txt \
  python -m motioncapture.wholebody_optimize_bench \
  "$HOME/Downloads/20260906_030954.mp4" \
  --detector-provider coreml-all --pose-provider coreml-all \
  --allow-cpu-partitions --research-only \
  --diagnose-from sessions/dwpose-detector-coreml-abba-rerun.json \
  --output sessions/wholebody-optimize-r1.json
```

Do not reset or discard existing local changes. All files in this patch are new.
Existing output/checkpoint names are refused, never overwritten. Return the main
`wholebody-optimize-r1.json`; on native abort, also return the last started record
and preceding checkpoints. Images, audio and individual coordinates are not stored.

## Boundary diagnostics first

The command selects up to 24 frames from the eight largest nonzero reported pose
errors plus immediate neighbors. Source SHA and original PTS must match the input.
It checks source pixels, detector outputs, box edges, affine matrices, crop pixels,
normalized tensors, SimCC argmax changes, repeated identical tensor outputs and
coordinate-transform-only counterfactuals. These probes are NOT timed tracking FPS.
No rounding, coordinate filtering or output suppression is used to hide differences.
Zero/multi-person diagnostic frames are explicitly unmatched, not zero error.

CPU-detector and selected-detector crops run A/B/B/A through ONE fixed pose session.
This separates input changes from same-input variability; it does not reproduce
all device load histories. No exact root cause or ground-truth accuracy is assumed.
The source arrays remain in RAM; reports contain hashes, deltas and identifiers.

## Six-pass optimization comparison

All selected source frames (default 900) run in this order, using fresh sessions:

1. Sequential detector/pose, original preprocessing.
2. Sequential detector/pose, exact lookup normalization.
3. Detector(next frame) overlapped with pose(current frame), lookup normalization.
4. Repeat 3, then 2, then 1.

All arms keep the SAME requested providers, assets, thresholds and source order.
Detection runs on EVERY source frame. Every detected person up to the existing
capacity eight is processed. No retained poses, interpolation, omitted hands,
changed model precision, changed crops or hidden provider fallback is added.
The capability remains WholeBody133 2D, NOT world coordinates, facial blendshapes,
retargeting, calibrated depth or complete 3D capture.

The existing OrtModel owns provider setup, profiling evidence and rejection.
`--allow-cpu-partitions` is explicit mixed-graph consent. CoreML ALL does not prove
physical GPU/ANE dispatch. Requested unavailable providers fail, not become CPU.

`--suite pipeline` isolates overlap with sequential-LUT / overlap-LUT / overlap-LUT /
sequential-LUT. `--suite normalize` isolates original/LUT in ABBA order.
`--suite diagnostics` only runs the source-locked probes and requires the report.
`--max-frames 0` selects the full file after smoke verification. No source timebase
is rewritten to 60Hz. CPU on Windows/Linux is explicitly selected with both
`--detector-provider cpu --pose-provider cpu`; native GPU paths there are not claimed.

## Implemented optimization and timing contracts

A fixed 256x3 float32 lookup table implements the original float64 subtract/divide
followed by float32 conversion for every uint8 BGR pixel. Affine geometry remains
unchanged. No previous image or observation is cached. The alternative recipe is
byte-checked against the imported original `pose_tensor` on source preflight and
all diagnostic crops; incompatible local recipe changes fail instead of being hidden.

One dedicated worker owns each model through construction, inference and destruction.
Both sequential controls and overlapped arms use those same ownership boundaries.
Each stage admits ONE outstanding future, including a completed-but-unreceived one.
Exactly N prefix frames produce N detector calls; no speculative N+1 inference occurs.

Queue wait, per-person pose calls, summed model work, and detector-submit-to-pose-end
latency are distinct measurements. Loop throughput is NOT the inverse of pipeline
latency. Both models may share hardware and slow one another; overlap is a candidate,
not a demonstrated M5 speedup.

All six loops include pixel/result hashing and bookkeeping. Comparison adds work
outside the enrollment arm. These FPS values are not exactly the old instrumentation;
compare arms within this report and use stage times for cross-run context. The lookup
recipe preflight is included in its first loop iteration and explicitly labelled.
All frames remain reported; the after-first-60 summary is additional, not a discard.

## Quality, failure and completion gates

Frame counts, original PTS, hashes, numeric disagreements, per-person accounting,
terminal reasons and cleanup must remain visible. Comparisons use the first successful
sequential arm, NOT ground truth. The known CPU-detector vs CoreML-detector sparse
large pose differences remain OPEN even if all six new hashes match.

Exceptions retain the completed prefix. Each arm writes an intent checkpoint before
native setup and a terminal checkpoint afterward. A native hang cannot be forcibly
cancelled; shutdown waits for the owner. A crash can leave an unmatched started file.
No watchdog, process isolation, silent timeout extension or automatic retry is claimed.

Execution success is not optimization completion. Native-device speed, visual quality,
sustained new-frame rate, thermals and live end-to-end latency remain gates. There is
no 60FPS, accuracy, M1/Windows-native performance or commercial release clearance here.

## Executed verification

See `benchmarks/2026-09-06-stage-optimization-verification.json`. The 76 targeted tests
use real OpenCV and FFmpeg VFR fixtures; model/session tests use explicit doubles.
The optional ONNX runtime installation was unresolved in this environment. Full repo
and uncommitted user-source tests are not claimed. Ruff and native Python3.12 execution
were unavailable; Python3.12 syntax was checked during preparation.

The recorded-pixel microbenchmark used a SYNTHETIC fixed ROI, not a detected person.
Its mounted input hash differs from the M5 golden-clip hash, so it is not a replay
of that inference benchmark. All 2,700 compared preparations were byte-identical;
local kernel time fell from median-round-mean 1.0543 to 0.6058ms. That 42.54% reduction
is NOT a native M5 or whole-application speedup. All 256 pixel values were also tested.

```bash
uv run pytest tests/test_wholebody_optimization.py
uv run ruff check src/motioncapture/wholebody_fast_input.py \
  src/motioncapture/wholebody_stages.py src/motioncapture/wholebody_boundary_probe.py \
  src/motioncapture/wholebody_stage_comparison.py \
  src/motioncapture/wholebody_optimize_bench.py tests/test_wholebody_optimization.py
```

Recipe attribution remains in `third_party/WHOLEBODY_NOTICE.md` (RTMLib/MMPose).
Runtime documentation: https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html
No new provider option or runtime version is introduced.
