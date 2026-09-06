# DWPose / RTMW ONNX benchmark candidates

Additive experiment based on `669e7c1af6f381413ac9d819aeae17c6f34c9f9f`.
**Not a verified speedup, full mocap replacement, or commercially cleared release.**
Existing live/native/browser commands and dependency lock are unchanged.

## The candidates

| CLI model | Actual exported checkpoint | Pose input H x W |
| --- | --- | --- |
| `dwpose-m` | RTMPose-m DWPose-distilled ucoco checkpoint, 2023-07-28 | 256 x 192 |
| `rtmw-m` | RTMW `dw-l-m` Cocktail14, 2023-11-22 | 256 x 192 |
| `rtmw-l` | RTMW `dw-x-l` Cocktail14, 2023-11-22 | 256 x 192 |

Every candidate uses the SAME CPU YOLOX-tiny HumanArt person detector at 416x416.
Detector cost is included. It runs on EVERY selected original frame. All its
person boxes are processed, not just the first. There is no person-ID tracking.
Score/NMS thresholds are recorded; detector results are not ground-truth persons.

Source frames remain original size; letterbox and 1.25-padded person crops are
explicit model preprocessing. No adaptive downscale, frame skip, prediction,
interpolation, FPS-header rewriting or unannounced 1-hand mode.

### Capabilities are different

WholeBody133 gives body17 + feet6 + face68 + hand21 each, in source-image **2D**.
It does NOT give MediaPipe's body33/world coordinates/face478/blendshape channels.
The optional MediaPipe baseline computes different and additional outputs. Its
FPS is contextual, NOT a feature-equivalent speedup claim. Do not zero-fill 3D
or produce fake MediaPipe records. Detailed facial capture remains a separate
future input. Keypoint confidence coverage is NOT accuracy or hand-detection AP.

## Setup on the new branch

From the repository root, keeping existing local edits intact:

```bash
git fetch origin
git switch --track origin/feat/wholebody-onnx-bench
uv sync
ffprobe -version
uv run python -m motioncapture.wholebody_assets list
uv run python -m motioncapture.wholebody_assets fetch
```

Downloads are explicit and digest checked. The selected mirror revision and
original URLs are shown by `list`. No silent mirror retry or overwrite. Expect
about 419 MB of archives plus extracted graphs. This patch does NOT ship weights.
Read `THIRD_PARTY_WHOLEBODY.md` before distributing the models commercially.

`onnxruntime==1.23.2` is an explicit **optional uv overlay**, not a replacement of
MediaPipe or the project's lock. That release has a macOS arm64 Python3.12 wheel.
At runtime we still check the actual version and provider availability. No CUDA,
DirectML, MPS, CoreML or ANE support is inferred merely from the machine name.

## First: 900-frame screen, six candidate cells + two native controls

```bash
uv run --with onnxruntime==1.23.2 python -m motioncapture.wholebody_bench \
  "$HOME/Downloads/20260906_030954.mp4" \
  --models dwpose-m rtmw-m rtmw-l \
  --providers cpu coreml-all \
  --max-frames 900 --include-mediapipe \
  --output-dir sessions/wholebody-smoke
```

No browser, HTTP bridge, Node or GPU tab. No preview composition/display in ANY
of these passes, including native controls. This is an unpaced file service
benchmark, NOT camera/display latency. Each pass resets sessions and source PTS.
The prefix is explicitly labelled; it does not validate sustained/full-clip FPS.

Per candidate: checks assets, opens real runtime, validates tensors and one
real-frame provider-placement probe, then reopens the file from its first PTS.
A full-frame pose ROI is used ONLY for that separate placement probe; when the
measured detector sees no person, the runner does NOT substitute a full-frame ROI.
Model setup/compilation/probe are outside frame timing; the first measured frames
remain in all-frame statistics. A separate post-first-60 summary is also provided.

## CoreML policies

| CLI | Requested pose policy | Detector |
| --- | --- | --- |
| `cpu` | ORT CPUExecutionProvider | ORT CPU |
| `coreml-gpu` | CoreML CPUAndGPU | ORT CPU |
| `coreml-ane` | CoreML CPUAndNeuralEngine | ORT CPU |
| `coreml-all` | CoreML ALL | ORT CPU |

These are allowed-compute policies, NOT GPU-only/ANE-only assertions. CoreML can
choose its own CPU internally. Unexpected ORT CPU graph fallback is disallowed;
unsupported graph/provider combinations become FAILED cells, never CPU successes.
Strict placement may reject a graph that could work with explicitly permitted
mixed-provider partitioning; this slice does not add such a mode silently.
The independent requested cells continue after a caught failure.

To screen all policies, use a NEW output directory:

```bash
uv run --with onnxruntime==1.23.2 python -m motioncapture.wholebody_bench \
  "$HOME/Downloads/20260906_030954.mp4" \
  --models dwpose-m rtmw-m rtmw-l \
  --providers cpu coreml-gpu coreml-ane coreml-all \
  --max-frames 900 --output-dir sessions/wholebody-policies
```

## Full, order-balanced confirmation for a selected candidate

For example, DWPose-m CPU / CoreML ALL / CoreML ALL / CPU:

```bash
uv run --with onnxruntime==1.23.2 python -m motioncapture.wholebody_bench \
  "$HOME/Downloads/20260906_030954.mp4" \
  --models dwpose-m --providers cpu coreml-all \
  --max-frames 0 --repeats 2 \
  --output-dir sessions/wholebody-dwpose-full
```

This is an example, not a recommendation that DWPose-m has won. Choose the model
and provider after the screen. Reversed repeat order reduces a simple ordering
bias but does not establish matched heat, power state or external load.
On Windows, `--providers cpu` is the initial supported policy; GPU providers are
not implemented in this patch. M1/M5 speeds remain a real-device measurement.

## Reports to return

Send the entire result directory zipped, not only `summary.json`:

```bash
zip -r wholebody-smoke-results.zip sessions/wholebody-smoke
```

- `manifest.json`: source hashes/PTS, active candidates/providers/config/runtime.
- `run-XX.started.json`: experiment intent BEFORE native model construction.
- `run-XX.json`: terminal status, partial or full counters, timings, actual schema,
  probe provider events, cleanup and optional same-model CPU disagreement.
- `summary.json`: compact matrix and paths to the per-cell reports.

Output directories/files are never overwritten. If a native library aborts or
hangs outside Python's exception handling, previous cells remain saved and the
last started record identifies the unfinished one; no watchdog/process-isolation
claim is made. Ordinary caught errors preserve the completed prefix with FPS=null.
Exit status 2 means failures/interrupts were recorded; it is NOT a successful run.

Same-model CPU reference positions are bounded RAM only. Comparison requires
one detected person on each side, matching original sequence/PTS and the same
model. Per-part confidence changes and per-frame XY disagreement distributions
are recorded. Zero/multiple-person cases are explicitly unmatched, not zero error.
Different models' raw confidence scores are not calibrated or directly comparable.

## Verified here / remaining gates

See `benchmarks/2026-09-06-wholebody-candidates-verification.json` for executed tests.
Real OpenCV pixels and real FFmpeg VFR fixture decoding are tested. ONNX session
policy, model output and failure paths are tested with clearly labelled doubles.
This execution host cannot download the weights/ORT wheel; actual ONNX inference,
CoreML conversion, model-specific quality, Mac/Windows performance, Ruff and
full Python3.12 project execution remain UNVERIFIED. Nothing is promoted by
synthetic output FPS. No private clip/pixels/individual coordinates are committed.
