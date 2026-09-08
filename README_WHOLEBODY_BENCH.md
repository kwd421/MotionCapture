# Whole-body ONNX benchmark candidates

Prepared against `feat/hand-gpu-lab` at
`669e7c1af6f381413ac9d819aeae17c6f34c9f9f`. **Additive research slice, not a
validated tracking upgrade or a 60 FPS release. No live defaults change.**

## What is actually provided

| Key | Person-pose input H x W | Output | Candidate purpose |
| --- | --- | --- | --- |
| `dwpose-m` | 256 x 192 | 133 2D points | Lower-compute whole-body candidate |
| `rtmw-m` | 256 x 192 | 133 2D points | RTMW medium comparison |
| `rtmw-l` | 256 x 192 | 133 2D points | Higher-capacity quality/speed comparison |

All use the same official **YOLOX-tiny COCO 416 x 416 person detector**; CPU is
the explicit default and detector CoreML is available only when selected.
Detector, NMS, affine crops, normalization, pose calls and output decoding are
included in frame service time. Detector and pose providers are independently
selected.
All detected people are processed (a declared capacity of eight); more than eight
is an error, not a silently truncated success. This is NOT persistent multi-actor tracking.

133 means 17 body + 6 feet + 68 face + 21 left hand + 21 right hand. There are no
facial blendshape coefficients, MediaPipe-style world landmarks, measured depth,
bone rotations or retargeting in these candidates. The optional native control
still computes the existing full MediaPipe result and is labelled **not feature
equivalent**. Faster 133-point service does not by itself replace full facial capture.

No browser/Node installation is required. ONNX runs inside the Python process.
OpenCV owns video/crop/NMS infrastructure; the small model-specific codecs are
adapted with attribution, rather than importing RTMLib's implicit CPU fallback
or full-image pose fallback on missing detections.

## License and weight gate — read before fetching

Apache-2.0 upstream **code** is not blanket commercial clearance for these exact
checkpoints, their training sources or downstream redistribution. DWPose uses
COCO-WholeBody/UBody; RTMW uses Cocktail14, including datasets with additional
terms. The patch does not redistribute pretrained weights or those datasets.

Every asset and report remains `commercial_release_cleared: false`. The
`--research-only` switch records research intent and acknowledgement of that
uncertainty. **It does not grant permissions, confirm even every research use is
allowed, or resolve rights from dataset terms.** Review the linked publishers'
terms for the planned use; obtain clarification before commercial deployment.
The engine/code can later be used with appropriately licensed/trained assets,
but substituting arbitrary ONNX files is not supported by this fixed catalog.

Official model URLs and training provenance are in `wholebody_catalog.py`.
Download availability and pretrained inference were **not verified in this
execution environment**, which could not download/install the runtimes/models.
Do not treat these sources as fully inspected weight binaries.

No publisher SHA-256 was available during preparation. The explicit `fetch`
command enrolls the first downloaded bytes in `asset-lock.json`: **TOFU**, not
publisher-authenticated hashes. Thereafter every run verifies those exact bytes.
Keep/share the resulting lock metadata when comparing machines. No mirror,
model change, HTTP downgrade or overwrite is attempted on failure.

## Apply this additive patch

Remote push was not available in the current connection. In the existing repo,
start from the named base and preserve your local changes:

```bash
git fetch origin
git switch feat/hand-gpu-lab
git pull --ff-only origin feat/hand-gpu-lab
git switch -c feat/wholebody-onnx-lab
git apply --check "$HOME/Downloads/MotionCapture_wholebody_onnx_lab.patch"
git apply "$HOME/Downloads/MotionCapture_wholebody_onnx_lab.patch"
uv sync
```

The patch adds files only. It does not rewrite `pyproject.toml`, `uv.lock`, the
baseline tracker, the browser lab or user captures. Do not force-reset your tree
if switching fails. The ZIP is a patch-file set, NOT a standalone full repository.

## Runtime setup

Use the repo's Python 3.12 environment, FFmpeg/ffprobe, and optional pinned
`onnxruntime==1.22.1`, `onnx==1.18.0`. The commands below use `uv run
--with-requirements` for optional runtime dependencies without editing the project
lock. These versions are explicit experiment pins, not claims to be the latest.
The pinned macOS wheels require macOS 13+; M1+ arm64 is the hardware target, not a
verified hardware result here. Windows CPU is a target; Windows GPU/WinML/DirectML
and native Apple Vision are NOT implemented in this patch.

```bash
ffprobe -version
uv run --with-requirements tools/requirements-wholebody.txt \
  python -m motioncapture.wholebody_bench catalog

# Explicit first-time model enrollment; review the licensing gate above first.
uv run --with-requirements tools/requirements-wholebody.txt \
  python -m motioncapture.wholebody_bench fetch --research-only
```

The common detector and the three selected pose assets are fetched. On a
corrupt existing asset, execution fails and preserves it; it is not overwritten.
Optional `--models dwpose-m` fetches just that model plus the common detector.

## First comparison: three models on CPU, same 900-frame prefix

```bash
uv run --with-requirements tools/requirements-wholebody.txt \
  python -m motioncapture.wholebody_bench run \
  "$HOME/Downloads/20260906_030954.mp4" \
  --models dwpose-m rtmw-m rtmw-l --providers cpu \
  --native-control --max-frames 900 --research-only \
  --output sessions/wholebody-cpu-smoke.json
```

Plan: native MediaPipe -> DWPose-m CPU -> RTMW-m CPU -> RTMW-l CPU -> native.
`--native-control` needs the existing verified MediaPipe models. Omit it only
when deliberately selecting an ONNX-only diagnostic; do not label it a native
comparison. No GUI/preview composition is included for any arm.

The prefix is explicit; the file reader still verifies the input's original PTS.
Each pass starts at the beginning with a fresh model session. No seeking, frame
interpolation, retained poses, detector skipping or hidden full-image person boxes.
Cold start includes a separately reported zero-tensor provider-placement preflight.
All real frames count in `all_frames`; the first 60 are also separated from steady
statistics, not silently discarded. Pose inference is not called when the person
detector finds nobody, and missing-person counts are reported rather than hidden.

## CoreML comparison

First select a model, then compare against its **own** CPU reference:

```bash
uv run --with-requirements tools/requirements-wholebody.txt \
  python -m motioncapture.wholebody_bench run \
  "$HOME/Downloads/20260906_030954.mp4" \
  --models dwpose-m --providers cpu coreml-all --abba \
  --max-frames 900 --research-only \
  --output sessions/dwpose-coreml-smoke.json
```

This is CPU -> CoreML -> CoreML -> CPU. `coreml-gpu` requests `CPUAndGPU`;
`coreml-ane` requests `CPUAndNeuralEngine`; `coreml-all` requests `ALL`.
None means GPU-only/ANE-only, and selecting it does not prove hardware dispatch.

Default is **strict ORT partitioning**: reject CPU EP nodes in the pose graph,
missing CoreML, provider substitution and zero observed CoreML node executions.
The legacy pose-provider comparison keeps the detector explicitly on CPU; the
separate detector comparison above may explicitly select CoreML for that model.
Python/OpenCV geometry and decoding work remain CPU.
Input symbols are bound to the catalog's fixed shape before compilation.

Some valid CoreML graphs need small CPU partitions. To deliberately test that
mixed mode, add `--allow-cpu-partitions` and use a fresh output filename. There is
no automatic retry into it. Even mixed mode must exhibit actual CoreML node
execution in the preflight profile; a CPU-only run cannot be labelled CoreML.
The raw ORT trace is temporary and discarded after its provider counts are
extracted. It uses a synthetic tensor, not capture data, and does not verify
CoreML's internal per-operator GPU/ANE dispatch.

## Detector versus pose provider comparison

The default detector remains explicit CPU. To compare the detector path without
changing the DWPose pose path, select two detector providers and one fixed pose
provider. `--abba` makes the arm order A/B/B/A:

```bash
uv run --with-requirements tools/requirements-wholebody.txt \
  python -m motioncapture.wholebody_bench run \
  "$HOME/Downloads/20260906_030954.mp4" \
  --models dwpose-m --providers coreml-all \
  --detector-providers cpu coreml-all --allow-cpu-partitions --abba \
  --max-frames 900 --research-only \
  --output sessions/dwpose-detector-coreml-abba.json
```

Both the detector and pose sessions have their own explicit provider metadata
and placement preflight. The report also includes detector prediction hashes,
person-count distributions, pose timing grouped by detected-person count and
per-person pose-call samples. Box comparison uses bounded-RAM greedy IoU
pairing only; raw boxes, actor IDs and landmarks are not written.

Remove `--max-frames 900` for the full clip after initial checks. ABBA reduces
simple order effects but does not control thermals/power/background work. Review
native/CPU controls, per-person work and sustained tails, not only best FPS.

## What the JSON can and cannot establish

- `unpaced_loop_fps`: includes frame processing, hashing, comparisons and stats;
  excludes session setup and final source hashing. **Not live/camera/display FPS**.
- Timings: decoder, detector pre/infer/post, whole-body crop/infer/post, full
  service, 16.67 ms exceedance counts and p50/p95/p99/max. Target 60 sets a budget,
  not an output-rate limiter or interpolator.
- Part counts use an uncalibrated SimCC score threshold (default 0.3). Scores and
  missing rates across different models are **not calibrated accuracy measures**.
- `provider_disagreement`: matching frames and 133-point indices against that
  model's first successful CPU arm; pixel deltas by body/feet/face/left/right hand,
  reference-only/candidate-only point counts and worst frame identifiers. It does
  not pretend to align 17 body points with MediaPipe's 33-point schema.
- Exact hashes include source PTS, geometry, scores and valid masks. Different
  hashes require quality review; identical hashes do not establish ground truth.
  Provider comparison skips frames with multiple people as ambiguous rather than
  guessing identities. Model-to-model/MediaPipe accuracy needs labelled or visual
  evaluation; this patch does not supply a new ground-truth dataset.
- Failure: active arm retains completed-prefix statistics, source frame/PTS,
  phase and cleanup state. Failed arm FPS is null, not successful prefix FPS.
  Each completed/failed arm is also written to a unique `.arm-NN.json` checkpoint.
  A native process abort can still lose the active arm; earlier checkpoints survive.
- No raw frames, facial imagery, audio or individual keypoint coordinates are
  written to reports. References live in bounded RAM for one model at a time.
  Output files/checkpoints are never overwritten.

## Verification scope

See `benchmarks/wholebody-onnx-preparation-verification.json`. Tests use real
OpenCV affine/NMS and FFmpeg/OpenCV VFR decoding, with **explicit synthetic model
and runtime doubles** for inference/provider handling. That is not an actual
pretrained model or CoreML benchmark. Real model fetching, ONNX inference,
Mac/Windows execution, output quality and sustained 60 FPS remain unverified.

```bash
uv run pytest tests/test_wholebody_assets.py tests/test_wholebody_onnx.py \
  tests/test_wholebody_bench.py tests/test_recording.py tests/test_recording_bench.py
uv run ruff check src/motioncapture/wholebody*.py tests/test_wholebody*.py
```

## Recorded whole-body preview

For the later native recorded-preview slice, see
[recorded preview verification](benchmarks/2026-09-08-recorded-preview.md).
The preparation-only verification section above does not describe these later runs.

```bash
uv run --with-requirements tools/requirements-wholebody.txt \
  python -m motioncapture.wholebody_recorded_preview \
  benchmarks/inputs/macbook-720p30-20260905T175716Z.mp4 \
  --provider coreml-all --allow-cpu-partitions --research-only \
  --output sessions/my-recorded-preview.json
```

The window labels file mode and 2D estimates. Esc/Q/window close stops the run.
Each validated frame is sent through a bounded FIFO; slow display does not drop
frames. Add `--snapshot-frame 300` only to explicitly save that annotated frame.
Use a new report name for every run. This path has no persistent actor IDs,
calibrated 3D, retargeting or facial blendshape output.

## Primary research sources

- DWPose models/code: https://github.com/IDEA-Research/DWPose
- RTMW definitions/training sources:
  https://github.com/open-mmlab/mmpose/blob/main/configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw_cocktail14.md
- RTMLib model-specific inference/codecs and model URLs (version pinned in NOTICE):
  https://github.com/Tau-J/rtmlib
- CoreML EP options/support: https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html
- ONNX Runtime 1.22.1 distribution: https://pypi.org/project/onnxruntime/1.22.1/
- YOLOX official ONNX export: https://github.com/Megvii-BaseDetection/YOLOX/blob/main/tools/export_onnx.py

Recorded preview optimization candidates (explicit selection):

```sh
uv run --with-requirements tools/requirements-recorded-preview.txt \
  python -m motioncapture.wholebody_recorded_preview INPUT.mp4 \
  --provider coreml-all --allow-cpu-partitions --research-only \
  --opencv-threads 1 --display-backend sdl \
  --output sessions/UNIQUE-sdl-preview.json
```

`opencv` remains the default display. SDL is an optional pygame-ce window with
no automatic backend substitution. Both paths submit every validated frame.
`--opencv-threads` sets a process-wide OpenCV budget before starting workers;
omitting it retains the runtime default. Schema 2 reports the selected display,
actual driver, OpenCV budget and `display_submit_and_event_pump_ms`; submission
rate and event-return age do not measure physical display refresh or photon latency.
See `benchmarks/2026-09-08-optimization-audit.md` for comparisons and limitations.

Original-model ONNX Runtime 1.29.0 experiment:

```sh
uv run --with-requirements tools/requirements-wholebody-ort129.txt \
  python -m motioncapture.wholebody_recorded_preview INPUT.mp4 \
  --provider coreml-all --allow-cpu-partitions --research-only \
  --expected-ort-version 1.29.0 --opencv-threads 1 \
  --output sessions/UNIQUE-ort129-preview.json
```

The expected version is checked before reading the source or creating models.
Both native model specializations record the actual runtime. The original
requirements and default expected version remain 1.22.1. This is a separately
selected research candidate, not an automatic runtime upgrade or accuracy claim.
