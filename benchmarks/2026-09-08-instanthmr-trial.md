# InstantHMR bounded trial

Scope: explicitly selected offline-file, monocular estimated 3D trial. Compare
the published InstantHMR ONNX on this Mac with existing recorded sources, then
stop. This does not activate a production implementation or additional DWPose
optimization.

Contract: pin source/model revisions and hashes; use unchanged source pixels and
original PTS; run verified YOLOX-tiny on every processed frame; process every
detected person without persistent-ID claims. No stale boxes, skipped frames,
fake per-joint confidence, substituted model or implicit provider retry. Test
CoreML placement separately from performance. CPU-only is a labelled control.
Any hybrid provider experiment must explicitly report CPU partitions.

Authority: upstream MHR70 names, published five-output ONNX schema and original
pixel-based CLIFF conditioning. Camera calibration is unavailable; 3D, camera
translation, and shape are model estimates, not calibrated measurements. No
mesh decoding, bone rotation interpretation, retargeting, multi-camera fusion,
live-camera or screen refresh claim.

Acceptance: report actual runtime/provider placement, finite schema-valid
outputs, stage timing, all-frame counts/hash checks, selected local visual
observations and remaining quality unknowns. Unavailable acceleration or invalid
outputs are recorded failures, never silently converted to a successful run.
Keep raw captures/model assets and experiment artifacts local under ignored
directories. Record only aggregate results here. No commit/push in this trial.

Status: bounded trial completed; candidate not promoted. All three full-file
passes completed, but the approximately 60FPS source accumulated substantial
backlog. A visible-hands failure reproduced on both CPU and CoreML. Production
DWPose code/defaults were not modified; further DWPose optimization is deferred
until the user resumes work.

## Pinned inputs and execution

- Repository: [InstantHMR](https://github.com/mohamdev/InstantHMR), revision
  `dfa7a2daebca3c6d74b692c560546b828f493f62`.
- Published weights: [momolesang/InstantHMR](https://huggingface.co/momolesang/InstantHMR),
  revision `3504446fc31e7f76fdb1cd7e463189e7cf0fdd0f`, `instanthmr.onnx`.
- ONNX SHA-256:
  `f717558094f57d7c9cd084d7981874dbcc3f08e7308501eb98c12844e2cce3b7`.
- Native host: Apple M5, macOS 27.0 arm64; Python 3.12 and ORT 1.29.0. Existing
  isolated optional requirements were used; production dependency pins unchanged.
- Actual graph inputs: float32 `image [batch,3,224,224]` and `cliff_cond [batch,3]`.
  Outputs: `mhr_params [batch,204]`, `shape_params [batch,45]`,
  `cam_trans [batch,3]`, `joints_2d [batch,70,2]`, `joints_3d [batch,70,3]`.
  The published graph has no custom metadata. This trial explicitly uses its
  documented pixel-conditioned crop recipe, not a guessed focal-aware variant.
- MHR70 includes both 21-point hands and foot landmarks. No per-joint confidence
  or visibility output exists. Finite predictions are not labelled reliable or
  visible merely because they exist. MHR vectors remain uninterpreted; no claim
  that all 204 values are retargetable rotations.
- Both detector and pose use strict CoreML ALL / MLProgram with batch fixed to
  one, ORT intra-op threads 4, inter-op 1, spinning disabled. CPU EP fallback is
  disabled, constructor provider substitution rejected, and runtime retry disabled.
- Existing StagePipeline owns detector/pose workers and bounded ready handoff.
  The local experiment substitutes only its per-process pose callback with the
  pinned InstantHMR codec. Detector runs every frame; people are processed serially
  within that frame. OpenCV threads 1. No frame skipping, box reuse, or display.

## Placement and limited standalone controls

Strict CoreML initialization and finite-output preflight passed. The pose graph
ran as one CoreML region: 25 inference calls yielded 25 CoreML execution events
and zero ORT CPU execution events. Detector preflight likewise used one CoreML
region and zero ORT CPU events. Later full-file sessions rechecked placement
before timing and stopped profiling during measured passes.

The CoreML compute-plan log names Apple M5 and contains 849 ANE-preferred and
417 GPU-preferred operation entries. These are planned-device entries, **not**
physical dispatch traces, utilization percentages, or measured device time.

| Standalone control | Mean ms | p95 ms | Scope |
|---|---:|---:|---|
| CoreML, repeated zero tensor | 4.35 | 4.63 | 20 warm samples; profiling enabled |
| CPU, repeated zero tensor | 42.62 | 43.26 | 20 warm samples; profiling enabled |
| CoreML, repeated real phone crop | 12.07 | 15.87 | 50 warm samples; profiling off, detector idle |
| YOLOX, repeated real phone input | 8.41 | 10.77 | 50 warm samples; profiling off, pose idle |

The initial synthetic 4.35ms result did **not** represent real-video inference.
CPU profiling emits many more events than CoreML, so this is not a clean CPU/GPU
speedup ratio. The real-input control also failed to establish that overlap alone
caused the longer pose time: isolated real-input inference was already about 12ms.
Do not diagnose GPU/ANE contention or weak hardware from these numbers.

First strict pose session construction took 33.63s; the first zero inference
took 1.63s. Those cold costs are excluded from warm/pipeline rates and must not
be hidden in a future startup flow.

## Full recorded-source results

All rows include decode, every-frame detection, pose preprocessing, every-person
inference and scalar/hash validation, with detector-next / pose-current overlap.
They exclude GUI display, mesh decoding, retargeting and camera capture. Age is
scheduled original-source time to validated output, not motion-to-photon latency.

| Source / mode | Frames | Pipeline FPS | Pose inference mean ms/frame | Detector inference mean ms/frame | Age p95 ms | Final age ms | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Phone / unpaced | 6442 | 55.54 | 12.77 | 14.61 | n/a | n/a | 1362.0 |
| Phone / original PTS | 6442 | 56.40 | 12.61 | 14.17 | 5681.36 | 5699.90 | 1253.4 |
| MacBook / original PTS | 3599 | 29.99 | 11.13 | 13.11 | 63.98 | 41.84 | 1075.8 |

- Phone source: 1920x1080, original 59.3730FPS, SHA-256
  `6089f23e171d0cc5d4481328f8a476e39a9f1bc74bf1ee17108dd74ad35e6f72`.
- MacBook source: 1280x720, original 29.9917FPS, SHA-256
  `f0346fc4539e6b8f17c67287f6827a3d775459307c3eff515ac574134986414d`.
- 16,483 total frames processed, zero intentional skips or unemitted reads.
  All decoded files reached verified EOF. Detector, pose, and decoder owners
  released after every run. Every emitted model output matched the schema and
  was finite; that is not a quality certificate.
- Phone paced/unpaced input-pixel, ordered-box and raw model-output SHA-256
  digests all matched exactly. Counts in each phone run: 5991 one-person,
  130 two-person, one three-person and 320 zero-person frames. MacBook: 3581
  one-person and 18 two-person frames. These are detections, not stable actor IDs.
- Phone maximum scheduled age was 5722.30ms: it failed to keep up even without
  display. MacBook did not retain a backlog at the end, but age p99 was 450.48ms
  and maximum 657.80ms. A 29.99FPS average does not establish smooth low latency.
- This is not a controlled feature-equivalent MediaPipe/DWPose speed ranking.
  The candidate was not given all DWPose optimizations, and it outputs a different
  schema. No further DWPose optimization was performed in this task.

## Visual observations and a reproduced failure

Eight selected frames used identical source pixels and detector boxes for
InstantHMR and the existing DWPose model: four phone, two MacBook and two public
full-body dance frames. Original pixels stayed unchanged. Side-by-side overlays,
hand details, and orthographic views of estimated 3D joints are saved locally;
no images or raw coordinates are included in this report or Git.

- Public full-body frames 60 and 180: both models broadly followed the visible
  torso/limb configuration. InstantHMR produced plausible-looking 3D skeletons,
  but their depth, scale and hidden limbs have no measured ground truth here.
- Phone frame 1200: visible V-sign hands showed displaced fingertips and incorrect
  bends in the InstantHMR overlay. DWPose also had point errors. These images do
  not establish an aggregate hand-accuracy winner.
- **MacBook frame 1200:** hands are visibly inside the source frame, but
  InstantHMR placed both wrists and all 42 hand points outside the image. With
  the same detector box, DWPose body wrists were in-frame and valid at the
  existing threshold. This is a concrete candidate failure on this sample.

The failure was checked using explicit CPU and strict CoreML on identical
normalized tensors, without saving coordinates:

| Selected frame | Max CPU/CoreML 2D difference | Max 3D joint difference | Wrist result |
|---|---:|---:|---|
| MacBook 1200 | 6.82px | 5.33mm | both wrists outside on BOTH runtimes |
| Phone 1200 | 1.26px | 1.55mm | both wrists inside on BOTH runtimes |

CPU/CoreML outputs are not bit-identical. The observed MacBook hand failure is
not unique to CoreML, so it cannot be explained simply as an Apple GPU-specific
calculation failure. Its deeper cause (checkpoint/data/crop sensitivity) is not
resolved. The public checkpoint and documented recipe were not changed.

No temporal-jitter, occlusion-recovery, calibrated-depth, finger-rotation, or
retargeted-avatar quality claim is made from these selected static frames.

## Outcome and reproducibility

InstantHMR is a runnable portable 3D candidate, but this checkpoint/adapter is
not a demonstrated replacement for DWPose: approximately 60FPS replay failed,
transient stalls remain at 30FPS, and visible-hand failures were found. Keep it
as an explicit research candidate. No default/model replacement is selected.

The machine was shared. WindowServer and other applications were active; a
later snapshot during visual inspection showed another image-viewer test and
Rust compilation. Those observations do not establish their load during all
earlier passes. GPU utilization, frequencies and thermals were not measured.
The hypothesis that the GPU is inherently too weak remains unproven.

Local artifacts are under `sessions/instanthmr-20260908/`: `summary.json`, three
full-pass reports, `model-schema.json`, strict/CPU preflight reports and profiles,
`isolated-real-input.json`, `visual-manifest.json`, `selected-cpu-coreml-check.json`,
and pinned upstream source receipts. `experiment-manifest.json` records script
hashes. These files and model weights remain local/ignored or untracked.

Commands from the repository root, with unused output names/directories:

```sh
uv run --with-requirements tools/requirements-wholebody-ort129.txt python sessions/instanthmr-20260908/trial.py benchmarks/inputs/phone-1080p60-20260906_030954.mp4 phone-new --paced
uv run --with-requirements tools/requirements-wholebody-ort129.txt python sessions/instanthmr-20260908/trial.py benchmarks/inputs/macbook-720p30-20260905T175716Z.mp4 macbook-new --paced
```

Scripts were syntax-checked and exercised with real native inference. No
production source was changed, so the prior 375-test result was not rerun or
represented as new verification. No commit or push. Trial execution is stopped;
wait for the user before the next DWPose optimization task.

## 2026-09-09 speed retry after low-load DWPose control

The user requested rerunning earlier negative benchmarks after low-load DWPose
repeats removed sustained backlog. In the absence of a narrower selection, this
bounded retry starts with InstantHMR speed: full phone unpaced, full phone at
original PTS, and full MacBook at original PTS, using the unchanged `trial.py`
recipe in fresh processes. The exact model SHA was checked before launch.
No quality/mesh/retargeting scope is added and no default or dependency pin is
changed. All prior outputs are preserved under distinct `rebench-20260909-*`
names. A parent samples process-name CPU usage and native thermal/low-power
state every 10 seconds. No application is stopped or assigned a different
priority. Acceptance is unchanged full-frame counts, exact prior input/box/pose
hashes, strict provider placement, terminal cleanup and stage/age timing; a
speed improvement does not overturn the independently reproduced hand error.

### Retry outcome

| Mode | Prior FPS | Retry FPS | Retry final age ms | Peak sampled Blender CPU |
|---|---:|---:|---:|---:|
| rebench-20260909-phone-unpaced | 55.543 | 59.002 | unpaced | 2.0% |
| rebench-20260909-phone-paced | 56.403 | 58.269 | 2037.709639 | 1.9% |
| rebench-20260909-macbook-paced | 29.989 | 29.990 | 36.009583 | 2.1% |

All 16,483 frames completed with exact prior pixel, detector-box and five-output
pose hashes, source EOF verified and model/decoder owners released. No CPU
fallback, video writing or pose-coordinate recording was selected. The parent
monitor and all three fresh child processes exited normally. No production
change, new test claim, commit or push was made.

Phone unpaced throughput improved about 6.2%, and original-PTS throughput about
3.3%. However the original-PTS phone run still ended 2.04 seconds behind, with
p95 2.28 seconds and max 2.33 seconds. Its mean detector/pose inference calls
were 14.18 / 12.23 ms per frame, vs the earlier 14.17 / 12.61 ms. The overlapping
pipeline therefore still lacks sufficient sustained headroom for this source.
These stage means are not additive pipeline latency or physical GPU-time traces.

MacBook 30fps replay kept up: p95 39.72 ms, p99 41.56 ms, max 58.21 ms and final
36.01 ms. This is headless validation latency, not live camera/display latency.
The full-recorded 30fps result is better behaved than the prior max 657.80 ms,
but the earlier independently reproduced hand-position error remains. Identical
output hashes do not establish ground-truth accuracy or fix that error.

Conclusion: low-load testing improves some speed/tail results but does not
reverse the prior rejection of InstantHMR as the DWPose replacement. The earlier
negative speed result cannot all be assigned to Blender. No candidate model or
provider was silently substituted. This retry did not repeat GPU/ANE/SDL policy
arms; its stated selected scope was the InstantHMR speed trial.
