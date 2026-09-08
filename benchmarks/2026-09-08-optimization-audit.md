# Optimization audit and display experiments — 2026-09-08

Scope: preserve every source frame, original PTS, model/resolution/thresholds,
body/feet/both hands/fingers and exact inference hashes. No commit or push.
Historical results below are the linked conversation's reports and local review
records; they are not new reruns. Current experiments are listed separately.

## Conversation audit

| Method | Evidence and disposition |
|---|---|
| MediaPipe RGB buffer reuse | Implemented and benchmarked; small preprocessing candidate, not a new ONNX improvement |
| MediaPipe serial / parallel / staggered | Tested; staggered did not establish repeatable benefit |
| Browser MediaPipe hands CPU/GPU | Tested; rejected for throughput / timeout limitations |
| DWPose / RTMW CPU model comparison | Tested; DWPose-m selected for later research |
| Pose CoreML, then detector CoreML | Tested with explicit CPU partitions; selected ALL/ALL |
| Static input shape and ORT thread settings | Already active: MLProgram, static shape, sequential ORT, intra-op 4, inter-op 1, spinning disabled |
| NumPy LUT / OpenCV normalization | Tested; OpenCV LUT retained |
| Detector-next / pose-current overlap | Tested and retained |
| Ready-only handoff | Tested and retained |
| Two pose sessions | Tested; mixed outcome, not promoted |
| Detector GPU / pose ANE and reverse | Tested; not promoted |
| Dependency handoff | Tested; worsened timing, not promoted |
| Pose ANE / intra-op 1 | Tested; unstable or mixed, not promoted |
| Same-frame person batch 2 | Tested and retained |
| Original PTS / fixed-60 release | r11 file replay completed; not live capture/display proof |
| ROI reuse / detector cadence reduction | Discussed; not established. Changes the every-frame detection contract and needs its own explicit mode and quality gate |
| Standalone hand CoreML conversion | Discussed; browser route and whole-body model route were explored, not proof this specific conversion was tested |
| FastPrediction | Source examined; build-applied hint and performance not established |
| Physical CoreML compute plan / native graph simplification | Not established by the existing provider-event counts |
| Quantization / pruning / precision changes | Not established; must be separate model candidates with quality evaluation |
| Hardware decode / cross-process pipeline / alternate display backend | Not established in this ONNX preview path |

Earlier optimized MediaPipe rendering already used resize-before-drawing,
vectorized edges and cached labels. The new DWPose renderer resizes first, but
its measured roughly 1 ms composition cost is smaller than the current native
UI event-pump and pose costs. Do not assume rewriting all drawing is the main win.

## Primary-source research

- [ORT CoreML options](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html):
  compute-plan diagnostics, specialization hints, precision and cache options.
  Model caching targets compilation/startup; do not count it as steady inference acceleration.
- [Pinned ORT 1.22.1 implementation](https://github.com/microsoft/onnxruntime/blob/v1.22.1/onnxruntime/core/providers/coreml/model/model.mm):
  optimization hints and compute-plan logging are compile-time gated. Accepted
  option text alone does not prove the wheel applied the hint.
- [OpenCV HighGUI](https://docs.opencv.org/4.x/d7/dfc/group__highgui.html):
  waitKey is not an exact timer; event processing is required.
- [OpenCV Cocoa implementation](https://github.com/opencv/opencv/blob/4.x/modules/highgui/src/window_cocoa.mm)
  includes a 10 ms sleep in its wait loop. The common
  [pollKey implementation](https://github.com/opencv/opencv/blob/4.x/modules/highgui/src/window.cpp)
  can call cvWaitKey(1), so renaming the call is not a justified fix.
- [pygame-ce display](https://pyga.me/docs/ref/display.html),
  [shared image buffer](https://pyga.me/docs/ref/image.html),
  [event ownership](https://pyga.me/docs/ref/event.html): maintained SDL display
  candidate, main-thread synchronous transfer with explicitly selected backend.
- [Apple compression overview](https://apple.github.io/coremltools/docs-guides/source/opt-overview.html):
  quantization, palettization and pruning are candidates, not guaranteed speedups
  or quality-preserving transformations for these models.

## Current experiments

OpenCV defaults to 10 threads on this host. A/B/B/A compares 10/1/1/10 threads,
1200 original phone frames per fresh process, unchanged GUI and inference.
An initial run using only the base environment failed before inference with
ModuleNotFoundError. It is excluded; subsequent runs use the pinned optional
whole-body requirements. Prefix results are not full-video or sustained60 proof.
External CPU work is present on this shared machine; no other process was stopped.
Thermals, clock rates and physical screen timing are not controlled or measured.

### OpenCV thread-budget A/B/B/A

Reports: `sessions/opt-20260908-cv-r2-{a1,b1,b2,a2}.json`.
Every arm completed 1200 validated and submitted frames with matching pixel,
detector and pose SHA-256 across all four arms and released decoder/UI owners.

| Arm | OpenCV threads | UI submissions/s | UI age p95 ms | Final UI age ms | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|
| A1 | 10 | 54.75 | 2026.93 | 1875.67 | 1001.4 |
| B1 | 1 | 59.71 | 511.68 | 46.62 | 1005.8 |
| B2 | 1 | 59.80 | 573.95 | 40.06 | 1003.5 |
| A2 | 10 | 52.81 | 2094.03 | 2661.29 | 992.6 |

Decision: retain explicit `--opencv-threads 1` as the next benchmark candidate.
This improves the tested prefix but does not establish a global optimal thread
count, low startup latency or full-video performance. UI rate excludes the delay
before first submission; a rate slightly above source rate can be backlog catch-up.
The candidate is not automatically made the default.

### OpenCV / SDL display A/B/B/A (OpenCV threads = 1)

Reports: `sessions/opt-20260908-display-{a1,b1,b2,a2}.json`.
All four arms completed 1200 frames; pixel/detector/pose hashes match exactly.
SDL uses pygame-ce 2.5.8 / SDL 2.32.10, Cocoa driver. OpenCV 4.14.0 uses COCOA.

| Arm | Display | UI submissions/s | UI age p95 ms | Final UI age ms | Submit + events mean ms | RSS MiB |
|---|---|---:|---:|---:|---:|---:|
| A1 | OpenCV | 59.78 | 346.79 | 40.88 | 12.96 | 966.3 |
| B1 | SDL | 56.68 | 1266.06 | 1356.45 | 12.35 | 1005.7 |
| B2 | SDL | 56.17 | 1322.53 | 1387.91 | 1.18 | 1009.9 |
| A2 | OpenCV | 48.53 | 4860.93 | 4868.23 | 14.59 | 1022.6 |

Decision: no repeatable end-to-end SDL win established; keep it an explicit
research option. B2 reduced measured display cost but did not keep source pace.
The unchanged A arm varied greatly, and pose/detector durations changed too.
For example pose stage means A1/B1/B2 were 13.82/16.22/16.53 ms. Therefore
neither all latency change nor inference slowdown can be attributed solely to
the GUI backend. Power scheduling, thermals, OS contention and external work
remain possible causes, not proven diagnoses. `pmset -g therm` returned no
recorded warning/status levels; this is not a measured thermal baseline.

The official development documentation advertised pygame-ce 2.5.9, but that
version was unavailable to this resolver. The tested available release 2.5.8 is
pinned in the optional requirements; the core dependency set is unchanged.

### Native partition and compute-plan probe

`opt-20260908-compute-plan.log` and `...-events.json` preserve a synthetic-zero
preflight, NOT steady real-video timings. The pinned wheel emitted compute-plan
logs. Detector plan: 277 GPU operator entries. Pose plans: 222 GPU and 16 CPU
operator entries, no Neural Engine entries. These are preferred-device plans
and estimated costs, not physical dispatch traces or time percentages.
The six ORT CPU kernels are four HardSigmoid and two Split operations; single
cold-probe durations were only 4–26 microseconds each. Seven CoreML regions
remain around them. This motivates investigating partition boundaries; it does
not prove the CPU kernels themselves are the dominant cost.

### Graph-expression candidate contract

Preserve original ONNX asset bytes/checksum and all weights. Derive a separate
research-only ONNX artifact expressing HardSigmoid as Clip(alpha*x+beta, 0, 1),
using each node's actual attributes. This is an algebraic graph transformation,
not a lower-resolution or smaller model. Floating-point execution/fusion can
change outputs, so require ONNX checker, CPU output comparison, native provider
placement and original-video landmark disagreement/hash reporting before any
performance/quality decision. Do not silently substitute the candidate or call
numerical agreement ground-truth accuracy. First acceptance gate is fewer CoreML
partitions with finite valid outputs; real-video equivalence and sustained
performance remain separate gates. Do not change Split without inspecting it.

HardSigmoid probe passed the partition gate: 7 CoreML + 6 ORT CPU regions became
3 CoreML + 2 CPU. CPU outputs on three seeded synthetic normalized tensors were
finite, max absolute difference 4.18e-7 (not bit-exact; not accuracy proof).
The remaining Split nodes have inspected static split lengths [512,512,128] and
[1,1], both axis 2. Extend a second, separately named candidate by expressing
these as constant contiguous Slice outputs, preserving order and lengths.
Check it independently; retain both intermediate and original assets.

### Full-file OpenCV thread-1 follow-up

| Source | Frames | UI submissions/s | Age p95 ms | Final age ms | RSS MiB |
|---|---:|---:|---:|---:|---:|
| phone | 6442 | 44.20 | 34464.60 | 37421.13 | 1075.0 |
| macbook | 3599 | 30.13 | 663.71 | 50.94 | 885.5 |

Both completed, retained all source frames and matched the earlier full FIFO input/detector/pose hashes. The phone run failed to keep pace badly; thread-1 is therefore **not promoted**. MacBook caught up after initial delay, but its p95 is also worse than the earlier run. These are sequential shared-machine runs, not a controlled causal estimate of thread count. AC power was observed.

SDL native smoke: 120 RGB-bar canvases were submitted; RGB backbuffer readback exactly matched the BGR source after channel conversion. A posted Q event raised KeyboardInterrupt and the display owner released. Physical monitor pixels were not measured. Initial test mistakenly requested unsupported BGR readback and failed; corrected RGB readback passed. Observed vsync flag was false, yet the isolated pump averaged 16.32 ms: lack of a requested vsync flag does not imply nonblocking presentation.

### Upstream runtime candidate

[ORT PR 28182](https://github.com/microsoft/onnxruntime/pull/28182), merged April
24, 2026, adds CoreML HardSigmoid support and describes the same DWPose graph-break
problem. An isolated available ORT 1.29.0 preflight on our **original unchanged**
model produced one CoreML region and zero ORT CPU kernels for both batch 1 and 2.
This is a preferable maintenance candidate to carrying custom graph rewrites.
Cold zero-tensor timings are not steady performance results.

Extend the research adapter's exact-version allowlist to 1.22.1 and 1.29.0,
record the actual version in each session, and provide a separately pinned
requirements file for 1.29.0. The original requirements and CLI expected version
remain 1.22.1. Selecting the new preview runtime requires both its isolated
requirements and `--expected-ort-version 1.29.0`; a mismatch must fail before
inference. Show the selected runtime in the preview. No version/provider retry
or production default promotion. Verify both full recorded sources and retain
unverified cross-version numerical/ground-truth quality separately.

### Graph-expression A/B/B/A (unpaced, no display)

`opt-20260908-graph-abba.json`: unchanged source pixels and detector hashes in
all arms, 1200 frames each, original pose A1/A2 hashes match, transformed pose
B1/B2 hashes match. Same weights, threads, provider policy, batch-2 and pipeline.

| Arm | Graph | Pipeline frames/s | Mean pose stage ms |
|---|---|---:|---:|
| A1 | Original, 7 regions | 72.67 | 12.87 |
| B1 | HardSigmoid + Slice, 1 region | 94.25 | 8.45 |
| B2 | HardSigmoid + Slice, 1 region | 88.85 | 8.52 |
| A2 | Original, 7 regions | 45.46 | 20.82 |

Both B runs beat both A runs in this sequence, but A drift prevents a precise
portable speedup claim. This is unpaced inference+validation, not a 94FPS camera
or screen result. On single-person frames matched-coordinate p95 was zero for
all parts; the largest hand delta was 3.90 px. Pose hashes differed across graphs.
Native preflight for the final derived graph passed strict ORT CPU exclusion.
No derived graph replaced the catalog's original model.

### Exact runtime selection and output comparison

Full original-model 1.29.0 OpenCV runs:

| Source | Frames | UI submissions/s | UI age p95 ms | Final UI age ms | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|
| Phone | 6442 | 59.17 | 588.42 | 509.94 | 996.8 |
| MacBook | 3599 | 21.66 | 74703.68 | 46342.33 | 918.5 |

Both completed with matching source pixels and detector hashes versus 1.22.1;
pose hashes differ. Phone pose stage mean was 9.11 ms. MacBook suffered a long
period of UI delay: submit+event p95 114.88 ms, mean 39.54 ms. This is a failed
real-time run despite completing all frames. A runtime speedup alone did not
solve the OpenCV user flow. The cause of long UI stalls is not established;
App Nap/occlusion/OS contention remain hypotheses, not diagnosed facts.

A separate 1200-frame cross-runtime comparison used each version in its own
process and transferred poses through owned pipes into transient parent RAM.
No coordinates/images were written. All 1200 frames had byte-equal ordered
boxes; 68 contained multiple detections, allowing within-frame slot comparisons
without claiming persistent actor IDs. Across 118955 valid joint observations,
only five moved over 1 px (one body, two left-hand, two right-hand), max 3.8958 px.
Every part's p95/p99 delta was zero; no validity classification changed. Maximum
absolute score delta was 1.193e-6. This is output disagreement, not ground-truth
accuracy or a full-video quality certificate. See `opt-20260908-runtime-quality.json`.

Exact experiment scripts and SHA-256 receipts are preserved under
`sessions/opt-20260908-experiment-scripts/`. They use fixed immutable output names;
choose new stems before repeating. Failed setup attempts remain separate from
completed reports. Original model SHA-256 remains
`94ca58fa2d6c4530b6957ac9548084ebc2fa27ed71e4e01f0b73844306ed01a6`.

### Full 1.29.0 + SDL follow-up

| Source | Frames | UI submissions/s | Age p95 ms | Final age ms | RSS MiB |
|---|---:|---:|---:|---:|---:|
| phone | 6442 | 22.74 | 173941.12 | 175075.33 | 1043.6 |
| macbook | 3599 | 31.17 | 44916.78 | 17950.93 | 946.1 |

Both retained all frames and exactly matched the same-runtime OpenCV input/detector/pose hashes. Neither is a low-latency success. MacBook submission rate excludes its first-submission delay and therefore does not show its severe backlog. Phone maximum pose stage was 24.82 s, decode read 24.80 s, and detector stage 18.23 s; UI submit/event p95 was only 6.84 ms. Thus SDL display cost alone cannot explain that failure. Other application/build CPU work was observed during the run (single snapshot in `opt-20260908-shared-host-observation.json`). A later memory query reported 47% system-wide memory free; cumulative VM counters do not establish pressure during earlier stalls. The stall root cause remains unknown.

## Outcome and next priorities

- The prior conversation did not exhaust runtime/operator-level work. This run established a concrete unsupported-operator boundary and a maintained upstream solution.
- Prefer continued testing of the original model on explicitly selected ORT 1.29.0 over maintaining graph-expression derivatives. It has verified batch-1/batch-2 provider placement and bounded cross-version output comparisons. It is not the production default.
- Do not promote thread-1 or SDL based on a short prefix. Full-file failures are retained above.
- Next, profile long application stalls with host activity and window state recorded. Repeat comparisons under a controlled workload; do not stop user-owned work to manufacture clean results. Separate headless model timing from display submission and photon timing.
- Revisit ALL/GPU/ANE policy on the now-contiguous 1.29.0 graph: prior policy experiments used the fragmented 1.22.1 graph. FastPrediction build/application proof and precision/compression quality gates remain separate candidates. Hardware decode and allocation changes need measured stage costs first.
- Preserve model resolution, every source frame, body/feet/both hands/fingers and original PTS. ROI reuse/cadence changes would require a separate explicit contract.

## Verification

- 375 tests passed, two optional browser E2E tests skipped. Ruff across src/tests/tools and git diff whitespace checks passed.
- Real ONNX graph checks, CPU numerical probes, native CoreML partition/compute-plan probes, A/B/B/A file benchmarks and full-file GUI runs are recorded above with their failures.
- Native ORT1.29 + SDL quit flow: posted Q after the third display submission; process exit 130, status interrupted, three submitted/four validated/five read, zero replacements, inference owner joined, detector/pose/decoder/UI owners released. This was a programmatic SDL event, not a physical keypress.
- No raw frames or coordinates were saved by these experiments. Existing private inputs, original model and earlier user edits were preserved. No commit or push.
