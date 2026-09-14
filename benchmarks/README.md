# Experiment record index

Audited on 2026-09-14 against the current branch history and the recorded
experiments in this conversation. This index locates evidence; it does not
upgrade a reported, failed, incomplete or untested method into a verified one.
The linked records retain their own measurement scope and limitations.

## Coverage

| Method family | Record |
|---|---|
| M5 CPU scheduling; MediaPipe rendering and RGB-buffer reuse | [CPU scheduling](2026-08-30-m5-cpu-task-scheduling.md), [live performance](2026-09-06-live-performance.md), [input validation](2026-09-06-recording-inputs.md) |
| Serial, parallel and staggered task scheduling; original-PTS and 60fps file workload | [Recording60 scheduling](2026-09-06-recording60-scheduling.md) |
| Browser MediaPipe hands CPU/GPU and hybrid backend trial | [Hand backends](2026-09-06-hand-backends.md) |
| DWPose/RTMW ONNX candidates, detector/pose acceleration, exact normalization and stage overlap | [Benchmark guide](../README_WHOLEBODY_BENCH.md), [method audit](2026-09-08-optimization-audit.md) |
| Ready-only handoff and NumPy/OpenCV native LUT | [Ready handoff](2026-09-06-ready-handoff.md), [r2](2026-09-06-r2-review-native-lut.md) |
| Original-PTS pacing and full-file backlog/cadence | [r3](2026-09-06-r3-review-pts-replay.md) |
| Two pose lanes; GPU/ANE policies; dependency handoff; pose execution/thread budget | [r4](2026-09-06-r4-review-pose-lanes.md), [r5](2026-09-06-r5-review-compute-policy.md), [r6](2026-09-06-r6-review-dependency-handoff.md), [r7](2026-09-06-r7-review-pose-execution.md) |
| Same-frame person batch 2 and full-pipeline integration | [r8](2026-09-07-r8-review-pose-batch-lab.md), [r9](2026-09-07-r9-review-full-pipeline-batch.md) |
| Fixed-60 scheduling and full recorded-input gate | [r10](2026-09-07-r10-review-fixed60-batch.md), [r11](2026-09-07-r11-review-fixed60.md) |
| OpenCV preview, bounded FIFO, explicit SDL, OpenCV thread budget | [Preview](2026-09-08-recorded-preview.md), [optimization audit](2026-09-08-optimization-audit.md) |
| CoreML partition/compute-plan probes; HardSigmoid/Split graph rewrites; ORT 1.22.1 vs 1.29.0 and matched-output checks | [Optimization audit](2026-09-08-optimization-audit.md) |
| ORT 1.29 GPU/ANE pairs; window/thermal observations; macOS activity; SDL duplicate libraries; headless/display ABBA under changing and low load | [DWPose follow-up and retries](2026-09-08-dwpose-runtime129-followup.md) |
| InstantHMR schema/placement, CPU/CoreML and synthetic/real-input controls, full-file speed, selected hand-quality failure and low-load retry | [InstantHMR trial and retry](2026-09-08-instanthmr-trial.md) |

The method audit also distinguishes proposals from executed tests. ROI reuse,
reduced detector cadence, quantization/pruning, applied FastPrediction hints,
and hardware decoding must not be labelled successfully tested merely because
they were discussed. Later dated sections supersede earlier provisional
performance interpretations; shared-host contention limits comparisons.

## Git coverage and local-only evidence

All method families above have a committed implementation and/or experiment
record after this documentation commit. There is not one separate commit per
experimental arm. The two 2026-09-08 follow-up reports include their 2026-09-09
retries and are added together with this index. Historical statements such as
"no commit/push" describe the experiment turn, not the later archival commit.

A later user-authorized archive now preserves **463 text artifacts** under
[`sessions/archive-20260914/`](../sessions/archive-20260914/README.md): temporary
probe/controller scripts, per-run JSON, native profiles and logs. Its manifest
records original and archived hashes and path redactions. Originals remain
untouched locally. Images/crops/plots, bytecode and filesystem metadata (42 files)
are excluded, along with private input videos and model binaries outside sessions.
A clean clone has the archived text evidence, but not every input required to
rerun it; historical paths/dependencies may still need adaptation.

No fresh performance run or source test run was performed for this documentation
and archival work. History, links, JSON/JSONL/Python parsing and archive hashes
were checked; this is not new native-performance or accuracy verification.

## Record-to-commit lookup

The commit column is the latest existing change to each Markdown record at the
2026-09-14 audit, or "this documentation commit" for the newly added reports.
Use `git log --follow -- benchmarks/FILENAME.md` for complete history. This
column does not imply that each native measurement was performed by that commit.

| Record | Existing record commit |
|---|---|
| [2026-08-30-m5-cpu-task-scheduling](2026-08-30-m5-cpu-task-scheduling.md) | 9a2608a |
| [2026-09-06-hand-backends](2026-09-06-hand-backends.md) | 834cc85 |
| [2026-09-06-live-performance](2026-09-06-live-performance.md) | dd92ae0 |
| [2026-09-06-r2-review-native-lut](2026-09-06-r2-review-native-lut.md) | fa357ff |
| [2026-09-06-r3-review-pts-replay](2026-09-06-r3-review-pts-replay.md) | f376d02 |
| [2026-09-06-r4-review-pose-lanes](2026-09-06-r4-review-pose-lanes.md) | ca29fee |
| [2026-09-06-r5-review-compute-policy](2026-09-06-r5-review-compute-policy.md) | 8555a63 |
| [2026-09-06-r6-review-dependency-handoff](2026-09-06-r6-review-dependency-handoff.md) | 951706d |
| [2026-09-06-r7-review-pose-execution](2026-09-06-r7-review-pose-execution.md) | 4142f9f |
| [2026-09-06-ready-handoff](2026-09-06-ready-handoff.md) | ec5103a |
| [2026-09-06-recording-inputs](2026-09-06-recording-inputs.md) | 4091f17 |
| [2026-09-06-recording60-scheduling](2026-09-06-recording60-scheduling.md) | d77c332 |
| [2026-09-07-r10-review-fixed60-batch](2026-09-07-r10-review-fixed60-batch.md) | b4bb4b7 |
| [2026-09-07-r11-review-fixed60](2026-09-07-r11-review-fixed60.md) | e02ba18 |
| [2026-09-07-r8-review-pose-batch-lab](2026-09-07-r8-review-pose-batch-lab.md) | a7b3d99 |
| [2026-09-07-r9-review-full-pipeline-batch](2026-09-07-r9-review-full-pipeline-batch.md) | 261fed4 |
| [2026-09-08-dwpose-runtime129-followup](2026-09-08-dwpose-runtime129-followup.md) | this documentation commit |
| [2026-09-08-instanthmr-trial](2026-09-08-instanthmr-trial.md) | this documentation commit |
| [2026-09-08-optimization-audit](2026-09-08-optimization-audit.md) | 451740f |
| [2026-09-08-recorded-preview](2026-09-08-recorded-preview.md) | 451740f |
| [wholebody-model-license-review](wholebody-model-license-review.md) | dfb7b12 |
