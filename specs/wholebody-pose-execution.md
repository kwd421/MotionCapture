# Controlled pose execution and source-time stage evidence

Definition: compare only the pose execution policy and its ORT host thread budget
while retaining the measured ALL detector, one pose owner, and ready-only handoff.
Parent: 951706d8ef1f8f259a4a3e905edf67044fd84cbe. This is an opt-in file experiment.

## Contract

The fixed `pose-execution` plan is A/B/C/C/B/A:
A detector ALL / pose ALL / pose intra-op threads 4;
B detector ALL / pose CPUAndNeuralEngine / pose intra-op threads 4;
C detector ALL / pose CPUAndNeuralEngine / pose intra-op threads 1.
The detector's ORT thread count stays 4 in every arm. A->B changes one policy;
B->C changes one host thread budget. Do not attribute A->C to only one change.
Conflicting explicit providers, --ort-threads or boundary diagnostics are rejected.
The immutable ExecutionArm owns the override; factories, metadata checks, started
and terminal reports derive from that record. Legacy arms inherit --ort-threads.

Every original PTS/frame and every detected person is processed, including empty
observations. Models, preprocessing, float32 input, thresholds, original schedule,
full pixel/box/pose validation, output ordering, admission and cleanup remain the
same. No new model sessions or custom scheduling. The unsuccessful dependent
handoff and split-detector suites remain explicitly selectable, never defaults.
CPUAndNeuralEngine still permits CoreML CPU execution; ORT intra-op threads do not
control CoreML/ANE/GPU internal thread pools or the total process thread count.
No claim of physical ANE dispatch, accuracy, live camera/display or commercial use.

## Evidence and acceptance

r7 shrank submission gaps but worsened source-age tails. r6 changed detector and
pose policy together; its faster ANE-permitted pose was confounded by a slower
GPU-permitted detector. This slice tests the previously unmeasured ALL/ANE pair
and then whether a smaller CPU pool helps the pose graph's explicit CPU partitions.
Neither speedup nor output identity is assumed; both need target-device evidence.

ReplayAges optionally owns one-second original-PTS stage windows and correlated
stage times on its existing 16 worst-age records. These contain scalar timings,
workload counts and frame IDs only. Existing 10-second age summaries remain intact.
The same-frame source-age decomposition is checked before counters mutate; the
signed unattributed remainder includes observer gaps, not invented model time.
One-second statistics are correlations, not hardware attribution or causal proof.
Sample storage is bounded by the selected frame limit; no pixels/poses are added.
The extra observer's cost remains inside every measured arm and is timed.

Acceptance requires all-frame/count/hash/PTS/cleanup contracts; compare source-age
tails, full and per-frame output differences, source-time stage bursts and host CPU
across both repeats. Completed/failed/partial and unmatched data remain distinct.
Raw timing samples are not implied by summaries. No benchmark silently approves
an SLA, ground-truth quality or sustained 60Hz. Tests use event-controlled real
workers, real FFmpeg/OpenCV VFR, explicit neural fixtures, and fixed-clock stats.
