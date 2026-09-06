# r6 policy review and bounded dependency handoff

Parent: 8555a6363e69db7b5439c820e0123569e03562b2.
Decision: keep ALL/ALL as the research reference; do not promote either split
compute policy. Live defaults, models, weights, precision and detector cadence
are unchanged. Performance optimization is NOT complete.

## User-device evidence, not a native benchmark rerun here

Input report: wholebody-optimize-r6-compute-policy.json, SHA256
383bed55ebc14c74b0c0ab41216178b91d886ef939a661639bf851f481bfbdd5.
Six full-file original-PTS runs: A/B/C/C/B/A; 6,442 frames each, no source skips
or clock rebases. Per pass: 5,991 single-person, 130 two-person, one three-person
and 320 empty observations; 6,254 total detected person observations. These are
model observations, not verified actor counts or accuracy.

| Policy (detector / pose) | Pooled mean age ms | p95 each, ms | Maximum each, ms |
| --- | ---: | --- | --- |
| A ALL / ALL | 29.0541 | 34.7872 / 38.2099 | 190.4587 / 227.9419 |
| B CPUAndGPU / CPUAndNeuralEngine | 47.8938 | 112.3263 / 300.2371 | 385.0999 / 400.4314 |
| C CPUAndNeuralEngine / CPUAndGPU | 5984.9597 | 10061.7841 / 12330.8080 | 10520.5227 / 13035.0831 |

Both B runs have worse mean/p95/max source ages than either A control. C fails
to keep up: 54.1638/53.0173 paced FPS and last-frame ages 10.4200/12.9921 seconds.
All runs completed, but completion is not a latency verdict. A/B FPS near 59.35
is source-limited, not a measured new maximum throughput. Percentiles were NOT
pooled by averaging; means/counts were pooled by sample counts.

Detector/pose-call means: A 9.0540/12.1498ms, B 12.7974/10.7570ms,
C 13.9937/16.2537ms. Process CPU means: A 1.0389, B 1.6814, C 1.7754 cores during
the loop. This is host process CPU time, not device power, utilization or physical
GPU/ANE routing. Requested policies permit CPU internally; routing is unverified.

Pixels agree across all six passes. Each policy's two repeats have identical
box and pose hashes. B's boxes exactly match A, but all 6,122 nonempty frames have
some pose-record byte difference (records include scores). Most compared valid
coordinates match, p95/p99 zero, maximum 6.1657px; this is NOT ground-truth quality.
C changes 6,094 ordered-box frames without changing counts. Rare coordinate
mismatches include body 1,640.21px and right hand 672.15px. A is not ground truth,
so those are disagreement, not proven accuracy losses. No outlier is filtered.
Numerical correspondence excludes 131 ambiguous multi-person frames; hashes do not.
The reported source is 8555a plus dirty=true; equivalence to user uncommitted
source bytes is not claimed. 3,543 aggregate consistency checks passed; raw timing
samples were not present, so quantiles were not independently recomputed.

## Changed scheduling boundary

In both A controls, validation averages ~2.083ms/frame. The old pipeline checks
next-detector readiness ONCE before yielding the current result. It was not ready
at 5,518/6,441 and 5,190/6,441 opportunities. If it finishes during validation,
pose cannot start until the consumer resumes. Those counts do NOT measure the
number actually finishing inside that interval; no device saving is inferred.

The new opt-in mode enqueues one continuation on the idle pose owner. It awaits
the ALREADY admitted independent detector Future, validates frame identity and
calls the same pose code on the same native owner. Result validation can proceed
without holding either stage back. The caller still acknowledges the detector
before any next source admission. No extra session, buffer, source frame, model
operation, provider selection, precision change or early result publication.

Python's Future/result primitives are used, not a custom polling scheduler:
https://docs.python.org/3.12/library/concurrent.futures.html
The dependency is pose -> independent detector only, with no reverse or same-pool
wait. A completed detector error is surfaced before borrowing; later failures
propagate through the dependent request. Existing cancellation wakes source waits;
shutdown joins detector before pose. Native inference remains non-force-cancellable.

Source age still includes all waiting. pose_queue_ms retains submission-to-body
semantics, now including dependency waits. Separate pose_owner_dispatch_ms and
pose_dependency_wait_ms distinguish dispatch from waiting on input. No throughput
claim follows from a shorter submission gap. All reference/pixel/box/pose hashes,
observations and final result validation remain enabled in every arm.

## Run the complete comparison

```bash
git fetch origin &&
git switch feat/wholebody-onnx-lab &&
git pull --ff-only origin feat/wholebody-onnx-lab &&
uv run --with-requirements tools/requirements-wholebody.txt \
  python -m motioncapture.wholebody_optimize_bench \
  "$HOME/Downloads/20260906_030954.mp4" \
  --model dwpose-m --detector-provider coreml-all --pose-provider coreml-all \
  --allow-cpu-partitions --research-only --suite dependency-handoff \
  --max-frames 0 --output sessions/wholebody-optimize-r7-dependency-handoff.json
```

A ready-only / B dependent / B dependent / A ready-only, one pose session,
original PTS and complete source every pass. Compare source-age tails, full hashes,
detector-to-pose waiting and process CPU within this JSON. No browser or camera
preview. Keep local changes intact if pull conflicts. Existing result/checkpoint
prefixes are refused, not overwritten. Actual M5 improvement remains pending.
The new mode does not combine with dual pose; old suites remain selectable.

Verification scope is in 2026-09-06-dependency-handoff-verification.json. Event-
controlled tests establish that next pose can start while the result consumer
holds the previous result, even when detection was initially incomplete. Real
OpenCV/FFmpeg VFR and exact all-person output comparisons are exercised, but neural
sessions are explicitly synthetic. No actual ONNX/CoreML speed, camera/display
latency, M1/Windows device performance, ground-truth quality or release approval.
