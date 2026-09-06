# r7 review and controlled pose execution candidates

Parent: 951706d8ef1f8f259a4a3e905edf67044fd84cbe.
Decision: do NOT promote dependency handoff. The next experiment uses ready-only
handoff, one pose model, an ALL detector and every original source frame.
Live defaults and the older explicitly selectable research suites remain unchanged.

## Measured on the user's device, not rerun natively here

Input: wholebody-optimize-r7-dependency-handoff.json. Four complete 6,442-frame
original-PTS passes, A ready / B dependent / B / A. Each has 6,254 person
observations (5,991 single, 130 double, one triple, 320 empty frames). Source,
box and pose hashes agree; all-slot frame hashes include all 131 multi-person
frames, with zero changed frames. Counts do not establish actual actor identity.

| Arm | Mean source age ms | p95 ms | Maximum ms | Above 100ms |
| --- | ---: | ---: | ---: | ---: |
| A1 | 45.7765 | 158.2037 | 397.4029 | 586 |
| B1 | 157.5783 | 807.2049 | 1365.4578 | 1922 |
| B2 | 67.1370 | 314.4162 | 586.6266 | 1014 |
| A2 | 51.3557 | 200.1342 | 325.1842 | 867 |

Pooled means A 48.5661ms / B 112.3577ms; above-100ms counts 1453 / 2936.
Both candidate repeats worsen mean/p95/max versus either control. This is a
rejection for promotion, not proof that all timing variation has one root cause.
Paced FPS ~59.35 is source-limited. Percentiles are not pooled by averaging.

Pose submission gaps decrease 3.0192 -> 0.0915ms, but much of the earlier
submission is waiting for input. Actual detector-complete-to-pose-start wait
averages 0.8982 -> 0.7859ms, a ~0.1123ms difference, not a 3ms latency saving.
Overall pose stage means are similar (13.7884 / 13.7121ms). Transient distributions
and source backlog matter more than these global means. B1's worst source age
is near source sequence 2491 (~41.95s), a one-person frame already released late.
The old worst-age record alone cannot locate the earlier cause of this backlog.
No thermal, memory, GIL or physical GPU/ANE contention cause is claimed.

The report records 951706d plus dirty=true, fingerprint
370c728e6e747aff0761e22f821a35732822c6cf139945dbdaddeb436dc442c8.
Equivalence to all uncommitted user source is not claimed. 2,392 checks of the
provided aggregate records passed. Raw timing samples were not supplied, so the
quantiles are read from the source, not independently reconstructed.

## Implemented next experiment (NOT a measured speedup)

r6 changed detector and pose policies together: its ANE-permitted pose calls
averaged ~10.757ms versus ALL ~12.150ms, but its GPU-permitted detector increased
~9.054 -> 12.797ms. Therefore r6 does not answer whether retaining the ALL
detector while selecting ANE-permitted pose improves the complete pipeline.

`--suite pose-execution` now runs A/B/C/C/B/A, with fresh sessions:
A: detector ALL, pose ALL, pose ORT intra-op threads 4;
B: detector ALL, pose CPUAndNeuralEngine, pose ORT intra-op threads 4;
C: detector ALL, pose CPUAndNeuralEngine, pose ORT intra-op threads 1.
Detector ORT threads remain 4. A/B isolates pose policy; B/C isolates host thread
budget. No model/precision/crop/threshold change, session increase, frame skipping,
pose reuse or unrequested retry. Explicit CPU partitions remain required in this
comparison. The existing low-level adapter and standard ORT thread option are reused.

Official semantics checked:
https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html
https://onnxruntime.ai/docs/performance/tune-performance/threading.html
CPUAndNeuralEngine permits CPU within CoreML. The intra-op budget controls ORT
operator parallelism, not total app/CoreML/GPU/ANE thread counts. Physical dispatch
remains unverified. Changing execution policy or arithmetic scheduling may change
outputs: all-slot hashes, scores, validity and coordinate differences remain visible.
The declared runtime remains ONNX Runtime 1.22.1; no dependency upgrades are added.

FastPrediction was also inspected in ORT v1.22.1 coreml_options.cc and model.mm.
That path is SDK-build-conditional; this slice does NOT enable it or count a mere
accepted string as evidence that a wheel applied CoreML optimization hints.

## New bounded evidence

For this suite only, ReplayAges additionally records one-second ORIGINAL-PTS
windows with detector/pose/decode/verification times, source ages, release lateness
and person counts. The existing 16 worst-age records gain same-frame stage times.
A checked source-age path plus signed observer remainder makes gaps visible;
parent stage and nested model-call times must not be added twice. No raw frames,
model tensors or individual keypoint coordinates are recorded. The extra observer
runs inside and is timed in every arm. Its storage is bounded to 12 scalar samples
per selected frame plus existing metadata; this is not a native RSS measurement.
Legacy 10s age windows and old suites retain their previous semantics.

## Run

```bash
git fetch origin &&
git switch feat/wholebody-onnx-lab &&
git pull --ff-only origin feat/wholebody-onnx-lab &&
uv run --with-requirements tools/requirements-wholebody.txt \
  python -m motioncapture.wholebody_optimize_bench \
  "$HOME/Downloads/20260906_030954.mp4" \
  --model dwpose-m --detector-provider coreml-all --pose-provider coreml-all \
  --allow-cpu-partitions --research-only --suite pose-execution \
  --max-frames 0 --output sessions/wholebody-optimize-r8-pose-execution.json
```

This is six full original-speed file runs, NOT six live-camera tests. Compare
source-age tails, stage windows, per-arm CPU and full output preservation; mean
paced FPS is not a capacity gain. Keep local changes on any pull conflict. Existing
output/checkpoint prefixes are not overwritten. A selected failure stops the plan
with preserved prior reports. Native calls are not force-cancelled by the runner.

See 2026-09-06-pose-execution-verification.json. Targeted tests pass with real
OpenCV/FFmpeg VFR and explicit neural fixtures. Actual M5/CoreML performance,
physical dispatch, camera/display latency, sustained multi-actor operation and
accuracy remain unverified. Performance optimization is NOT complete.
