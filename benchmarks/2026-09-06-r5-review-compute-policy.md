# r5 review and explicit compute-policy candidates

Parent: ca29fee08106b58abb7f5301983bb83dc01dfe2b.
Decision: retain the single-pose reference; do not promote dual pose globally.
Dual-pose remains an explicit research mode, not removed or silently substituted.

## User-device r5 evidence

Source: wholebody-optimize-r5-pose-parallel.json. Four full 6,442-frame original-PTS
replays, A single pose / B dual pose / B / A. Every pass observed 5,991 single,
130 double, one triple and 320 empty frames (6,254 person observations). These
are detector outputs, not verified actor counts. Source/box/pose hashes match.
Per-frame hashes include all 131 multi-person frames, with zero changed frames.

| Arm | Mean age ms | p95 age ms | Maximum age ms | Frames above 100ms |
| --- | ---: | ---: | ---: | ---: |
| A1 | 36.9711 | 81.5144 | 504.4808 | 299 |
| B1 | 44.0887 | 158.9299 | 468.0463 | 445 |
| B2 | 28.2339 | 43.9303 | 114.2951 | 16 |
| A2 | 34.5954 | 93.7784 | 419.2195 | 294 |

Pooled source-age means: A 35.7832ms, B 36.1613ms. Above-100ms counts improve
593 -> 461, but above-50ms counts worsen 863 -> 1102 and above-33.33ms counts
1235 -> 2415. These thresholds are diagnostics, NOT declared latency acceptance
limits. Aggregate percentiles are not reconstructed or pooled from arm quantiles.
B2 is promising but does not erase B1. Both B arms are needed in the decision.
Paced throughput near 59.35 FPS is input-limited, not a measured capacity increase.

Single-person pose-call mean worsens 11.8972 -> 12.9763ms and detector inference
8.7025 -> 9.4803ms. Two-person-ending output intervals shrink about 25.34 ->19.02ms,
but interval attribution is not isolated pose wall time. Existing frame_pose_ms
is a SUM of potentially overlapping model calls, not elapsed latency; it cannot
by itself prove the dual mode slower. Additional elapsed-time stratification is
included below. Device state, driver routing and memory costs remain unmeasured.
The r5 log records ca29fee + dirty=true, not a verified clean source-only run.

## What changed

Add --suite compute-policy, a single-pose-session A/B/C/C/B/A comparison:
A detector ALL, pose ALL; B detector CPUAndGPU, pose CPUAndNeuralEngine;
C the reverse. This tests permitted compute policies without increasing pose
session count, changing model/precision, reducing input, skipping detections or
reusing outputs. It is NOT proof of physical GPU/ANE separation or a measured
speedup. The inference adapter already supported these policies; the new suite
makes the complete comparison reproducible with one command and correct labels.

Official runtime semantics reviewed:
https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html
MLComputeUnits CPUAndGPU/CPUAndNeuralEngine still permit CPU; ALL permits all.
No new low-precision, specialized compilation, cache or undocumented flags are
introduced. OrtModel's existing spin-disabled CPU pools remain unchanged.

ExecutionArm is the one owner for per-arm mode/provider selection. The manifest,
started checkpoints, constructors and terminal rows derive from it. Conflicting
CLI provider overrides are rejected for this fixed suite. A requested policy
failure stops with a failed checkpoint; there is no ALL/CPU retry. Session
metadata must match the requested policy before measured frames are admitted.
Legacy suites retain their previous modes/provider policies and live defaults.

Add actual elapsed pose-stage and detector-stage times by person count while
retaining and labelling the old summed-call fields. Add process CPU time for the
measured loop (all host threads, not GPU time or power). Add per-frame ordered
box hashes alongside existing all-slot pose hashes; box-order changes are visible
but are not interpreted as real actor changes. No private imagery, tensors or
coordinates are written. Extra observer work stays inside every arm's loop.
Cross-policy point agreement is NOT accuracy and is no longer labelled as a
same-provider comparison. Prior sparse crop-sensitive pose outliers remain open.

## Run

```bash
git fetch origin &&
git switch feat/wholebody-onnx-lab &&
git pull --ff-only origin feat/wholebody-onnx-lab &&
uv run --with-requirements tools/requirements-wholebody.txt \
  python -m motioncapture.wholebody_optimize_bench \
  "$HOME/Downloads/20260906_030954.mp4" \
  --model dwpose-m --detector-provider coreml-all --pose-provider coreml-all \
  --allow-cpu-partitions --research-only --suite compute-policy \
  --max-frames 0 --output sessions/wholebody-optimize-r6-compute-policy.json
```

Every arm processes the full source with original PTS pacing, no preview or
native camera. Inspect source-age tails/windows, elapsed-stage stratification,
process CPU, box/pose disagreements and cleanup, not maximum paced FPS. Fresh
sessions and reversed order reduce simple order bias, not all hardware-state
confounding. Preserve local changes on pull conflicts. Existing outputs and
checkpoint prefixes are refused rather than overwritten. On native abort/hang,
prior checkpoints survive; the current native call is not force-cancelled.

Validation scope is recorded in 2026-09-06-compute-policy-verification.json.
Actual M5/CoreML policy performance, physical dispatch, memory, camera/display
latency, tracking accuracy and commercial clearance remain unverified. This is
NOT a declaration that all performance optimization is complete.
