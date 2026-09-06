# r1 review and ready-only handoff candidate

Base commit: `971dad890128a542774deb07d9a79d3286042c49`.
This changes scheduling only; no live default, model, provider, precision,
preprocessing or observation policy is promoted.

## Observed on the user's device (NOT a local native rerun)

Reviewed completed `wholebody-optimize-r1(1).json`, SHA256
`244ffb84bf464d1eeabb52e017727cb333a5993fe32a7dfd16e2e8d68f9a68b4`.
The other uploaded JSON failed at input inspection with FileNotFoundError and
zero runs; it is not a failed inference-speed measurement.

Each mode was measured twice, 900 original frames per pass; 968 person observations
per pass. Pooled FPS = sum(frames)/sum(loop_s), not the arithmetic mean of FPS.

| Mode | Pooled FPS | Mean host submit-to-pose ms | Mean validation ms |
| --- | ---: | ---: | ---: |
| sequential-reference | 37.4012 | 23.3294 | 2.3709 |
| sequential-lut | 36.6763 | 23.8193 | 2.4017 |
| overlap-lut | 50.8166 | 35.9739 | 2.1978 |

Overlap is +35.8688% vs reference throughput, but has greater host latency.
The LUT-only full-loop speed gain is NOT established. All six prediction,
detector and source-pixel hashes match. Numerical comparisons exclude 68 ambiguous
multi-person frames; full-stream hashing includes those frames. This is still a
900-frame, unpaced, 2D file test without presentation, not sustained live 60Hz.

### Boundary findings

All 22 selected diagnostic frames have byte-identical repeated same-input pose
outputs. Eight have identical crops across detector providers. In the other 14,
some uint8 pixel CHANNELS differ by at most one. Frame 253 differs in 77/147456
channels; SimCC argmax changes by up to 53 horizontal bins and a valid foot differs
by 243.7295 source pixels. Holding model outputs fixed and changing only coordinate
restoration gives at most 0.000535925 pixels across the 22 probes.

This local counterfactual supports an input/crop-sensitivity and argmax-switching
mechanism, not a concurrent-buffer corruption diagnosis. It does not establish
which estimate is correct or fix model robustness. Large provider-induced pose
outliers remain OPEN. No quantization, filtering or validity suppression was added.

## Implemented change

The existing generator yields each result while the pose owner is idle. The
consumer then spends about 2.2 ms on hashing/comparison before requesting the next
packet. The candidate consumes an ALREADY-completed next detection and submits
its pose before yielding the current packet. It does not wait for a slow next
detection, change the one-request/stage cap, read extra prefix frames or remove
any pixel/result verification. No new worker is introduced.

`pose_submit_gap_ms` measures the gap between consecutive pose jobs. The first
frame has no predecessor and contributes no gap sample. The new
`verified_output_interval_ms` likewise has N-1 samples, not a made-up zero first
sample. Neither is sensor-to-display latency. Counters expose successful ready
handoffs and next detections that were not yet ready. An early consumer stop may
leave one admitted pose unpublished, explicitly counted.

## Run the new A/B/B/A comparison

```bash
git fetch origin
git switch feat/wholebody-onnx-lab
git pull --ff-only origin feat/wholebody-onnx-lab

uv run --with-requirements tools/requirements-wholebody.txt \
  python -m motioncapture.wholebody_optimize_bench \
  "$HOME/Downloads/20260906_030954.mp4" \
  --detector-provider coreml-all --pose-provider coreml-all \
  --allow-cpu-partitions --research-only --suite handoff \
  --max-frames 900 --output sessions/wholebody-optimize-r2.json
```

Plan: overlap-LUT / overlap-ready-LUT / overlap-ready-LUT / overlap-LUT.
The original six-arm suite remains unchanged. Full per-pixel hashing is retained
in every arm. Completed diagnostics need not be repeated for this schedule-only
comparison. Existing output/checkpoint paths are never overwritten. Preserve local
changes on a Git conflict; do not force reset.

## Validation scope

See the adjacent verification JSON. Tests exercise actual stage owners, OpenCV
preprocessing, FFmpeg VFR decoding, whole runner ABBA, hashing and failure paths.
Neural sessions in local tests are explicit synthetic doubles. No local M5/CoreML
speed, physical GPU/ANE dispatch, new quality improvement, commercial clearance
or optimization-complete claim is made. Native candidate benchmarking is pending.
