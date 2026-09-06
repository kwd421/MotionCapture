# r8 review and same-frame batch-2 feasibility

Parent: 4142f9ff8acf68fa6a9a8fcd502c6142ba363397.
Decision: retain ready-handoff detector ALL / pose ALL / pose ORT threads 4 as
the research reference. Do not promote the ANE-permitted or 1-thread pose arms.

The supplied r8 report completed six 6,442-frame original-PTS passes. Source-age
results (mean / p95 / max ms): A ALL/4 = 47.43/158.32/645.18 then
30.12/45.60/191.42; B ANE-permitted/4 = 29.97/39.78/171.18 then
750.90/3395.17/3604.03; C ANE-permitted/1 = 90.18/396.20/688.86 then
30.75/55.63/156.82. The candidate repeats are not stable enough for promotion.
B2 ends about 3.08 s behind. All runs preserve source pixels and detector-box
hashes; pose hashes differ by execution policy. A is not ground truth.

The one-second stage trace shows backlog can occur in single-person regions too;
therefore the next experiment is deliberately narrow. The source still contains
130 two-box frames and one three-box frame per pass. Existing serial two-person
pose work is a burst cost, but prior dual-session parallelism was not repeatable.

## Added lab

`python -m motioncapture.wholebody_pose_batch_lab` tests one different mechanism:
use the pose export's symbolic batch dimension to process two detector slots from
the SAME frame in one native batch-2 model call. Discovery still runs the detector
on every original frame. The lab never batches different source timestamps.

The batch-2 session uses the same model file, CoreML ALL policy, explicit CPU
partitions, precision and crop recipe as batch 1. It binds symbolic batch=2 and
requires preflight outputs `(2,133,384)` and `(2,133,512)` for DWPose-m. There is
no fallback from a failed batch-2 call to two serial calls.

This commit is a feasibility lab only. It does not alter the live/default path or
claim a measured M5 speedup. Target-device output JSON is required before deciding
whether same-frame batching should be integrated into StagePipeline.

## Run

```bash
git fetch origin &&
git switch feat/wholebody-onnx-lab &&
git pull --ff-only origin feat/wholebody-onnx-lab &&
uv run --with-requirements tools/requirements-wholebody.txt \
  python -m motioncapture.wholebody_pose_batch_lab \
  "$HOME/Downloads/20260906_030954.mp4" \
  --detector-provider coreml-all --pose-provider coreml-all \
  --allow-cpu-partitions --research-only \
  --output sessions/wholebody-pose-batch-r9.json
```

The result is an isolated pair-kernel comparison, not a full pipeline FPS or
camera/display latency result. Preserve local changes if pulling conflicts.
