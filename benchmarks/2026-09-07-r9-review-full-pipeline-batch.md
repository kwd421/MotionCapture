# r9 batch-2 feasibility and full-pipeline gate

Source: `wholebody-pose-batch-r9.json`, run on the user's M5 environment from
`a7b3d994e04d7ba0e4ec4ed37ebd585d064ca6d3` with a dirty working tree. The
report completed a full 6,442-frame detector scan and found 131 frames with at
least two detected slots. This is detector output, not verified multi-actor
identity. No raw frames, tensors or coordinates were written.

For the first two slots of those 131 frames, A/B/B/A compared two batch-1 pose
calls with one batch-2 pose call. Mean isolated pair time was 15.3847ms serial
versus 10.7968ms batch-2, a 1.4249x ratio. p95 was 30.8548ms versus 22.5688ms.
This is an isolated pose-kernel result: detector, crop preparation and the real
paced pipeline backlog were not inside the timed pair region.

Batch-2 did not reproduce batch-1 numerics bit-for-bit. Across compared valid
points there were eight >1px differences in total (body 2, feet 4, left hand 2),
no >10px differences, and no validity-only changes. Maximum observed coordinate
difference was 6.2436px. Face and right-hand matched coordinates were identical
in this run. These are disagreements, not ground-truth accuracy results.

Decision: do not promote batching to a default yet. Add a full original-PTS
A/B/B/A lab that keeps detector=CoreML ALL, pose=CoreML ALL, one pose owner,
OpenCV LUT preprocessing, every source frame, every detected slot, ready-only
handoff and full verification. A uses the established serial pose session. B
owns batch-1 and batch-2 specializations on the same pose worker and routes only
adjacent slots from the SAME frame through batch-2; an odd remainder uses batch-1.
No future-frame batching, padding, frame drops, result reuse or batch failure
fallback is permitted.

Full-pipeline acceptance is evidence-driven: compare source-age mean/tails and
100ms exceedances in both B repeats against both A controls, require identical
source pixels and detector boxes, and verify any prediction changes are confined
to multi-person frames. The earlier r9 coordinate deltas remain the quality
context; batching is not accuracy-approved. Additional native session memory and
camera/display latency remain unmeasured.

Run:

```bash
git fetch origin &&
git switch feat/wholebody-onnx-lab &&
git pull --ff-only origin feat/wholebody-onnx-lab &&
uv run --with-requirements tools/requirements-wholebody.txt \
  python -m motioncapture.wholebody_pose_batch_pipeline_lab \
  "$HOME/Downloads/20260906_030954.mp4" \
  --detector-provider coreml-all --pose-provider coreml-all \
  --allow-cpu-partitions --research-only --max-frames 0 \
  --output sessions/wholebody-pose-batch-pipeline-r10.json
```

This remains an offline recorded-input performance experiment, not proof of
live camera 60Hz, physical GPU/ANE dispatch, 3D mocap, retargeting or release
readiness.
