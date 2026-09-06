# Full-file capacity result and source-PTS replay acceptance

Base commit: fa357ff0a1768abe6715a0931170b24a61ba776d.
No live default, model, provider, precision, detector cadence or geometry is changed.

## Observed r3 on the user's device (not rerun here)

Reviewed `wholebody-optimize-r3-full.json`, SHA256
`c323d923d2e821dc2e8a9667a49b0154a59db7b96a9b3bf1be579f7daab129ea`.
Four complete passes each processed all 6,442 original frames. The source is
108.5003 seconds, 1920x1080 HEVC, PTS-span rate 59.3730053Hz. Output/source/box
hashes are identical across all four arms. The report records fa357ff plus a dirty
working tree; it is not evidence of a clean commit-only execution.

| Ready-handoff normalization | FPS per pass | Pooled FPS | Pose preparation mean ms |
| --- | --- | ---: | ---: |
| NumPy | 78.3492 / 72.7310 | 75.4356 | 0.560017 |
| OpenCV LUT | 82.1143 / 79.2898 | 80.6774 | 0.291950 |

Pooled rates use total frames / total loop seconds. Observed throughput gain is
6.95%; preparation time falls 47.87%. Native detector/pose times also vary, so not
all of the full-loop difference can be attributed to normalization alone. Do not
compare the earlier 900-frame rate to the full-file rate as a pure patch speedup.

The finite full-file, predominantly-one-person 60FPS capacity milestone is met.
OpenCV arms have verified-interval p95 14.9904 / 15.5477ms and p99 19.6065 /
23.9238ms; 286/12,882 intervals exceed 16.67ms (2.22%), NOT dropped frames or live
deadline misses. All source-time bins exceed 60Hz; the slower candidate's minimum
is 67.8895Hz. These are host completion rates within source bins, not paced capture.

Per pass: 5,991 one-person, 130 two-person, one three-person and 320 zero-person
frames (6,254 person calls). The zero-person frames still run detection but not
pose; their cause is not established by aggregate logs. Of those, 253 are in the
60-70s source bin. Identity/ground truth is absent, so extra boxes are not proof
of additional real actors either. Two-person ending intervals average 18.9252 /
24.0099ms in the candidate; sustained multi-person 60Hz is NOT established.

Numerical point matching excludes 131 ambiguous multi-person frames, while full
prediction hashing includes their outputs. Matching outputs is not accuracy, nor
a fix for the previously diagnosed CPU/CoreML detector crop-sensitivity outliers.

## Decision and implemented acceptance mode

Retain OpenCV-LUT/ready handoff as the file-engine performance candidate. Rather
than add another micro-optimization, test the unchanged engine against original
source arrival times. `--suite paced` performs unpaced / original-PTS-paced /
original-PTS-paced / unpaced, all with the same OpenCV kernel and fresh sessions.

Release uses one fixed host epoch plus the absolute rational source PTS offset,
rounded up to nanoseconds. Late frames do not reset the clock, get dropped, or
become reused observations. The detector worker owns waiting; the consumer can
receive the current pose while the next frame is not due yet. No new worker or
larger frame queue is introduced. Cancellation wakes scheduled waits and cannot
mask native inference/close failures. The initial decode is outside source-age
mapping but included in ordinary loop/decode timing.

`paced_loop_fps` is a delivered replay rate, not maximum capacity;
`unpaced_loop_fps` is null in paced arms. The source is about 59.37Hz, so a replay
around 59.37FPS is expected, not an 80->59 performance regression. Existing
16.67ms interval counters are explicitly NOT replay deadline misses; the source
itself is not exactly 60Hz. Review `replay_ages.source_age_ms`, release lateness,
source-time age windows, last-frame age and output hashes instead. Age thresholds
are diagnostics rather than invented product acceptance criteria.

## Run

```bash
git fetch origin
git switch feat/wholebody-onnx-lab
git pull --ff-only origin feat/wholebody-onnx-lab

uv run --with-requirements tools/requirements-wholebody.txt \
  python -m motioncapture.wholebody_optimize_bench \
  "$HOME/Downloads/20260906_030954.mp4" \
  --detector-provider coreml-all --pose-provider coreml-all \
  --allow-cpu-partitions --research-only --suite paced \
  --max-frames 0 --output sessions/wholebody-optimize-r4-paced.json
```

No original video is re-encoded, no frame is repeated, and full per-pixel/result
verification stays enabled. Old suites remain unchanged. Existing files/checkpoints
are not overwritten. Preserve local changes on a Git conflict; do not force reset.
The new run is still an explicit file replay, not a hardware camera, display,
3D, facial blendshape, multi-camera or commercial-release validation.

## Local verification scope

See adjacent verification JSON. The targeted tests use real OpenCV transforms,
FFmpeg/OpenCV variable-PTS fixtures, actual stage owners/checkpoint writers,
controlled clocks and events. Neural calls are explicitly synthetic fixtures.
A legacy six-arm plan test is corrected to use DEFAULT_MODES, not the expanded
optional-mode registry; the six-arm production plan itself is unchanged.
Actual M5/CoreML paced execution, camera/display behavior, full repository/Ruff,
Python 3.12 runtime and known tracking-quality issues remain unverified here.
