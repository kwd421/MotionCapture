# r4 source-aged replay review and bounded same-frame pose candidate

Base: f376d022ddbf22844e7ca09040909698c4b4114f.
No live default, model/precision/provider/threshold, crop geometry, source release
schedule or detector cadence is promoted or relaxed by this patch.

## User-device evidence (not native inference rerun here)

Reviewed wholebody-optimize-r4-paced.json, SHA256
45e06ef66e5b37f5035995133662389b683ed49b4338d98506e63f316402d488.
Four full 6,442-frame passes: maximum speed / original PTS / original PTS / maximum
speed. All source pixels, detector boxes and prediction hashes agree. Each pass
has 6,254 person observations: 5,991 single, 130 double, one triple and 320 empty.
Counts are model observations, not verified actors or ground-truth accuracy.

| Metric | First paced pass | Second paced pass |
| --- | ---: | ---: |
| Delivered file replay FPS | 59.3565 | 59.3562 |
| Scheduled release to verified output mean, ms | 76.0694 | 53.7180 |
| Median age, ms | 28.9897 | 29.3555 |
| p95 age, ms | 437.7406 | 237.4538 |
| Maximum age, ms | 816.1679 | 481.0735 |
| Last-frame age, ms | 15.7124 | 16.3071 |
| Frames with age over 100ms | 708 | 712 |

The unpaced controls measure 78.3556/72.7070 FPS (pooled 75.4257). The source itself
is about 59.373Hz, so replay near 59.36 is expected; it is neither a capacity
regression nor proof of low latency. Transient backlog is large despite recovery
by the last frame. The first 0-10s bin peaks at 816/481ms; the 70-80s bin peaks at
282/342ms. Native model/wait/decode maxima do not isolate the cause of every age
spike; there is no per-frame correlation trace in the old report.

Two-person pose inference is sequential in the current owner and averages
26.7568/25.1554ms per frame in the paced arms (all person calls added). Its cost is
an observed burst-capacity problem against ~16.85ms source arrivals. This supports
a same-frame parallel candidate, but does NOT prove that all backlog is caused
solely by multi-person inference, physical GPU contention or thermal throttling.
The run records f376d022 plus dirty=true; not a verified clean commit-only run.

## Implemented candidate

`--suite pose-parallel`: paced one-lane / paced two-lane / paced two-lane / paced
one-lane, fresh sessions in every arm. The second lane is ONE additional identical
pose model instance on a dedicated worker, not a different model or provider.
For 0/1-person frames, primary-only behavior is unchanged. For 2..8 detected people,
even slots use the primary model and odd slots use the auxiliary model. Both finish
before output is reconstructed in detector-slot order. No actors/points are dropped,
no padding observations, and no previous-frame results or future frames are used.

Additional native session/worker memory and contention are accepted EXPERIMENTAL
costs; actual M5 memory and speed are not measured here. Physical CoreML GPU/ANE
parallelism is unverified. Runtime threading considerations were reviewed against:
https://onnxruntime.ai/docs/performance/tune-performance/threading.html
https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html
No new runtime flags or automatic hardware routing are introduced.

The source schedule/epoch and one-request-per-owner bounds remain unchanged.
The primary worker owns auxiliary submit/receive; cleanup joins primary before
auxiliary destruction. A selected auxiliary failure is terminal, never a retry on
the serial path. Startup reports retain earlier successful model preflights if
auxiliary construction fails. Native calls still cannot be force-cancelled.

`pose_inference_ms` is still SUM of all person-call durations and can now exceed
elapsed wall time. Use `pose_stage_ms` and `replay_ages` for latency. Lane stage
and join timings expose overhead on multi-person frames. Ending-person-count age
summaries and at most 16 worst-age frame identifiers expose burst propagation;
they do not attribute a backlog only to the current frame or define an SLA.

The new suite also verifies per-frame SHA256 of all output slots, scores, validity
and PTS. This covers multi-person frames excluded by coordinate matching. A changed
multi-person hash is reported rather than disguised as zero error. Original stream
hashing and source-pixel verification remain enabled in all arms. Additional hash
and age observer work is included in timing, so compare internally first.

## Run

```bash
git fetch origin
git switch feat/wholebody-onnx-lab
git pull --ff-only origin feat/wholebody-onnx-lab

uv run --with-requirements tools/requirements-wholebody.txt \
  python -m motioncapture.wholebody_optimize_bench \
  "$HOME/Downloads/20260906_030954.mp4" \
  --detector-provider coreml-all --pose-provider coreml-all \
  --allow-cpu-partitions --research-only --suite pose-parallel \
  --max-frames 0 --output sessions/wholebody-optimize-r5-pose-parallel.json
```

All four arms use original-PTS pacing. Compare source-age tails/windows, all-person
hashes and cleanup; FPS should stay near source rate. Do not accept a fast result
with changed outputs automatically. Existing suites/live defaults stay unchanged.
Preserve local work on pull conflicts; no forced reset. Full-device performance,
actual camera/display latency, crop-sensitive quality, M1/Windows validation and
commercial release clearance remain open. Optimization is NOT complete.
