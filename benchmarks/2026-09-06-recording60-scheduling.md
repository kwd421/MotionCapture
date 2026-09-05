# Recording60: measured bottleneck and a bounded scheduling candidate

## Evidence (supplied log, not newly measured model performance)

Source revision reported `4091f17bdfe96f8a6a259423ffc72a8ebee7df37`, with
`dirty=true`. The nature of the local changes is unknown; do not discard or reset
user work. This report cannot establish byte-identical execution of the remote
commit. The next report adds an aggregate on-disk Python-source fingerprint.

Three completed native MediaPipe 0.10.31 runs on the user's macOS/Python 3.12.10
processed all 6,442 original frames each (19,326 total). Input is 1920x1080 HEVC
with original PTS rate 59.373fps. No resizing/pacing was reported. Detection counts
and 10-second-window detection counts agree across runs. That is reproducibility
of these counts, NOT equality of landmark coordinates or accuracy.

| Run | Unpaced FPS | Hands mean / p95 ms | Tracker mean ms | Frame service mean / p95 ms | Service >16.67ms |
|---|---:|---:|---:|---:|---:|
| 1 | 46.016 | 18.878 / 23.390 | 20.207 | 21.649 / 26.403 | 84.69% |
| 2 | 40.451 | 21.394 / 25.850 | 22.895 | 24.612 / 29.540 | 90.50% |
| 3 | 40.031 | 21.645 / 26.102 | 23.153 | 24.866 / 29.831 | 90.69% |

## Interpretation and limits

- Confirmed in this log: hand-task duration is the primary critical path. With
  two resolved hands and a resolved face, 3,338/3,338 samples exceed 16.67ms in
  EACH run. Hands means are 19.698 / 22.605 / 22.915ms for that workload.
- One resolved hand plus face is slower still: means 23.086 / 25.717 / 25.891ms.
  Palm redetection is a plausible contributor, but resolved output counts do not
  reveal the internal detector or per-hand landmark work. Do not reduce num_hands
  to one or skip detection merely to improve this metric.
- Run 3 inference wall alone averages 21.916ms. Even the idealized removal of all
  decode, conversion, assembly and preview cost leaves about 45.63fps if that
  inference cost remains unchanged. This is a bound, not a measured optimization.
- Decode and preview on the Mac each cost under 1ms on average. GUI presentation
  and event pumping are NOT in this runner; an old waitKey bottleneck is not an
  explanation for this benchmark's 40fps capacity.
- Throughput falls 13.006% from run 1 to 3. Broad stage slowdown is consistent with
  sustained-load effects, but thermals, frequencies, power mode and background
  load are unmeasured. No proven thermal-throttling claim.
- OpenCV and decoder report 10 threads each. Neither number describes MediaPipe's
  internal worker pool or actual simultaneous CPU utilization. Oversubscription
  remains a hypothesis, not a diagnosed fact.
- Missing face/body in distance/lens-switch sections is not labeled a regression.
  Those sections remain in full-pass totals and source-time windows. A fully
  detected 20-30s section also exceeds the hand-task budget on every frame.
- These are offline all-frame service times, not motion-to-photon latency or live
  frame drops. Budget-exceedance percentage must not be called a drop percentage.

## Candidate

Default parallel scheduling and all live CLI defaults are unchanged. The explicit
`staggered` recording option starts hands and pose on two workers, then starts face
when either completes. All three models still process every original frame using
unchanged PTS, images, thresholds and numeric conversion. This limits concurrent
TASK CALLS to two, not native internal threads. It may lower contention, or it may
be slower because face starts later. No native speedup is claimed.

The same runner now supports fixed ABBA scheduling comparison and optional exact
numeric prediction hashes. Verification work is measured separately from frame
service but INCLUDED in loop FPS and CPU load. All arms must use the same setting.
ABBA helps compare order effects, but does not eliminate nonlinear heating or
background-load changes; compare the two parallel controls too. No automatic
winner, acceptance, provider change or quality reduction is implemented.

## Local verification

Baseline 26 recording/clock tests passed before editing. The candidate passed
48 targeted tests (26 recording/clock, 13 scheduling/comparison, 9 preview).
The suite exercises actual FFmpeg/OpenCV file decoding and preview pixels, with
explicit synthetic native-task substitutes for scheduling and tracker wiring.
Native MediaPipe is unavailable in this execution environment. Exact native
prediction equivalence, target-host speedup and sustained 60fps remain unverified.
Python 3.12 syntax parsing is not Python 3.12 execution. Local runtime is Linux /
Python 3.13.5; the full repository suite and Ruff are not claimed as passed.

No original videos, audio, individual predictions, local paths or private device
identifiers are committed. Only code, tests and aggregate evidence are included.
