# DWPose runtime 1.29 follow-up optimization

Definition: reduce the measured all-frame recorded-preview latency using the
unchanged DWPose-m and YOLOX-tiny ONNX assets on the verified contiguous CoreML
graphs in ORT 1.29.0. This resumes the user's optimization request after the
separate InstantHMR trial; no commit/push is requested.

Contract: preserve every source frame, original PTS, detector cadence, every
detected person, model resolution/weights, body/feet/both hands/fingers, explicit
provider selection, bounded queues and terminal failure/cleanup semantics.
No stale observations, fallback, synthetic calibration, frame dropping or silent
precision change. Existing RecordedDecoder, StagePipeline/BatchStagePipeline,
OrtModel/PoseBatchModels, run_pass, and Preview remain the authoritative owners.

Initial experiment: remeasure current strict CoreML ALL/ALL baseline, then
permitted CPUAndGPU/CPUAndNeuralEngine pairs on 1.29.0 with the same-frame batch-2
path. These policy names permit CPU internally and do not prove device routing.
Compare repeated arms, actual placement, model/box/pixel hashes, matched outputs,
stage timing and original-source ages. If necessary, compare sequential versus
overlapping model work and isolate display backpressure. Any implementation
change must be driven by measured cost and verified on both full local videos.

Acceptance: a candidate must retain coverage/identity/cleanup and either exact
outputs or explicitly reviewed numerical differences. Short-prefix improvement
is insufficient for promotion. Verify both full recorded sources with preview,
including interruption and stage failure. Record transient stalls and shared-host
activity; no camera/photon, ground-truth, thermal or physical device claim from
host-side timings alone. No quantitative speedup is assumed in advance.

Status: bounded experiment complete; no new candidate met promotion criteria.

## Verified compute-policy result

Six native 1,200-frame phone-prefix runs, order A/B/C/C/B/A, used the
unchanged assets, ORT 1.29.0, original PTS and same-frame batch 2. A selects
ALL/ALL; B selects detector CPUAndGPU / pose CPUAndNeuralEngine; C reverses B.
Both A runs completed at 59.294 Hz with validation-age p95 22.18 / 22.59 ms.
B p95 was 41.73 / 196.56 ms; C was 1,467.80 / 38.42 ms. No alternate pair
improved this baseline. A repeat pixel, box and pose hashes matched exactly.
All 7,200 frames completed with released owners; strict ORT preflight reported
no CPU execution nodes. CoreML's internal physical GPU/ANE routing is unknown.
B/C outputs differ numerically from A, and are not promoted or represented as
quality-equivalent. Local raw reports: `sessions/dwpose-followup-20260908/policy*.json`.

## Display baseline and observation scope

The unchanged OpenCV full-phone run submitted all 6,442 frames at 57.607 Hz;
UI-return age p95 2,275.65 ms and final age 3,528.11 ms. Mean composition
cost was 0.90 ms, submit/event-pump cost 13.21 ms. The native window was active,
not hidden and unoccluded in observations after startup. This run had no
observed >1-second no-progress stall, but accumulated backlog near the end.
It therefore does not reproduce the earlier multi-second freeze mechanism.

The bounded probe samples native app/window state, low-power mode and thermal
state every 60 frames and on slow display calls. It records Python/native
stacks only on >1-second no-progress observations (maximum two per run).
Thermal state changed from nominal to fair; low-power mode was false. This is
state evidence, not a hardware-throttling diagnosis. Computer-use app selection
repeatedly timed out, so independent visual/physical-display inspection remains
unverified for this follow-up. No recorded image or landmark coordinates are
written by the probe.

OpenCV 4.14.0 uses GCD on this machine. `getNumThreads()` returns 10 even after
`setNumThreads(1)`. This is not evidence that the requested serial path failed:
[the exact upstream source](https://github.com/opencv/opencv/blob/4.14.0/modules/core/src/parallel.cpp)
sets `numThreads`, bypasses `parallel_for` for 1, but its GCD getter reports the
CPU count. Existing runtime metadata is a raw getter value, not measured worker
concurrency. Do not label a requested 1-vs-default test as observed 1-vs-10 workers.

The first SDL launch failed because the isolated overlay did not include the
optional pygame-ce dependency; no frames were processed and it was not counted
as a performance arm. The corrected explicit environment included pygame-ce
2.5.8. Its full-phone run submitted 6,442 frames at 55.786 Hz, UI-return-age p95
6,863.51 ms and final age 7,154.03 ms, despite mean UI submission cost falling to
1.04 ms. Detector/pose means rose from 8.92/8.76 ms to 10.97/11.41 ms. SDL was
inactive and occluded at every sampled callback. These observations cannot
establish that the different backend caused the inference slowdown.

There is also a concrete dependency concern: both OpenCV and pygame-ce load
separate SDL2 libraries, and the Objective-C runtime reports duplicate SDL
classes. An isolated import-only check with opencv-python-headless 4.14.0.94,
pygame-ce 2.5.8 and numpy 2.2.6 reproduces the duplicates: the headless wheel
still bundles SDL2. No installed library was deleted or replaced to suppress
this diagnostic. No crash was observed, but an unverified duplicate-runtime
combination is not promoted on the basis of a faster individual UI call.

## Final bounded result

The explicit macOS user-initiated activity arm also completed all 6,442 frames,
but achieved only 42.362 Hz, UI-return-age p95 45,507.55 ms and final age
43,905.21 ms. Mean UI cost was 1.73 ms. The activity token was acquired and
released in `finally`; no >1-second no-progress stack capture was triggered.
This does not support adopting the activity flag as a performance fix. It also
does not prove App Nap was the cause of the earlier failures: ordering, host
load and window state are confounders, and no App Nap transition was captured.
The import-only dependency check ran during this arm and is additional host
activity, so this is a rejection of promotion, not a controlled causal verdict.

All three full-phone runs have exactly equal complete-pass pixels, detector
predictions and pose-output hashes. Each submitted all 6,442 frames, joined the
inference owner, released the window and left zero undisplayed queued packets.
The observed process CPU costs were 148.93 / 186.34 / 265.80 seconds for OpenCV,
SDL, and SDL+activity respectively; peak RSS was 1,007.02 / 1,069.77 / 851.33 MiB.
The CPU measure covers timed work in all process threads; RSS includes setup.
No quality, camera latency or physical display timing claim follows from hashes.

Decision: retain the existing defaults and original model assets. The bounded
six compute-policy and three full-phone display trials processed 26,526 frames
but did not establish a new end-to-end performance improvement. Do not promote
SDL, the macOS activity adapter, a GPU/ANE split, or a thread-count claim. Since
no candidate passed the phone acceptance gate, no candidate was advanced to the
full-MacBook acceptance run; cross-source and real-window acceptance remain
incomplete. No production code or dependency configuration was changed, and no
new commit/push was performed. Existing interruption/failure tests were not
rerun because no production implementation changed; they are not new evidence.

The remaining useful optimization boundary is the GUI/runtime integration:
measure a single maintained display runtime without duplicate SDL classes,
with controlled visible-window state, alongside contemporaneous headless
control. The current observations support neither a claim that the GPU is
inherently too weak nor that a faster blit alone solves accumulated latency.
Do not continue an open-ended sweep of GPU/ANE policy pairs.

Reproduction scripts and bounded native state logs remain local under
`sessions/dwpose-followup-20260908/`; `preview-summary.json` collects all full-run
metrics. These diagnostics contain no written source frames or pose coordinates.

## User-requested lower-load retry

The user reported that resources should now be available and requested a retry.
The fixed full-phone plan is headless / OpenCV / OpenCV / headless in separate
fresh processes using the same ORT 1.29.0 environment and unchanged models.
Headless retains the exact one-slot FIFO, inference worker and main-thread
consumer/age observations but omits composition/window/event work. Its packet
count and age are explicitly labelled as headless consumption, not display.
The same observer runs in each arm; a parent records top process CPU/name samples
every 10 seconds. No user processes are stopped, no activity token is selected,
and no alternative SDL runtime is loaded. All 6,442 original-PTS frames are
required per arm; exact hashes, terminal cleanup and sampled window state are
checked. This controls more variables but does not make shared-host load or
physical display timing controlled measurements.

### Retry results

| Arm | Consumer rate Hz | Age p95 ms | Final age ms | Peak sampled Blender CPU |
|---|---:|---:|---:|---:|
| retry-A1 | 59.388 | 666.40 | 15.33 | 724.8% |
| retry-B1 | 55.386 | 8031.66 | 8004.03 | 718.6% |
| retry-B2 | 59.432 | 2578.05 | 29.51 | 667.1% |
| retry-A2 | 58.103 | 9074.32 | 2407.08 | 751.5% |

A is headless consumption and B is OpenCV submit/event return; both preserve the
same FIFO, inference and every original frame. Rates exclude initial consumer
latency and cannot alone establish real-time success. Both preview windows were
sampled unoccluded after startup. All four runs completed 6,442 frames each
(25,768 total), with exact pixel/box/pose hashes, no replacement/skips, joined
inference owners, released model owners, and no remaining queued packets.

The host was not continuously quiet: Blender CPU fluctuated from low single
digits to 667-752% during measured arms, and compiler/build activity was also
sampled. Thermal states were nominal/fair; no thermal throttling was measured.
No unrelated process was stopped. Thus the retry is not a clean low-load A/B.
B1 and B2 had almost identical mean UI costs (13.47 / 13.22 ms), yet different
backlog trajectories. A2 also accumulated latency without creating a window.
This rejects GUI display as the sole sufficient explanation. Shared-host
resource/scheduling variation remains a stronger hypothesis than a fixed GPU
throughput ceiling, but CPU percent correlation does not prove causation.

A1 recovered to roughly 20 ms mean age in source windows after 20 seconds.
B2 recovered to about 33-34 ms in the final two source windows after earlier
multi-second latency. A2 ended at 2.41 seconds despite having no GUI. Original
PTS and the no-drop policy expose these throughput shortfalls as accumulated
age rather than discarding old frames. The measurements do not establish a
smooth sustained 60 fps capture or source-to-photon latency.

Local evidence: `retry-summary.json`, each `retry-A/B*.json`, per-arm native
state logs, and `retry-host.jsonl` under the same session directory. No production
change was made and no implementation test rerun is claimed. All four planned
replay processes and the parent monitor exited normally.

## 2026-09-09 retry with Blender idle

The user requested another measurement after stopping Blender work. Repeated
exactly the same full-phone A/B/B/A plan, same fresh-process runtime, observer,
model assets, FIFO and original PTS. A omits rendering; B is OpenCV. No source,
model or production dependency edits were made. The existing probe/controller
were reused with unique `quiet-*` output names to preserve prior evidence.

| Arm | Consumer Hz | Age p95 ms | Final age ms | Max age ms | >100 ms frames |
|---|---:|---:|---:|---:|---:|
| quiet-A1 | 59.379 | 21.03 | 15.44 | 65.73 | 0 |
| quiet-B1 | 59.423 | 35.06 | 36.48 | 210.18 | 85 |
| quiet-B2 | 59.425 | 40.80 | 34.11 | 260.06 | 202 |
| quiet-A2 | 59.372 | 24.53 | 15.84 | 251.70 | 134 |

All four runs completed 6,442 frames each (25,768 total). Complete-pass pixel,
detector and pose hashes were exactly equal across arms. No intentional skips,
packet replacements or queued packets remained; inference owners joined and
model/display owners released. Headless arms created no window. Native window
observations showed active/unhidden/unoccluded preview windows after startup.
These observations are not physical display or source-to-photon verification.

Peak sampled Blender CPU during the measured arms was 1.9 / 1.9 / 1.9 / 2.1%,
versus 667-752% in the preceding retry. No comparable heavy build activity
appeared in the sampled top-process records. Sampling does not exclude short
unsampled work or measure GPU/ANE utilization. Thermal state was nominal/fair,
with no proof of actual throttling. No other application was stopped by the
measurement. The identical mean UI calls were still about 12.78 / 13.34 ms.

Result: both headless and both preview runs kept up with original source PTS
without terminal accumulated backlog. The low-load repeat strengthens the
shared-host contention explanation for the prior sustained slowdown, but is
not an isolated causal proof that Blender alone caused it. It argues against
a fixed inability of this model/device combination to process this source rate.
Short preview spikes (210-260 ms) remain; neither perfect smoothness nor live
60-fps camera capture is established. Consumer Hz excludes initial latency and
is not a physical refresh measurement. No changes were promoted or pushed.

Reproduction/evidence: `quiet_abba.py`, `retry_probe.py`, `quiet-summary.json`,
`quiet-A/B*.json`, per-arm state observations and `quiet-host.jsonl` under
`sessions/dwpose-followup-20260908/`. The four child runs and parent monitor all
exited normally. No written video frames or pose coordinates were produced.
