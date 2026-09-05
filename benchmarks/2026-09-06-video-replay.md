# Fixed-video replay: evidence, candidate and remaining gates

Base: `dd92ae03e8b3faa62810b4fdac17968e5461862f`.

## Input actually inspected

The user's local video is 1280x720 H.264, duration 120.000 seconds, 3,599
fully decoded frames. SHA-256:
`f0346fc4539e6b8f17c67287f6827a3d775459307c3eff515ac574134986414d`.
Original time_base is 1/15360. Adjacent PTS intervals: 3,597 at 1/30 second
and one at 1/15 second. The PTS-span rate is 29.991664 fps. There are zero
adjacent byte-identical decoded BGR frames. This does not prove sensor exposure
rate or absence of motion blur. It is NOT a recorded 60-fps clip.

Nine representative frames were visually inspected privately. Face, both hands,
upper-body motion, distance changes and blur are present. Dark background and
motion blur make this useful as a low-light stress input. It does not establish
full-body/foot accuracy, annotated ground truth, or generalization to bright rooms.
No source video, screenshots or individual pose/face records are committed.

## Actual execution and limits

Host: Linux x86_64, Python 3.13.5, OpenCV 4.13.0, NumPy 2.3.5, four OpenCV
threads. FFprobe 7.1.5. Every original frame was decoded and independently checked
against probed PTS. Measured decode/read mean: 3.2225 ms, p95: 12.0257 ms.
This includes the backend decoder/read/timestamp boundary, not model inference.
Decode/hash inspection took 41.467 seconds excluding the initial probe.

MediaPipe is not installed here. Explicit Python 3.12 environment provisioning,
PyPI package installation and a wheel download failed. The requested native
inference pass produced status=failed / ModuleNotFoundError / inference=false,
not substituted data. NO native tracker speed, accuracy or 60-fps claim is made.
The prior promise that uploading a video alone enables native benchmarking here
was too broad; this environment cannot currently obtain the pinned runtime.

## Measured optional optimization

BGR-to-RGB buffer reuse uses the same OpenCV operation and unchanged pixel values.
Same video, all 3,599 frames, three paired rounds with per-frame alternating order.
The first 60 frames of each round are processed and equality-checked but excluded
from timing statistics. Every round has 3,539 timed frames and 3,599 equality checks.
Decoder, model, MediaPipe Image construction, GUI and equality comparison are
outside the measured conversion region. Cache warming from paired input is a
limitation: this is NOT a native-pipeline A/B.

| Round | Allocate mean ms | Reuse mean ms | Allocate p95 ms | Reuse p95 ms |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2.006439 | 0.408175 | 5.994001 | 0.805926 |
| 2 | 1.313200 | 0.415066 | 5.273602 | 0.817077 |
| 3 | 0.850412 | 0.463827 | 4.409796 | 0.978727 |

Medians of round means: 1.313200 -> 0.415066 ms (68.39% less time in this
isolated conversion region). Baseline timings vary substantially, so report all
rounds. All 10,797 converted frames were pixel-identical across modes.

The reusable array is borrowed until the next synchronous tracker call. All
Pose/Hands/Face consumers must finish first; failures require explicit cleanup.
No shared scratch buffer is inserted into the capture queue. This retains one
current-shape RGB buffer rather than allocating it anew on every frame.

The live default remains `allocated`. The candidate is selected explicitly by the
new offline runner (`--rgb-modes allocated reuse`). It is not yet promoted as a
proven M5 live optimization. The supplied M5 log had only ~0.345ms input conversion
including MediaPipe Image construction; this change alone cannot solve the hand
inference bottleneck or prove full-pipeline 60fps.

## New benchmark contract

`python -m motioncapture.video_benchmark INPUT --stage track --rounds 2`
executes allocated/reuse then reuse/allocated (ABBA), with fresh model instances
and a fresh stream each pass. All frames are processed at original presentation
timestamps, never timestamped from file-read speed. Camera live time remains
host-monotonic; file PTS is an optional separately named identity field.

The report includes model hashes, code identity, environment, warmup scope,
stage mean/p50/p95/p99, 16.67ms budget exceedance, per-output-hand-slot counts and
transitions, and aggregate prediction digests. Slot counts are not a trace of the
internal palm detector. Equal digests imply identical outputs, not true accuracy.

Failure does not silently run inspect/preprocess. Native process abort isolation
is not implemented; no report after an abort must not be interpreted as success.
Unpaced file capacity excludes live camera, display, output transport and thermal
conditions. Original 30-fps motion cannot be turned into 60-fps capture evidence.

## Verification

42 focused tests passed on Linux/Python 3.13.5: 30 video/clock/conversion tests,
3 adapter-wiring tests using an explicitly synthetic native API, and 9 existing
preview tests using real OpenCV. Real tiny encoded fixtures exercise ffprobe,
OpenCV, PTS, full-frame counts, cleanup and all-frame ABBA runner behavior.
Python 3.12 syntax parsing passed. Ruff, the complete repository suite, native
MediaPipe inference, Mac/Windows cameras/GUI and long-duration M5 thermal behavior
remain unverified. The checked core conversion operation was not changed after
measurement; later edits add provenance metadata, formatting and failure labels.
