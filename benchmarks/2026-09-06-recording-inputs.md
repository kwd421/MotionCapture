# Recording input validation and benchmark limits

Repository baseline: dd92ae03e8b3faa62810b4fdac17968e5461862f.

## Measured source facts

| Input | Codec | Decoded pixels | Duration | Decoded frame count | PTS-span rate |
|---|---|---|---:|---:|---:|
| Phone recording | HEVC | 1920 x 1080 | 108.5003 s | 6442 | 59.373005 FPS |
| Earlier webcam recording | H.264 | 1280 x 720 | 120.0000 s | 3599 | 29.991664 FPS |

The phone file's rational time base is 1/90000. PTS intervals: minimum 8.5111 ms,
median 16.8556 ms, p95 16.8667 ms, p99 16.8778 ms, maximum 25.2 ms. PTS are strictly
increasing; no need to manufacture a 60Hz clock or duplicate frames. The webcam
file has one 66.6667ms PTS gap. These are encoded presentation timestamps, not
verification of sensor exposure timing or exposure uniqueness.

Each source was decoded twice through the new reader. All 6442 + 6442 + 3599 +
3599 = 20082 frames matched the independent FFprobe timeline/dimensions, with exact
frame-count completion and post-run source checksum verification. Original
recordings, audio, frames, identifying metadata and landmarks are not in the repo.

## Observed local file costs (not tracking speed)

Linux/Python 3.13.5, OpenCV 4.13.0, FFmpeg CPU file decode, four reported decoder
threads, no AI model or GUI. Unpaced all-frame passes:

| Input | Run | Mean decode read | Decode-read p95 | Complete file-loop FPS |
|---|---:|---:|---:|---:|
| Phone | 1 | 7.1041 ms | 17.8136 ms | 139.1563 |
| Phone | 2 | 7.0957 ms | 17.8829 ms | 139.3401 |
| Webcam | 1 | 1.5323 ms | 4.1380 ms | 627.9464 |
| Webcam | 2 | 1.4625 ms | 4.0609 ms | 662.9993 |

These are shared-container measurements, not isolated hardware certification.
The inference runtime was unavailable; none of these FPS numbers includes hand,
body or face inference. They do not prove a model speedup or 60FPS live capture.
The clips have different pixel sizes, codecs, lighting, scale and movement; the
between-clip time difference is not a code A/B result. Decode is separately
measured to avoid conflating compressed-file costs with camera or model costs.

A preliminary 180-frame decoder-thread screen found 4 threads faster than 1 on
this host; default decoding already reported 4 threads. No live decoder default
was changed and no default-vs-default performance improvement is claimed.

## Visual inspection and workload interpretation

Sampled frames contain face/upper-body movement, both hands near and farther from
the camera, occlusions and camera handling/lens changes. The wide-view section
makes the subject and hands smaller in the frame. These are useful stress cases,
not a controlled accuracy dataset. The approximately 60-80s phone-camera handling
and transition region should be examined separately using source-time windows.
This region is a conservative visual annotation, not a recovered lens-event log.

Low-light segments remain part of the test set. A low detection count without
visibility annotation is not proof of a regression. Feet/ground contact and
multi-actor identity are not established by these upper-body recordings.

## Implementation and verification

An explicit all-frame recording path shares the production tracker's unchanged
model/inference/numeric-conversion body. Only time/identity dispatch is added;
live camera, default models, thresholds, resolution and provider are unchanged.
File timestamps are never disguised as host-monotonic receive times. There is no
fallback from unavailable inference to decoder-only benchmarking.

26 targeted tests passed on Linux/Python 3.13.5. Actual FFmpeg + OpenCV VFR and
pixel-equivalence tests run without a camera. Model-clock tests use declared
native-task substitutes. A separate preflight correctly failed with
`pinned_mediapipe_unavailable` and zero native inference runs.

Attempts to install the pinned native runtime/required Python failed because
external package downloads were unavailable in this execution environment.
Native MediaPipe A/B and accuracy comparison, Python 3.12 execution, Ruff, macOS
and Windows native capture/GUI, thermal stability and 60FPS acceptance are OPEN.
Do not promote this benchmark infrastructure to a verified 60FPS optimization.
