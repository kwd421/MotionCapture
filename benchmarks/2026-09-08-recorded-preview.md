# Recorded whole-body preview verification — 2026-09-08

Implemented original-PTS file replay through DWPose-m, YOLOX-tiny, CoreML ALL
with explicitly allowed CPU partitions, same-frame batch-2 and OpenCV
normalization. This is 2D inference and preview, not camera/3D/avatar completion.
See [the slice contract](../specs/wholebody-recorded-preview.md).

## Observed results

Local machine-readable evidence: `sessions/preview-20260908-summary.json` and its
referenced per-run reports. Input images, coordinates and audio are not in JSON.
Selected annotated snapshots were explicitly requested with `--snapshot-frame`.

| Input | Validated / requested | Result | Paced inference/enqueue FPS | UI-return age p95 | Last UI-return age | Peak RSS MiB |
|---|---:|---|---:|---:|---:|---:|
| MacBook recording, 720p, 29.992Hz | 3599 / 3599 | Completed | 29.992 | 64.41ms | 40.67ms | 871.19 |
| Phone recording, 1080p, 59.373Hz | 6442 / 6442 | Completed; cannot keep up | 54.790 | 8710.10ms | 9073.34ms | 1051.48 |
| Public single-person dance, 514x496, 25Hz | 244 / 244 | Completed | 24.968 | 75.53ms | 130.01ms | 825.83 |
| Public dance montage, 640x360, 25Hz | 368 / 1255 | Failed: >8 detected slots | — | Partial run only | — | 810.36 |
| Public street crowd, 720p, 50Hz | 102 / 205 | Failed: >8 detected slots | — | Partial run only | — | 818.03 |

Every completed row has equal decoded, validated and UI-submitted counts, with
zero intentional skips/replacements and all decoder/model/window owners released.
Failed rows retain partial counts and null successful-run FPS. The dance montage
stopped around 14.7 seconds, before its later two-dancer section; that later
section was not inferred. The crowd prefix already accumulated about 3.56s of
UI-return age before its capacity failure.

UI-return age means scheduled file release to `imshow`/`waitKey` return. It is
not sensor-to-photon latency. Display scanout/refresh, GPU versus ANE routing,
power, ground-truth pose accuracy and persistent actor identity remain unverified.
Peak RSS includes model setup and the process lifetime. A system snapshot during
the MacBook FIFO run observed other Rust compilation work; these are not isolated
hardware measurements, and do not establish a regression against older r11 runs.

## Preview scheduling change

The first synchronous phone preview processed 38.818 FPS and finished 57.44s
behind its source schedule. UI event handling averaged 13.64ms per frame and
pose submission gaps averaged 10.81ms. It blocked admission of subsequent work.

The selected implementation runs inference/validation on one owner thread and
uses a one-packet FIFO to the main-thread window. It applies backpressure and
never discards or reuses frames. Pose submission gaps averaged 2.27ms afterward;
the phone run reached 54.790 FPS and ended 9.07s behind. All 6442 input-pixel,
detector and pose hashes matched the synchronous run. MacBook before/after hashes
also matched. The trade-off is one extra worker and a bounded handoff, with peak
phone RSS observed at 929.72 MiB before and 1051.48 MiB after. These sequential
runs are observational comparisons, not thermally/background-load controlled ABBA.

This change removes the large synchronous bottleneck but does **not** close the
phone live-rate performance gap. No repeated frames, reduced models, lower
inference resolution, skipped people or hidden CPU fallback were used.

## People and input validity

In rendered OpenPose-source frames 0 and 100, the central foreground subject's
label changes from `slot 3` to `slot 0`. These labels are detector order within a
frame, not actor IDs. The pipeline can output multiple independent person
observations, but cannot maintain actor identity through time, crossing, occlusion
or mirror reflections. The existing eight-slot guard terminates the two larger
scenes instead of silently dropping extra detections. Its capacity is separate
from the product's dynamic camera contract and initial single-performer scope.

Another downloaded full-body example was rejected before model setup: the HybrIK
file declares 1298 frames but FFprobe decodes 1198. It was not silently remuxed or
accepted. Two MMPose fixtures contain only five frames each; they were inspected
but not used as sustained tracking benchmarks. Source URLs, checksums and input
metadata are retained locally in `benchmarks/inputs/public-sources-20260908.json`.

## Verification and limits

- Full repository: 373 tests passed, 2 optional browser tests skipped; Ruff and
  `git diff --check` passed.
- Real CoreML runtime and real FFmpeg/OpenCV decoding for the runs above.
- Native Ctrl-C replay stopped with exit 130: 331 validated, 329 UI-submitted;
  all model, decoder and window owners were released. Partial counts are retained.
- Tests cover main-thread rendering, bounded queue cancellation, consumer failure,
  input/coordinate preservation, invalid-point rendering, cleanup failure,
  explicit snapshots and report overwrite protection.
- The exact composed preview images were inspected. Independent native-window
  inspection through Computer Use timed out on the Python app; physical screen
  presentation and keyboard/window-close UX are not independently verified.

## Public input sources

- [OpenPose example video](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/examples/media/video.avi)
- [tf-pose dance example](https://github.com/SrikanthVelpuri/tf-pose/blob/master/dance.mp4)
- [Qengineering pose dance example](https://github.com/Qengineering/ncnn_Pose_RPi_64-bits/blob/main/Dance.mp4)
- [HybrIK dance example](https://github.com/jeffffffli/HybrIK/blob/main/examples/dance.mp4)

These are local evaluation inputs, not a redistribution or commercial-release
clearance of the source videos.
