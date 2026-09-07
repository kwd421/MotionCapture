# Same-frame batch-2 fixed-60Hz replay gate

## Purpose
R10 showed that batching two people from the SAME source frame reduces the multi-person pose burst while preserving every source frame. The source recording itself spans about 59.373 Hz, so original-PTS replay cannot by itself prove capacity at an exact 60 Hz admission cadence.

## Invariants
- Original decoded frames and `RecordedIdentity` PTS/time_base are never rewritten.
- Detector still runs on every admitted source frame.
- No frame, person, hand, foot, face point class, or verification work is skipped.
- Only adjacent detector slots within one source frame may use batch=2.
- No future-frame pose batching and no stale result reuse.
- CoreML ALL/ALL, DWPose-m 256x192, OpenCV normalization, ready-only handoff remain fixed.
- Batch-2 failure is terminal; no hidden batch-1 retry.
- Fixed-rate pacing uses an absolute sequence/rate offset from one epoch, never repeated rounded periods and never a late-frame rebase.

## Experiment
A/B/B/A:
1. batch2 + original PTS release
2. batch2 + exact 60 Hz host release
3. batch2 + exact 60 Hz host release
4. batch2 + original PTS release

The fixed-60Hz arms compress 6,442 admitted frames to a scheduled span of 6,441 / 60 = 107.35 s while retaining the recording's original PTS values for identity and output hashes.

## Interpretation
This is a file-capacity and scheduled-release latency test. It is not camera sensor timing, USB capture latency, display latency, or source-to-photon proof. A successful run can establish that the selected processing path does not need the recording's slower 59.373 Hz average cadence to avoid backlog; it cannot set `live_60fps_verified=true`.

Review source-age mean/p95/p99/max, final source age, >100ms counts, frame/pose coverage, hash equality, and fixed pacer summary. No latency threshold is silently introduced by this experiment.
