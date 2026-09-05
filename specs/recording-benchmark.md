# Recorded-video benchmark

## Contract

Run the pinned production Pose/Hands/Face tracker against every original decoded
frame of an explicitly selected local MP4/MOV. Never substitute file input for a
failed live camera. The source time authority is integer FFprobe video PTS and
its rational time base, not frame number/FPS or host decode speed. FFmpeg-backed
OpenCV decode positions and dimensions must agree with that timeline at every
frame. Missing/nonmonotonic PTS, unexpected rotation, decode failure, timestamp
mismatch and count mismatch fail explicitly. Automatic rotation, frame resampling,
seeking, dropping, resolution changes and model changes are not part of this slice.

`RecordedIdentity` is distinct from host-monotonic `FrameIdentity`. The original
live method and the recording method share one inference implementation and
numerical-result conversion. A tracker lifecycle may not switch sources/time
domains. All poses/hands/faces remain enabled with the pinned models/thresholds.

An unpaced full pass measures processing capacity, not live motion-to-photon
latency. Decoder service time, tracker stages, preview and bookkeeping are
separate. The first 60 frames are still processed and included in the full pass;
additional steady-state metrics exclude them. Each repeat creates a fresh tracker
and starts at original frame zero. Time windows never reset tracking state.

Decoder-only mode explicitly omits inference, reports `inference_executed=false`
and cannot establish the 60FPS tracking goal. Missing MediaPipe in track mode is
an error, never an automatic decoder-only fallback. Decoder thread count zero
means the decoder's own default; requested and actual values are both recorded.

Counts grouped by *detected* hand count and face detection are workload evidence,
not ground truth for hand visibility, accuracy or palm-detector invocation.
10-second source-time windows separate disturbances such as camera handling or
lens switching without hiding them from full-pass results. Additional warmup and
workload grouping are derived summaries, not discarded frames.

## Privacy and owners

Video/audio, thumbnails, individual landmarks and facial values are never written
by the benchmark. Source SHA-256 identifies the input; no absolute path, original
filename or container metadata tags are included in the report. FFprobe owns
metadata inspection; `RecordedDecoder` owns the selected file decoder;
`MediaPipeLandmarkTracker` owns model state; the runner owns timing and report
lifecycle. Only explicit local report paths are written, with no overwrite.

## Verification

- Variable-frame-rate synthetic video: all decoded PTS must match; no resampling.
- Truncated data/count, unavailable decoder and wrong dimensions/PTS must fail.
- Tracker clock: repeated/nonmonotonic times, changed source or mixed live/file
  provenance fail. Inference body and model configuration remain unchanged.
- Complete real-file decode and exact timestamp checks on both supplied clips.
- Real pinned native inference, target Python 3.12, macOS/Windows camera/GUI and
  60FPS live acceptance are separate gates. Tests with synthetic trackers do not
  close those gates.
