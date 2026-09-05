# Fixed-video benchmark and allocation experiment

Base: dd92ae03e8b3faa62810b4fdac17968e5461862f. This is an explicit offline
mode, never a live-camera fallback. No live model/settings changes by default.

## Contract

Use every decoded frame of one local file in display/PTS order, without frame
skipping, interpolation, rate conversion, exposure correction or synthetic
input. Preserve original integer PTS/time_base and the separate host decode time.
The model consumes original presentation time, NOT how quickly the file decodes.
A 30-fps file remains 30-fps input even if processed faster than 60 frames/s.

Run the pinned Pose/Hands/Face models with fresh instances each round. Report
initialization, full-pass and warmup-excluded measurements separately. Benchmarks
are unpaced throughput tests, not live-camera/GUI/motion-to-photon measurements.
Target 60 only means comparison to a 1000/60 ms compute budget. Every task and
both hands remain enabled; missing/failed/unavailable inference is never success.

Current hand-count and hand-count transition timing groups are correlations;
they do not expose whether MediaPipe's internal palm detector ran. Aggregate
prediction digests can establish exact repeatability, not real-world accuracy.

Allocation experiment: explicit reusable BGR-to-RGB buffer on the tracker owner.
No writes until the preceding synchronous process call has completed. Keep the
original allocation path as default until native end-to-end A/B is accepted.
Resize/reallocation remains bounded to the current frame shape. Pixel equality
alone does not establish native inference equivalence or speedup.

## Verification gates

Real-file PTS validation, all-frame decode, no FPS coercion, truncated/nonzero
exit detection; exact frame identities and no image persistence. Unit tests for
clock domain changes, PTS failure, allocation aliasing and numeric equivalence.
Native inference missing is a failing run, not a dummy/model-free benchmark.

No input video, extracted face images, raw landmarks, user paths or credentials
are committed. Output reports retain hashes, aggregate counts and timings only.
macOS M5/native MediaPipe and live 60-fps capture are separate acceptance gates.
