# Native lookup normalization and long-clip evidence

Definition: opt-in OpenCV LUT kernel replacing only NumPy's channel lookup in the
proven ready-handoff pipeline, plus bounded metadata-only output-cadence evidence.
Parent ec5103a5de25b6cd7fcf4ea8a4cf670c222bb251. Live/default suites are unchanged.

The authoritative uint8-to-float32 TABLE and affine crop geometry remain owned by
wholebody_fast_input. OpenCV LUT uses that identical three-channel table, followed
by a contiguous NCHW copy; no multiply/reciprocal approximation, geometry rounding,
model conversion, temporal filtering or missing-point suppression. All input values
and representative crop outputs must match the original recipe byte-for-byte.
A selected unsupported kernel fails, never falls back. Every source frame and
all detected people remain processed; workers, admission limits and handoff stay
unchanged. No native model or device performance claim follows from pixel tests.

The existing optimize runner compares ready-NumPy / ready-OpenCV / ready-OpenCV /
ready-NumPy with --suite native-normalize. --max-frames 0 selects the existing full
source contract, not a renamed prefix. Full-loop hashing and verification stay on.
Output-cadence summaries use ORIGINAL PTS for 10-second source bins and host
monotonic times for rates. A within-bin rate requires at least two completions;
intervals crossing source-bin edges are not attributed inside either bin. Separate
global interval counters cover all N-1 intervals and longest consecutive >16.67ms
runs. Sparse worst-interval records hold only frame/PTS/people/timing identifiers.
Counts are workload, not person identity or accuracy. Prefix/failed/short-window
coverage must stay explicit, and no summary sets live_60fps_verified=true.

Acceptance: tensor/output preservation, exact-N/all-person/PTS/cleanup, unchanged
old mode plans; test VFR source bins, boundary intervals, single/no-sample behavior,
partial failure and deterministic clock discontinuity rejection. Compare full-clip
mean, local cadence, tails and numeric quality on target device. A 900-frame mean
above 60Hz does not close sustained/live/quality gates. Known crop-sensitive sparse
CPU/CoreML outliers remain open. No raw frame, tensor or coordinate recording.
