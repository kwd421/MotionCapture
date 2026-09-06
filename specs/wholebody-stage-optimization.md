# Exact normalization and bounded detector/pose overlap

Definition: one opt-in file action compares original preprocessing, byte-exact
lookup normalization and bounded cross-frame stage overlap, with source-locked
counterfactual diagnostics for reported pose outliers.

Current contract: the original RecordingProbe/RecordedDecoder owns source PTS,
WholeBody133 primitive codecs own the reference recipe and OrtModel owns providers,
preflight evidence and graph fallback rejection. The alternative input adapter is
versioned, byte-checked against the reference and tested over all uint8 values.
No fake world landmarks, blended shape coefficients, actor IDs or 3D is added.

Ownership: decoding is caller-owned; one dedicated worker owns each model session
through construction, inference and destruction. A stage admits at most one future;
a completed-but-unreceived future still occupies its slot. Exact N inference applies
also to prefixes. Two stages may overlap across frames, never race inside one model
instance. Poses retain the exact source frame/boxes and are emitted in original order.
Normal termination waits for work; errors never publish observed failed work as valid.
Native force-cancellation/isolation is not provided and is not claimed.

Experiment plan: all modes retain the requested detector/pose providers, input
resolution, thresholds, original order and every-source-frame detector cadence.
Sequential controls also use the same owner threads. The six-arm palindrome isolates
normalization and overlap. Throughput/latency/summed-work are separate measured
quantities; the reciprocal of pipeline latency is NOT throughput.

Diagnostics: bounded source frame selection from prior error IDs and verified PTS.
CPU-detector and selected-detector inputs feed the SAME fixed pose session A/B/B/A.
Source/crop/tensor/response hashes, deltas, argmax changes, identical-input repeats
and geometry-only counterfactuals are retained. No frames, individual coordinates or
raw tensors are written. Ambiguous person association remains explicitly unmatched.
Known CPU-vs-CoreML detector-induced pose differences are not cleared by performance
parity of a new scheduling experiment. Diagnostics are not included in performance.

Acceptance: expected frame counts, PTS, hashes/disagreement evidence, per-person
accounting, terminal errors and owner cleanup are mandatory. Speed candidates do
not become live defaults without a native device/quality review. Unsupported mixed
models/providers fail, never become CPU success. No accuracy/commercial/license
clearance or sustained 60FPS claim is authorized by unit tests.

Verification: event-controlled overlap/failure/ownership tests, exact prefix/EOF,
real FFmpeg VFR fixture decoding, all uint8 normalization, random/ULP/out-of-frame
crop parity, source mutation guards, and interrupted/partial report semantics.
Native M5/M1/Windows inference, device thermals and visual quality remain target gates.
