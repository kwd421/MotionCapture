# Ready-only pose handoff

Definition: opt-in scheduling candidate that submits an already-detected next
frame to the idle pose owner before returning the current packet for validation.

Evidence: the user's r1 900-frame palindrome measured 50.8166 FPS in overlap-LUT
versus 37.4012 sequential-reference, with all six prediction/detector/pixel hashes
identical. Overlap verification averaged 2.1978 ms. The owning generator currently
yields BEFORE submitting the next pose, leaving that owner idle during validation.
This cost is observed, not a prediction that all of it can be recovered on M5.

Contract: same models, providers, crop/normalization, precision, detector on every
original source frame, all detected people, PTS/order and validation work. No
output smoothing, point suppression, skipped inference or disabled checks. Each
stage still owns at most one outstanding future. Completed results occupy a slot.
The next pose is submitted before yield ONLY when its detector future is already
done; never wait for a future frame to return the current one. No new worker or
unbounded result queue. Source decoding and verification remain caller-owned.

Owners: StageOwner reports readiness through Future.done(), consumes errors through
Future.result(), and retains construction/run/close on its dedicated thread.
StagePipeline owns ready-only handoffs and exact-N submissions. Observer delay is
not included in model inference timing. Source-frame identity remains authoritative.
Early consumer stop may finish one admitted next pose without publishing it; counts
and normal owner cleanup retain this distinction. No forced native cancellation.

Failure states: known later-frame errors prevent further publication. Unready
future is a normal scheduling state, not permission to substitute results or
providers. At most one ready-check per packet; no polling/sleep loop. Identity or
source errors remain terminal. Current input-inspection failures are not 0-FPS runs.

Acceptance: ABBA overlap-LUT / ready-handoff-LUT / ready-handoff-LUT / overlap-LUT
retains full pixel and prediction hashes in every loop. Compare actual throughput,
host latency, output intervals and pose-submit gaps; no inverse-latency FPS claims.
Default six-arm experiment and live defaults are unchanged. Probe-identified crop
sensitivity and sparse large CPU-detector/CoreML-detector pose differences remain
OPEN; same-provider hash equality is not accuracy nor an outlier fix.

Verification: deterministic events prove pre-yield admission and that a pending
next detector never blocks emission; exact-N, every PTS, errors/cancellation,
model ownership, incomplete cleanup and pixel/result immutability. Real FFmpeg/VFR
fixtures exercise the actual CLI path with labelled synthetic model sessions.
Native M5/CoreML performance and sustained live 60Hz remain device gates.
