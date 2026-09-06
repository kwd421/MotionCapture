# Source-PTS-paced whole-body replay

Definition: exercise the established exact-input engine at the original file's
release times instead of maximum decoding speed. This is an explicit file test,
NOT camera capture or a 60Hz frame generator. Source PTS remains authoritative.

Baseline evidence: r3 processed all 6,442 source frames in each of four passes,
with identical pixels, boxes and predictions. Both normalization kernels exceeded
60 FPS over the full predominantly-one-person file. The next open performance
boundary is source-age/backpressure under real-time arrivals, not nominal FPS.

The detector worker waits against a fixed host epoch plus the exact rational
PTS offset (rounded UP to integer ns). Epoch starts at first detector entry after
setup/first decode; first-decode latency is excluded from source age and remains
in loop/decode timing. Clock mapping never resets to hide a late frame. No frames,
people or model calls are dropped/repeated; no model/provider/precision changes.
One detector and one pose future stay bounded, with original ready-only handoff.
Decoding may run at most one frame ahead; its pixels are withheld from detector
inference until their due time. This does not emulate a camera driver/buffer.

Waiting belongs to the detector worker, NOT the consumer: the current pose may
be returned while a later source frame is not due yet. Cancellation wakes waiting
work but cannot force-cancel a native model. Known model errors remain errors;
intentional pacing cancellation during cleanup must not hide them. Owners close
on their original threads. No wall-clock timestamps or source pixels are logged.

Report source-schedule-to-verification age and source-release lateness, separated
from model service and pacing wait. Age thresholds are diagnostic, not invented
product SLAs. Source-time window ages show backlog; last-frame age exposes drain.
Paced rate is labelled paced_loop_fps; unpaced_loop_fps is null, not maximum
capacity. Camera, photon latency, physical ANE/GPU dispatch and accuracy stay
unverified. The source's ~59.37Hz must not be rewritten as exact 60Hz.

Acceptance: full original PTS/count/hash coverage, no early inference or hidden
clock resets, ordered outputs, observable delayed release, exact prefix with no
N+1, cancellation and failure cleanup, old unpaced plans unchanged. Fake-clock
and event-controlled tests plus real VFR decoding must exercise the CLI. M5
native replay is a separate device gate. No pixel, tensor or landmark logging.
