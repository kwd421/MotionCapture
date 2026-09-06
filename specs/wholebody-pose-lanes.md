# Bounded same-frame pose lanes

Definition: opt-in two-session pose execution for all detected people in one frame,
with the existing detector/next-frame pipeline and original-PTS replay preserved.

Evidence: r4 delivered about 59.36 frames/s but source ages reached 816/481 ms.
The same report measures two-person pose calls serially (about 27 ms per frame),
while its source interval is about 16.85 ms. Bursty multi-box work is a plausible
backlog contributor, not an established explanation of every delayed frame.

Contract: one primary pose session is unchanged for 0/1-person frames. For 2..8
people, even detector slots run on the primary owner; odd slots run on ONE extra
dedicated owner using the identical factory/model/provider/settings. Join both
lanes and restore detector-slot order before publishing. No batch padding, stale
poses, actor selection, confidence changes, early release, epoch resets, drops,
frame-parallel pose processing or extra source admission. A slot is not actor ID.

Owners: the primary pose worker exclusively submits/receives auxiliary lane work.
Pipeline creation opens the auxiliary model on its dedicated worker; cleanup waits
for the primary owner to join its auxiliary work BEFORE closing that owner. Every
owner allows one outstanding request including completed-but-unconsumed results.
Model construction/run/destruction remain thread-affine. Construction/model/close
errors remain terminal, including secondary errors during cleanup. No serial or
provider fallback on a requested parallel failure. Cancellation still wakes only
scheduled source waits; native inference cannot be force-cancelled.

Trade-off: one extra native model session and worker, possible extra runtime/driver
memory and contention. No claim of physical GPU/ANE concurrency or device speedup.
`pose_inference_ms` stays SUM of person call times (may overlap), whereas
`pose_stage_ms` is elapsed frame wall time. Their meanings must not be conflated.

Acceptance: paced single-lane/two-lane/two-lane/single-lane full-file comparison,
unchanged hashes/counts/PTS and successful cleanup; improve source-age tails, not
just final FPS. Per-frame prediction hashes also cover ambiguous multi-person
frames; coordinate comparisons still have their existing single-person scope.
Source-age metrics add ending-frame workload and sparse worst-age identifiers;
neither these diagnostics nor a completed run constitute a latency SLA or camera,
accuracy, sustained multi-person, 3D, face-blendshape or release approval.
