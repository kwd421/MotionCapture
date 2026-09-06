# Bounded detector-completion handoff

Definition: optional single-pose-owner continuation on an already admitted
next-frame detector request, so a detection completing during result validation
can start pose without waiting for the result consumer to resume.

Evidence: r6's two ALL/ALL controls spent ~2.1ms/frame validating outputs and
observed the next detector not ready at most handoff checks. The old path checks
readiness only once before yielding. Code establishes that a later completion
cannot trigger pose until the consumer resumes; r6 aggregate data does not
measure how many completions fall inside that validation interval. This candidate
removes that scheduling dependency; target-device speed/age improvement is pending.
Explicit GPU/ANE-permitted policies are not promoted after worse r6 source ages.

Contract: preserve one detector and one pose session, unchanged model/provider/
precision/crops/thresholds, every original PTS/frame/person and ordered output.
No new source admission, queue depth, actor selection, extra model, future result
reuse, skipping, interpolation or epoch reset. Old modes remain unchanged.

The detector StageOwner is still caller-owned for submit/receive and retains its
outstanding Future until caller acknowledgement. A bound read-only result getter
captures THAT request, not the mutable owner; it does not acknowledge or cancel.
The already idle pose owner may wait on this getter, then executes pose on its
own thread. Detector never waits on pose: there is no same-executor or cyclic
wait. The caller independently receives the identical detector result before
admitting a further source frame. Model setup/run/destruction remain thread-affine.

This path is explicitly incompatible with the two-lane pose experiment. An error
or early consumer stop cancels scheduled source waits; borrowed-result errors
propagate. Native calls cannot be force-cancelled. Close joins detector first,
then the dependent pose owner; no destroyed native session is used by a waiter.
No CPU/ALL/serial fallback is introduced.

Metrics: source ages still include the complete fixed source schedule. The new
pose_dependency_wait_ms includes any scheduled source wait and detector execution
remaining when pose's continuation enters; it is NOT extra model time. New
pose_owner_dispatch_ms records executor dispatch before the dependency wait.
Existing pose_queue_ms means original pose request submission to pose-body entry;
in the candidate it includes the dependency wait. pose_stage_ms still measures
actual pose-body elapsed time. Do not call reduced submit gaps inferred speedup.

Acceptance: event-controlled completion DURING consumer validation starts pose
before consumer resumes; exact-N/order/all-person/hash/PTS/no-extra-session tests,
full EOF/prefix handling, fixed borrowed-request identity despite later submits,
failed producer, failed consumer and cancelled wait cleanup. Full-file paced
ready/dependency/dependency/ready comparison with all hashes and ages retained.
Test-fixture scheduling progress is not real CoreML speed or live camera approval.
