# Explicit CoreML compute-policy experiment

Definition: a source-PTS-paced single-pose-session comparison of detector/pose
CoreML permitted compute-unit pairs. No live configuration is promoted.

Measured motivation: r5 same-frame dual pose sessions preserved all outputs but
had non-repeatable age improvement. A1/B1/B2/A2 source-age p95 was
81.51/158.93/43.93/93.78 ms. More sessions are not a demonstrated general win.
The next candidate changes permitted compute units, not models or observations.
Contention is a hypothesis, NOT diagnosed physical GPU/ANE behavior.

Contract: --suite compute-policy has the immutable plan A/B/C/C/B/A:
A ALL/ALL; B CPUAndGPU/CPUAndNeuralEngine; C the reverse. Every arm uses one pose
session, existing every-frame detection, OpenCV LUT, original PTS fixed epoch,
all people, queue bounds and ready handoff. Both CPU-permitted names mean permitted
CoreML units; they are NOT GPU-only/ANE-only or verified physical separation.
Existing OrtModel owns runtime registration, no-retry, explicit CPU-partition
policy and preflight. Unsupported policy is terminal, not retried as ALL or CPU.
--detector-provider/--pose-provider must remain coreml-all for this fixed suite;
conflicting overrides and --diagnose-from are rejected, not ignored. Legacy
suites and live defaults remain selectable and unchanged in execution policy.

Ownership: immutable ExecutionArm records drive factories, started checkpoints,
run metadata and manifest. All results compare against the first ALL/ALL arm,
NOT ground truth. Never label cross-policy comparison as same-provider equality.
Source/model hashes and existing reference observations remain authoritative.
Per-frame hashes cover pose and ordered detector boxes, including all multi-person
slots; slot correspondence is not actor identity. No images or coordinates written.

Accounting: preserve legacy frame_pose_ms as SUM of person calls. Add measured
elapsed_pose_stage_ms and elapsed_detector_stage_ms by ending-frame person count;
neither is process CPU time. Whole measured-loop process CPU time covers all host
threads, excludes setup on success, and is not GPU/ANE or device-power telemetry.
Caught failure records may include teardown; mark that scope rather than claim
pure loop timing. Extra observer work stays present in all arms and within timing.

Acceptance: exact frame/PTS coverage and cleanup; report changed boxes, point
validity and all-person hashes, source-age tails/windows and actual elapsed stages.
No pooled quantiles inferred from aggregate percentiles. Changed outputs require
quality review, and unchanged output does not establish accuracy. A/B/C repeats
reduce simple order effects, not nonlinear heating or external-load effects.
Native device and input-to-display measurements are separate remaining gates.
