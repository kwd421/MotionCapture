# Development workflow

Use this workflow for substantial features and architectural changes.

## Specification-driven slices

Before implementation, define only the durable facts needed for the slice:

- one-line definition;
- current contract;
- non-goals;
- authoritative owners and failure states;
- acceptance criteria;
- verification commands or user-flow path.

Freeze expensive public boundaries, not every internal detail. Implement one
narrow production-shaped vertical slice using the final boundary and types.
Create a separate specification or decision record only when it owns an
independent, durable fact; do not create documents merely to copy another
document or preserve a running diary.

Record verified facts separately from observations, inferred causes, decisions,
and unknowns. Update status only after the owning code/runtime and real
verification establish the new state.

## Slice workflow

1. Define the user, problem, scope, non-goals, and observable completion.
2. Identify authority boundaries, data ownership, failure states, concurrency
   semantics, persistence, protocols, and verification that are expensive to
   reverse.
3. Implement one real user action end-to-end using the intended production
   boundary. Do not create inactive abstractions, placeholder owners, or future
   fallback paths.
4. Verify the core path, realistic edge cases, and hard-stop failures. Re-run the
   actual user flow before closing the slice.

## Verification

Do not add tests merely because code changed. Keep a small number of high-value
tests for coordinate transforms, timestamps, calibration validity, model schema,
triangulation, failure propagation, skeleton invariants, and export contracts.

A build, lint, static check, model load, or unit test proves only its observed
scope. Skipped, unsupported, untested, contaminated, or visually plausible
results remain `unknown`, not clean.

Verify substantial slices through a real user path with representative camera
or recorded input. Diagnostics should expose at least the relevant subset of:

- per-camera timestamps, FPS, frame drops, and queue depth;
- selected inference provider and whether every model node ran as intended;
- model identity and checksum;
- calibration identity and reprojection error;
- synchronization offsets and drift;
- landmark confidence and visibility;
- triangulation residuals and rejected observations;
- bone-length variation, joint constraint violations, and foot contact state;
- stage latency and end-to-end latency.

## Performance and optimization records

Measure before optimizing. For completed slices, inspect relevant CPU, GPU/ANE,
memory, latency, task/process, thermal, and disk costs. An intuition about future
scale is not evidence.

Record every material optimization next to a durable benchmark, decision, or
implementation note with:

- the prior measured cost, symptom, or concrete threat;
- the owning boundary and intent;
- the product and safety invariants preserved;
- the accepted trade-off;
- the measurement and verification result.

Remove avoidable work and copying at the owning boundary, but do not exchange
correctness, observability, security, or maintainability for speculative speed.

## Session records and privacy

Every reproducible capture or benchmark session should be able to emit a
machine-readable manifest containing the applicable subset of:

- application version or source revision;
- OS, hardware, runtime, and inference provider;
- model identifier/checksum;
- camera identifiers, formats, resolution, and nominal/observed FPS;
- calibration identifier/checksum;
- active configuration and explicit operating mode;
- synchronization result;
- stage timings, quality metrics, warnings, and terminal errors.

Logs must be structured enough to correlate events by session, camera, and
frame/timestamp. Do not log raw frames, biometric imagery, credentials, tokens,
private paths, or user content by default. Recording or retaining such data must
be explicit and visible.
