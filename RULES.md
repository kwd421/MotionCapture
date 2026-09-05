# Engineering rules

These rules define mandatory invariants for implementation and review.

## Evidence and authority

Do not infer repository-specific facts from convention. Inspect code,
configuration, model metadata, camera capabilities, timestamps, and real runtime
behavior. If a fact is not verified, keep it `unknown`.

Maintain one authoritative representation for each of the following:

- coordinate system, handedness, axes, units, and transforms;
- skeleton and joint naming;
- camera identity and capabilities;
- intrinsic and extrinsic calibration;
- capture timestamps and synchronization mapping;
- model input/output schema;
- session configuration and lifecycle state.

Serialization, GUI state, validation, exports, and diagnostics must derive from
the authoritative owner or explicitly check for disagreement. Do not hide an
authoritative state in combinations of status flags, caches, timers, or UI
booleans.

For spatial values, always retain provenance. A value must be identifiable as
one of: `user_entered`, `measured`, `calibrated`, `derived`, or `estimated`.
Never display an estimate as a measurement.

## No implicit fallbacks

Fallbacks are forbidden by default. Find and fix the root cause. Do not add a
fallback without explicit user approval.

In particular, never silently:

- change CoreML, Windows ML, DirectML, MIGraphX, or another requested inference
  provider to CPU;
- replace an unavailable model with a smaller or different model;
- use default intrinsics or extrinsics when calibration is missing or invalid;
- degrade calibrated multi-view 3D into monocular 3D, 2D, or stale poses;
- ignore a failed or unsynchronized camera and report the requested capture as
  successful;
- replace live capture with prerecorded, generated, placeholder, or cached
  frames;
- convert missing, corrupt, unsupported, or failed data into empty values,
  zeroes, identity transforms, or success.

An explicitly selected CPU mode, monocular mode, offline-file mode, or reduced
camera set is a separate supported path, not a fallback. Its limitations and
active mode must be visible to the user and session record.

Distinguish `failed`, `unsupported`, `invalid`, `missing`, and `unknown`. Preserve
these states through the public result and GUI instead of normalizing them away.

## Honest completeness

Do not present a landmark overlay as completed 3D motion capture. Do not present
3D joint positions as completed retargeted bone rotations. Do not present
approximate finger bends as measured axial finger rotation.

Placeholder data, fake calibration, disabled synchronization, hard-coded camera
geometry, compatibility shims, or client-side reconstruction of a core-owned
contract cannot make a slice complete. An incomplete path remains explicitly
incomplete even when a test passes or a screen looks plausible.

Each demo must state which stages are real:

1. capture;
2. timestamping and synchronization;
3. 2D landmark inference;
4. calibrated 3D reconstruction;
5. constrained skeleton solve;
6. retargeting and export.

## Design and dependencies

Use mature, maintained libraries for solved infrastructure. Do not reinvent
video codecs, camera frameworks, model runtimes, serialization, networking,
linear algebra, calibration primitives, GUI frameworks, or generic concurrency
primitives.

Prefer the smallest boring design that fully preserves the contract. Before
adding a helper, abstraction, compatibility layer, provider-specific branch, or
new dependency, check whether an existing owner or a smaller complete design can
serve the active slice.

Keep platform-specific capture and acceleration behind explicit adapters. Core
pose records and geometry must not contain framework-owned tensors or
platform-specific objects. Heavy inference and numerical work should cross
portable array/model boundaries.

Dynamic camera support means algorithms consume a collection of identified
camera observations. A one-camera 2D path, a calibrated multi-view 3D path, and
an unsupported configuration must be explicit outcomes; do not scatter camera
count conditionals through the GUI and solver.

Dependencies and background work require explicit owners. Camera streams,
decoders, inference workers, timers, queues, network senders, and recording
sessions must have visible creation, cancellation, shutdown, and cleanup owners.

## Time, synchronization, and concurrency

Use presentation/capture timestamps, not decoded frame indices, as time
authority. Preserve original timestamps and record every synchronization or
drift transform applied to them.

Define what `started`, `ready`, `recording`, `synchronized`, `complete`, and
`stopped` mean. Preserve those meanings across retries, cancellation, parallel
camera startup, partial device failure, and application shutdown.

Do not test concurrency by sleeping and hoping. Use controlled events, barriers,
fake clocks, deterministic queues, and observable public outcomes.

## Repository and data safety

Do not commit or push unless explicitly requested. Do not modify, overwrite,
move, or revert user-owned changes. Avoid destructive commands and migrations.
Never expose secrets or private capture data in logs, fixtures, prompts, events,
or committed files.

When commits are requested, keep them independently buildable, verifiable, and
rollbackable. Check the diff before committing and avoid combining architecture,
model schema, capture plumbing, UI, exports, and unrelated cleanup in one commit.
