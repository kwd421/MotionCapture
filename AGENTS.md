# MotionCapture

MotionCapture is a cross-platform, camera-based markerless motion-capture
project. Build complete, observable user flows rather than isolated model
demos.

## Required reading

- Read `RULES.md` before implementation or review.
- For a substantial feature or architectural change, follow `DEVELOPMENT.md`.
- When a slice specification exists, read only the specification and owners
  relevant to the active slice. Do not treat old notes or plans as authority.

## Current product direction

- Support a dynamic number of cameras; never hard-code a three-camera product
  contract.
- The first capture source is the MacBook built-in front camera. Phone cameras
  and Windows capture are later sources using the same core contracts.
- The initial scope is one performer. Multi-person association is not part of
  the current contract.
- Body, feet, both hands, and articulated fingers are required. Face capture is
  a separate capability and must not be implied by body tracking.
- Keep the core usable on macOS and Windows. Python is the initial orchestration
  and reference implementation; ONNX is the portable inference boundary.
- The GUI must eventually represent room dimensions, camera placement, measured
  capture state, and recommended coverage/sweet spots. User-entered, measured,
  calibrated, derived, and estimated values must remain distinguishable.

## Standing constraints

- Fallbacks are forbidden by default. An alternative mode is valid only when it
  is explicitly selected and visibly reported.
- Do not turn missing, failed, invalid, unsupported, or unknown data into a
  plausible success value.
- Do not claim 3D capture, retargeting, synchronization, or measurement beyond
  the stages actually implemented and verified.
- Use maintained libraries for solved infrastructure and measure before
  optimizing.
- Do not commit or push unless explicitly requested. Preserve user-owned work
  and private capture data.

No implementation slice is active merely because it appears in the product
direction above. Define its contract and acceptance criteria before substantial
implementation.
