# WholeBody ONNX candidate benchmark

## Definition
An additive, explicit-file R&D runner comparing DWPose-m and RTMW-m/l with a
fixed CPU YOLOX-tiny detector and selectable native ONNX Runtime pose providers.
Parent: 669e7c1af6f381413ac9d819aeae17c6f34c9f9f. Live defaults are unchanged.

## Contract and owners
- `wholebody_assets` owns fixed publisher-mirror revision, archive SHA256, safe
  single-ONNX extraction and explicit download/verification. No inference-time
  fetch, mirror substitution, executable checkpoint, or model replacement.
- `wholebody_onnx` owns source-pixel affine transforms, BGR normalization,
  SimCC decoding and strict sessions. All detected persons are processed; zero
  detections never becomes a full-frame guessed person. A full-frame placement
  probe is separately labelled and excluded from measured observations.
- The existing `recording` owner remains PTS/decoder authority. Every selected
  original frame is decoded once per pass. No interpolation, skipping or seeking.
- `wholebody_bench` owns the serial experiment plan, per-pass model lifetime,
  counters and immutable result files. Repeats reverse the candidate order.
  Each provider/model failure is recorded; other explicitly requested cells
  remain independent experiments, not automatic fallbacks. KeyboardInterrupt
  stops the plan. Source integrity failure stops the plan.
- Default pose modes: CPU and CoreML ALL. GPU/ANE-labelled modes select Apple's
  CPUAndGPU/CPUAndNeuralEngine compute policy, not GPU-only/ANE-only execution.
  ORT CPU graph fallback is disabled for CoreML. One real-input profiling probe
  checks observed ORT kernel providers; physical dispatch within CoreML remains
  unverified. Detector CPU is an explicit fixed stage in all modes.
- Reports contain no frames, individual coordinates, private paths or tokens.
  Same-model CPU reference coordinates may exist only in bounded process RAM.
  Full source/model hashes and aggregate numerical disagreement are recorded.

## Output scope, not equivalence
COCO-WholeBody133 = body17 + feet6 + face68 + hands21 each; image XY only.
World XYZ, blendshapes, bone rotations, performer identity and calibrated 3D are
unsupported. Never construct fake MediaPipe33/478/world output for compatibility.
Optional native MediaPipe controls compute MORE and DIFFERENT capabilities and
therefore are not feature-equivalent speed benchmarks. Confidence coverage is
not accuracy; CPU/GPU agreement is not ground truth. Compare scores only within
one model; thresholds are not calibrated across different models.

## Boundaries and failure semantics
All camera pixels remain full source resolution before the model's required
416-square detector letterbox / 256x192 person crop. Those resamplings are
explicit. Detector executes every frame; all person crops count toward timings.
No-point frames are valid no-detection observations, not errors or quality wins.
Unknown export topology, malformed output, nonfinite values, unsafe archive,
checksum mismatch and unexpected provider execution fail that candidate.
Partial runs retain counters but never report successful FPS. Each cell writes
an intent record before model construction and a terminal record on completion
or caught failure. Native aborts/hangs are NOT isolated by subprocess watchdogs;
prior terminal files survive and an unmatched intent record indicates a crash.

## Acceptance
Runtime schema/placement validation, expected selected frame counts, finite
source coordinates, exact PTS, explicit capability and model provenance, no
silent CPU graph fallback, and acknowledged cleanup are mandatory. Repeats must
be reviewed for degradation. FPS alone cannot promote a model. This patch does
not claim commercial clearance or 60FPS; M1/M5/Windows native inference remains
a hardware gate. Test doubles only establish the orchestration contract.
