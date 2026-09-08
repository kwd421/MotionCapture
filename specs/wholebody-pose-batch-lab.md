# Same-frame pose batch feasibility lab

Definition: measure whether the existing DWPose-m ONNX graph benefits from a
static batch-2 specialization when two detector slots occur in the SAME source
frame. This is an isolated research benchmark, not a live-pipeline change.

## Contract

- Scan every original source frame with the existing YOLOX-tiny detector.
- Keep detector=CoreML ALL and pose=CoreML ALL with explicit CPU partitions.
- For frames with >=2 detector boxes, compare the first two slots only.
- A = two ordinary batch-1 pose calls. B = one batch-2 pose call.
- Per pair order is A/B/B/A. No future-frame batching, padding, frame skipping,
  detector cadence reduction, stale poses, model/precision/crop changes, or retry.
- Batch-2 is a second static specialization of the same ONNX file. Failure is
  terminal; it never retries as two batch-1 calls.
- This lab does not change the current ready-handoff ALL/ALL reference path.
- Cleanup attempts all created model owners. A cleanup failure cannot leave a
  completed status or zero exit code; an earlier primary failure is preserved.
  Interrupted runs return 130 and other failures return 2.

The published export config declares the pose batch dimension dynamic, and the
current downloaded model already reports a symbolic `batch` binding in batch-1
preflight. The lab requires that symbolic dimension and binds it explicitly to 2.
It verifies the requested CoreML provider using the same ONNX Runtime 1.22.1
provider plan and zero-tensor placement preflight used by the existing lab.
Physical GPU/ANE routing remains unverified.

## Evidence to inspect

The result records pair wall time for serial A and batch-2 B, plus aggregate
coordinate/validity disagreement against the first A observation. A is not
truth. The scan/discovery detector cost is intentionally outside the pair speed
comparison. Hashes from A and B are not directly comparable because each digest
contains a different repeated-call ordering; use coordinate disagreement fields.
No raw frames, pose tensors or individual coordinates are written.

A useful result is a repeatable reduction of `batch2_pair_ms` versus
`serial_pair_ms` without unacceptable coordinate/validity disagreement. Even a
large pair speedup does not prove full-pipeline or live 60 FPS because this source
contains only a small fraction of multi-box frames. If batch-2 is not materially
faster, do not integrate it into the production-shaped pipeline.
