# Whole-body ONNX benchmark slice

Baseline: `669e7c1af6f381413ac9d819aeae17c6f34c9f9f`.
Definition: evaluate alternate **2D** whole-body models on the existing integer-PTS
recording reader, using in-process ONNX Runtime rather than browser transport.

## Contract
- Candidates: DWPose-m, RTMW-m and RTMW-l, all 256x192 person inputs, 133 points.
  Common detector: official YOLOX-tiny COCO, 416x416. Include detector,
  affine crop, normalization, output decoding and all detected persons' cost.
- A detector miss is an empty observation, NEVER a full-frame person crop.
  All source frames are read sequentially. No interpolation, seeking, detector
  cadence, stale results, artificial third coordinates or facial coefficients.
- Separate 17 body, 6 foot, 68 face and 21+21 hand points. 133-point inference is
  not equivalent to MediaPipe's 478 face points, blendshapes or monocular world
  estimates. Calibrated 3D, retargeting, live display and persistent actor IDs
  remain unsupported. Optional native controls use the original full tracker.
- Provider modes are explicit. CPU is always an independently selected mode.
  CoreML modes default to strict ORT partitioning (no CPU EP nodes). The separate
  `--allow-cpu-partitions` flag explicitly permits CoreML+ORT CPU partitioning.
  Reject absent CoreML EP, provider substitution and zero CoreML nodes in the
  bounded preflight profile. CoreML compute-unit selection does NOT prove ANE
  or GPU dispatch. No automatic retry or model substitution.
- Official download URLs are catalogued. No upstream full-file SHA-256 was
  available for these archives during preparation. `fetch --research-only`
  explicitly enrolls first-download hashes in a local lock (TOFU, NOT publisher
  authentication). Every later run rechecks bytes against that lock. No weights
  or research videos are redistributed by this patch.
- Apache-2.0 repository code is not a release clearance for the selected weights,
  upstream dependencies or training-data terms. Every candidate is marked
  commercial-release **unreviewed**. `--research-only` records intent and
  acknowledgement, NOT a license grant. No release-approved flag is fabricated.
- Output never overwrites an existing file. A failed arm retains its processed
  prefix, current PTS, failure phase and cleanup evidence. Other arms may be
  independently attempted; overall status is not completed if any failed.
- Sampling and counts are not accuracy. Common 133-point outputs may be compared
  between providers against each model's first CPU pass; no cross-schema hash
  equality requirement against MediaPipe. Multi-person correspondence is not
  guessed; those frame comparisons are marked ambiguous.

## Acceptance
1. Same recording hash, actual PTS and whole/prefix counts are visible.
2. Download corruption, wrong shapes, unavailable providers and missing assets
   are failures before measured inference, not empty/zero successful results.
3. Real OpenCV geometry tests plus deterministic orchestration/provider tests.
4. Native ONNX/M1/M5/Windows and actual pretrained quality remain separate gates.
5. No changes to live defaults, existing model weights or previous benchmarks.
