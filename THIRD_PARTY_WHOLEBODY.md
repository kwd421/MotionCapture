# WholeBody candidate provenance and license review

Reviewed 2026-09-06. This is a research/dependency record, not legal clearance.
No pretrained weights, training datasets or private capture data are redistributed
in this patch. Explicit `wholebody_assets fetch` obtains the named model files.

## Code and mathematical conventions

The affine/SimCC and YOLOX pre/postprocessing in `wholebody_onnx.py` adapts
RTMLib/OpenMMLab conventions into a restricted, checked array interface.
Changes: explicit session ownership, export-schema validation, no auto-download,
no full-frame fallback for empty detections, no MPS-to-CPU fallback, provider
profiling outside pass timings. It is NOT an unchanged upstream wrapper.

RTMLib: https://github.com/Tau-J/rtmlib
License: Apache-2.0. Upstream license attribution: Copyright 2018-2020 Open-MMLab.
A copy is in `licenses/Apache-2.0.txt`. Retain these notices when redistributing.
Source files inspected (Git blob SHA, so content is identifiable if main changes):

| Upstream path under `rtmlib/tools/` | Blob SHA |
| --- | --- |
| pose_estimation/rtmpose.py | 50fc9119cd5fff8137fa58d5e5af66fddffebc01 |
| pose_estimation/pre_processings.py | 7511190767732050f19a17414b16aa540232c054 |
| object_detection/yolox.py | 5585ef9847ac5f546719fe7585290f7e05d683d8 |
| base.py | 7356ebec976f8cd1094531f80b4728632e2cc116 |

MMPose: https://github.com/open-mmlab/mmpose (Apache-2.0 code).
DWPose: https://github.com/IDEA-Research/DWPose (Apache-2.0 code).
DWPose is a distillation-based model family, NOT a newer name for RTMW.
ONNX Runtime 1.23.2: https://pypi.org/project/onnxruntime/1.23.2/ (MIT package).
Existing OpenCV/NumPy/FFmpeg dependencies and their licenses remain unchanged.
No FreeMoCap AGPL, EasyMocap or MAMMA restricted code is added.

## Models: declarations are not a blanket commercial-safety finding

Official catalog and publisher-maintained mirror:
https://github.com/Tau-J/rtmlib#model-zoo
https://huggingface.co/Tau-J/RTMPose
https://huggingface.co/yzd-v/DWPose

The mirror and DWPose author's model cards declare Apache-2.0. We record that
as a published label, separately from the source-code license. A complete
commercial release review of the exact checkpoints, their upstream notices and
rights has NOT been performed. Training-data restrictions should be inspected;
we neither assume they automatically relicense weights nor assume they cannot
matter. Do not market these candidates as already cleared for commercial use.

DWPose-m is catalogued as COCO-WholeBody + UBody training; RTMW-m/l use Cocktail14;
YOLOX-tiny uses HumanArt + COCO. Specific dataset and third-party asset terms
remain release-review items. No full training provenance audit is claimed.

## Immutable asset identity

Chosen source is the publisher mirror, not an automatic fallback. All four
archives are pinned at revision `cd4d7095f5cfc9cfc4f46289bee91ea4a1e1d9fd`.
Their published SHA256 (not Xet chunk hashes) are embedded in `wholebody_assets`.
The full original OpenMMLab and pinned mirror URLs appear in `assets list`.
The files were NOT downloaded on this execution host; network was unavailable.
Native loading and preprocessing acceptance therefore remain target-device gates.

| Key | Published archive SHA256 |
| --- | --- |
| yolox-tiny | 36e09ca555916253fa1b5d51bec01e48e98e1f0af966c51ce3e844d8c8fc4cfc |
| dwpose-m | a4e4d56e9dc043e2171c73f5918f334509a47fde60182c1961370c7160334ac0 |
| rtmw-m | a6d08a575c9d8b8da7ce6c13e1c7341047fca35594a6d35de04b739eba51139c |
| rtmw-l | 1e3e77558dfc199129bfff1c583e51b4ee190914de6ae30688243c20163c148c |

## CoreML execution-policy evidence

https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html

`CPUAndGPU` and `CPUAndNeuralEngine` explicitly ALLOW CPU within CoreML. Neither
is a GPU-only/ANE-only guarantee. This patch disallows unrequested ORT CPU graph
fallback and checks kernel-provider events, but does not measure physical
GPU/ANE dispatch inside CoreML. Keep `ane_execution_verified=false` until a
separate target-side compute-plan/hardware profiling establishes it.
