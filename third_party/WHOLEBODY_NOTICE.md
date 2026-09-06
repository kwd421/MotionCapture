# Model-specific codec attribution and licensing boundary

`src/motioncapture/wholebody_onnx.py` contains adapted (not verbatim) model-specific
pre/postprocessing from the following Apache-2.0 upstream projects. Original
copyrights remain with their authors. Apache-2.0.txt accompanies this patch.

- RTMLib, Tau Jiang and contributors, revision
  `03a1693e59e4f7cd84582c0fb30459b3bf18ad42` (0.0.16):
  https://github.com/Tau-J/rtmlib/tree/03a1693e59e4f7cd84582c0fb30459b3bf18ad42
  `rtmlib/tools/pose_estimation/pre_processings.py`,
  `rtmlib/tools/pose_estimation/post_processings.py`,
  `rtmlib/tools/pose_estimation/rtmpose.py`,
  `rtmlib/tools/object_detection/yolox.py`.
- MMPose, Copyright OpenMMLab:
  https://github.com/open-mmlab/mmpose/blob/main/LICENSE
- DWPose / IDEA Research:
  https://github.com/IDEA-Research/DWPose/blob/onnx/LICENSE
- YOLOX / Megvii:
  https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE

Changes: strict fixed tensor schemas and failure checks; no full-frame fallback
on detector miss; select only COCO person; OpenCV continuous-IoU NMS rather than
RTMLib's hand-written NMS; fixed-rotation affine transform; owned arrays and
provider instrumentation. BGR mean/std, 1.25 padding, split ratio 2 and RTMLib's
2D mean-axis SimCC score convention are explicit. These changes require visual
validation on real pretrained output and are not claimed byte-identical to an
upstream end-to-end pipeline.

The ONNX Runtime runtime is MIT licensed. ONNX is Apache-2.0. No runtime binaries
or pretrained model weights are redistributed here. Code licenses do NOT clear
upstream checkpoint/training-dataset terms. Catalog status is unreviewed for all
selected weights. RTMW Cocktail14 includes InterHand, MPII, UBody, Human-Art and
other datasets; their usage terms need individual review for a commercial release.
No claim is made that dataset conditions automatically transfer to every model;
that question is explicitly unresolved for this product, not silently assumed.
