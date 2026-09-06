# Selected asset provenance and release gate

This is a provenance checklist, not a legal opinion or a commercial-use approval.
The preparation inspected repository sources/model listings, **not the downloaded
ONNX bytes**. All four assets remain release-unreviewed.

| Asset | Code license evidence | Training provenance | Exact weight/data release clearance |
| --- | --- | --- | --- |
| YOLOX-tiny COCO | YOLOX Apache-2.0 repository | Official COCO detector export | Unreviewed |
| DWPose-m 256x192 | DWPose Apache-2.0 repository | COCO-WholeBody + UBody | Unreviewed |
| RTMW-m 256x192 | MMPose Apache-2.0 repository | Cocktail14 | Unreviewed |
| RTMW-l 256x192 | MMPose Apache-2.0 repository | Cocktail14 | Unreviewed |

The permissive repository license supports evaluation of the implementation. It
is not recorded as permission from every image owner, data publisher, checkpoint
publisher or downstream asset owner. A noncommercial dataset term does not, by
itself, establish a universal legal conclusion about every trained checkpoint;
neither does an Apache code badge settle that question. Obtain clarification
for the exact checkpoint, product, territory and redistribution plan.

## Sources checked

- DWPose source/model listing and Apache code license:
  https://github.com/IDEA-Research/DWPose
- Official RTMW model/config table, including the names of all Cocktail14 sets:
  https://github.com/open-mmlab/mmpose/blob/main/configs/wholebody_2d_keypoint/rtmpose/cocktail14/rtmw_cocktail14.md
- RTMLib model listing and concrete inference recipe, revision
  `03a1693e59e4f7cd84582c0fb30459b3bf18ad42`:
  https://github.com/Tau-J/rtmlib/tree/03a1693e59e4f7cd84582c0fb30459b3bf18ad42
- YOLOX export implementation:
  https://github.com/Megvii-BaseDetection/YOLOX/blob/main/tools/export_onnx.py
- UBody publisher repository (separate source to review, not bundled code):
  https://github.com/IDEA-Research/OSX
- Human-Art publisher explicitly requests authorization for non-commercial use:
  https://github.com/IDEA-Research/HumanArt

The common detector is deliberately the COCO-only official YOLOX release, not
the convenient HumanArt+COCO detector from RTMLib's default wholebody wrapper.
That narrows the provenance set; it is not a commercial-clearance conclusion.
RTMW itself still uses Cocktail14 including Human-Art/UBody/InterHand/MPII/etc.

## Before release

Identify exact model binary hashes and publisher terms; check all dataset and
third-party component terms relevant to the chosen use; retain notices and
permissions; review redistribution separately. A locally acknowledged
`--research-only` flag grants no rights. Do not promote this asset catalog to
release-approved solely because a runtime benchmark passes.
