import json
import time
from functools import partial
from pathlib import Path

import cv2
import onnxruntime as ort
from motioncapture.recording import inspect_recording
from motioncapture.wholebody_catalog import ASSETS,verify_asset
from motioncapture.wholebody_onnx import OrtModel
from motioncapture.wholebody_pose_batch import PoseBatchModels
from motioncapture.wholebody_pose_batch_pipeline_lab import BatchStagePipeline
from motioncapture.wholebody_optimize_bench import parser,run_pass

ROOT=Path('sessions/dwpose-followup-20260908')
assert ort.__version__=='1.29.0'
cv2.setNumThreads(1)
args=parser().parse_args(['benchmarks/inputs/phone-1080p60-20260906_030954.mp4',
    '--research-only','--suite','compute-policy','--max-frames','1200',
    '--output',str(ROOT/'policy.json')])
assert not args.output.exists()
probe=inspect_recording(args.input)
det,_=verify_asset(args.asset_dir,'yolox-tiny');pose,_=verify_asset(args.asset_dir,'dwpose-m')
plan=[('A1','coreml-all','coreml-all'),('B1','coreml-gpu','coreml-ane'),
      ('C1','coreml-ane','coreml-gpu'),('C2','coreml-ane','coreml-gpu'),
      ('B2','coreml-gpu','coreml-ane'),('A2','coreml-all','coreml-all')]
rows=[];bank=None
for name,d,p in plan:
    args.detector_provider=d;args.pose_provider=p
    start=None
    def consume(packet):
        global start
        if start is None:start=time.perf_counter()
        if time.perf_counter()-start>120:
            raise RuntimeError('explicit candidate processing budget exceeded; not a fallback')
    print(json.dumps({'starting':name,'detector':d,'pose':p}),flush=True)
    row,enrolled=run_pass(args,probe,'source-pts-ready-cvlut',
        (partial(OrtModel,det,ASSETS['yolox-tiny'].shape,d,allow_cpu=False),
         partial(PoseBatchModels,pose,ASSETS['dwpose-m'].shape,p,allow_cpu=False)),
        reference=bank,pipeline_factory=BatchStagePipeline,packet_consumer=consume)
    row['trial_arm']={'name':name,'detector':d,'pose':p,'ort':ort.__version__,'opencv_threads':cv2.getNumThreads()}
    (ROOT/f'policy-{name}.json').write_text(json.dumps(row,indent=2));rows.append(row)
    if enrolled is not None and row['status']=='completed':bank=enrolled
    print(json.dumps({'finished':name,'status':row['status'],'loop':row.get('loop'),
                      'pose':row.get('full',{}).get('stages',{}).get('pose_stage_ms')}),flush=True)
    # A failed explicitly selected arm is terminal for this fixed plan.
    if row['status']!='completed':break
args.output.write_text(json.dumps({'runtime':ort.__version__,'plan':plan,'runs':rows},indent=2))
