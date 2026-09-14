import contextlib,pickle,sys
from functools import partial
import cv2
import onnxruntime as ort
from motioncapture.recording import inspect_recording
from motioncapture.wholebody_catalog import ASSETS,verify_asset
from motioncapture.wholebody_onnx import OrtModel
from motioncapture.wholebody_pose_batch import PoseBatchModels
from motioncapture.wholebody_pose_batch_pipeline_lab import BatchStagePipeline
from motioncapture.wholebody_optimize_bench import parser,run_pass
assert ort.__version__==sys.argv[1]
args=parser().parse_args(['benchmarks/inputs/phone-1080p60-20260906_030954.mp4','--research-only','--allow-cpu-partitions','--suite','pose-parallel','--max-frames','1200','--output','sessions/unused-quality-worker.json'])
cv2.setNumThreads(1);probe=inspect_recording(args.input)
det,_=verify_asset(args.asset_dir,'yolox-tiny');pose,_=verify_asset(args.asset_dir,'dwpose-m');packets=[]
def consume(p):
 f=p.detected.frame.identity
 packets.append((f.sequence,f.pts,str(f.time_base),p.detected.boxes.copy(),[(v.xy.copy(),v.scores.copy(),v.valid.copy()) for v in p.people]))
factories=(partial(OrtModel,det,ASSETS['yolox-tiny'].shape,'coreml-all',allow_cpu=True),partial(PoseBatchModels,pose,ASSETS['dwpose-m'].shape,'coreml-all',allow_cpu=True))
with contextlib.redirect_stdout(sys.stderr):
 row,_=run_pass(args,probe,'overlap-ready-cvlut',factories,pipeline_factory=BatchStagePipeline,packet_consumer=consume)
assert row['status']=='completed' and len(packets)==1200, row['error']
pickle.dump({'version':ort.__version__,'packets':packets,'row':row},sys.stdout.buffer,protocol=5)
