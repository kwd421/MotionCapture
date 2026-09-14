import hashlib,json,platform,resource
from pathlib import Path
from functools import partial
import cv2
from motioncapture.recording import inspect_recording
from motioncapture.recording_bench import write_report,_source_revision
from motioncapture.wholebody_catalog import ASSETS,verify_asset
from motioncapture.wholebody_onnx import OrtModel
from motioncapture.wholebody_pose_batch import PoseBatchModels
from motioncapture.wholebody_pose_batch_pipeline_lab import BatchStagePipeline,_fix_batch_timing_labels
from motioncapture.wholebody_optimize_bench import parser,run_pass
args=parser().parse_args(['benchmarks/inputs/phone-1080p60-20260906_030954.mp4','--research-only','--allow-cpu-partitions','--suite','pose-parallel','--max-frames','1200','--output','sessions/opt-20260908-graph-abba.json'])
assert not args.output.exists()
cv2.setNumThreads(1);probe=inspect_recording(args.input)
detector,dm=verify_asset(args.asset_dir,'yolox-tiny');original,om=verify_asset(args.asset_dir,'dwpose-m')
candidate=Path('models/wholebody/experiments/dwpose-m-hardsigmoid-slice/model.onnx')
report={'experiment':'algebraic_hardsigmoid_and_static_slice','source':probe.summary(),'source_revision':_source_revision(),'platform':platform.platform(),'opencv_threads':cv2.getNumThreads(),'candidate_sha256':hashlib.sha256(candidate.read_bytes()).hexdigest(),'original_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),'accuracy_verified':False,'live_60fps_verified':False,'pacing':'unpaced','preview':'none','runs':[]}
reference=None
for arm,path in [('a1',original),('b1',candidate),('b2',candidate),('a2',original)]:
 factories=(partial(OrtModel,detector,ASSETS['yolox-tiny'].shape,'coreml-all',allow_cpu=True,threads=4),partial(PoseBatchModels,path,ASSETS['dwpose-m'].shape,'coreml-all',allow_cpu=True,threads=4))
 row,ref=run_pass(args,probe,'overlap-ready-cvlut',factories,reference=reference,pipeline_factory=BatchStagePipeline)
 _fix_batch_timing_labels(row);row['arm']=arm;row['graph']='original' if path==original else 'hardsigmoid+slice';report['runs'].append(row)
 write_report(args.output.with_name(args.output.stem+'-'+arm+'.json'),row)
 if reference is None:reference=ref
 print(json.dumps({'arm':arm,'status':row['status'],'fps':row['unpaced_loop_fps'],'pose_mean_ms':row['all_frames']['stages']['pose_stage_ms']['mean_ms'],'hash_equal':row['reference_hash_equal'],'disagreement':row.get('provider_disagreement')}),flush=True)
 if row['status']!='completed':break
report['process_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
report['memory_scope']='lifetime peak across all arms; not per-arm comparison'
write_report(args.output,report)
