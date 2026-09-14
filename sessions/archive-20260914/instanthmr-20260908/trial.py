"""Bounded local experiment; no production import or default changes.

Only this process replaces the StagePipeline pose callback with the MHR70 codec.
Upstream pinned preprocessing is authoritative. No invented landmark confidence.
"""
import argparse
import collections
import gc
import hashlib
import json
import platform
import resource
import subprocess
import time
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from upstream_inference import InstantHMR

from motioncapture.recording import RecordedDecoder, inspect_recording
from motioncapture.recording_bench import Samples
from motioncapture.wholebody_catalog import ASSETS, verify_asset
from motioncapture.wholebody_onnx import OrtModel, profile_placement
from motioncapture.wholebody_replay import SourcePacer
from motioncapture import wholebody_stages as stages

ROOT = Path('sessions/instanthmr-20260908')
MODEL = Path('models/instanthmr/instanthmr-3504446.onnx')
EXPECTED = [(1,204),(1,45),(1,3),(1,70,2),(1,70,3)]


class HMRModel:
    def __init__(self, mode='coreml-strict'):
        assert ort.__version__ == '1.29.0'
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.add_session_config_entry('session.intra_op.allow_spinning', '0')
        so.add_session_config_entry('session.inter_op.allow_spinning', '0')
        so.add_free_dimension_override_by_name('batch', 1)
        if mode == 'coreml-strict':
            so.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
        so.enable_profiling = True
        so.profile_file_prefix = str(ROOT / f'preflight-{time.time_ns()}')
        opts = {'ModelFormat':'MLProgram','MLComputeUnits':'ALL',
                'RequireStaticInputShapes':'1','EnableOnSubgraphs':'0'}
        providers = ['CPUExecutionProvider'] if mode == 'cpu' else [('CoreMLExecutionProvider',opts)]
        expected = 'CPUExecutionProvider' if mode == 'cpu' else 'CoreMLExecutionProvider'
        self.session = ort.InferenceSession(str(MODEL), sess_options=so, providers=providers)
        self.session.disable_fallback()
        assert expected in self.session.get_providers(), 'provider substitution'
        # Reuse only the pinned upstream preprocessing owner, not its provider fallback.
        self.codec = InstantHMR.__new__(InstantHMR)
        self.codec.cliff_focal = False
        self.codec.focal = None
        assert self.session.get_modelmeta().custom_metadata_map == {}, 'model metadata changed'
        feed = {'image':np.zeros((1,3,224,224),np.float32),
                'cliff_cond':np.array([[0.,0.,.8]],np.float32)}
        for _ in range(5):
            self.validate(self.session.run(None,feed))
        events = json.loads(Path(self.session.end_profiling()).read_text())
        self.metadata = {'mode':mode,'runtime':ort.__version__, 'requested':providers,
                         'placement':profile_placement(events,expected,False),
                         'model_sha256':hashlib.sha256(MODEL.read_bytes()).hexdigest(),
                         'batch':1,'profiling_during_timed_pass':False,
                         'conditioning':'published pixel-based CLIFF; no calibrated focal supplied',
                         'coordinates':'estimated body-centred metres, Y-down; not calibrated 3D',
                         'joint_confidence':'not provided by model',
                         'mhr_decode':False}

    @staticmethod
    def validate(values):
        assert [v.shape for v in values] == EXPECTED, 'unexpected model schema'
        assert all(np.isfinite(v).all() for v in values), 'nonfinite output'

    def close(self):
        self.session = None
        gc.collect()


def hmr_pose(session, submitted, detected, *_unused):
    start = time.perf_counter_ns()
    rgb = cv2.cvtColor(detected.frame.image_bgr, cv2.COLOR_BGR2RGB)
    h,w = rgb.shape[:2]
    pre = (time.perf_counter_ns()-start)/1e6
    infer = post = 0.
    people, per_person = [], []
    for box in detected.boxes:
        t = time.perf_counter_ns()
        assert np.isfinite(box).all() and np.all(box[2:]>box[:2])
        crop,x,y,size,cond = session.codec._preprocess(rgb, box, h, w)
        a = time.perf_counter_ns()
        values = session.session.run(None,{'image':crop[None], 'cliff_cond':cond[None]})
        b = time.perf_counter_ns()
        session.validate(values)
        xy = (values[3][0]+1)*.5*size + np.array([x,y],np.float32)
        assert np.isfinite(xy).all()
        people.append({'xy':xy, 'outputs':[v[0] for v in values]})
        c = time.perf_counter_ns()
        pre += (a-t)/1e6
        infer += (b-a)/1e6
        post += (c-b)/1e6
        per_person.append((b-a)/1e6)
    end = time.perf_counter_ns()
    times = {**detected.times, 'pose_pre_ms':pre,'pose_inference_ms':infer,
             'pose_post_ms':post,'pose_stage_ms':(end-start)/1e6,
             'pose_queue_ms':(start-submitted)/1e6,
             'detector_to_pose_wait_ms':(start-detected.completed_ns)/1e6,
             'decode_read_ms':detected.frame.decode_ms,
             'submit_to_pose_completion_ms':(end-detected.submitted_ns)/1e6}
    return stages.Posed(detected,people,times,tuple(per_person),end)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('input',type=Path)
    parser.add_argument('name')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--paced',action='store_true')
    parser.add_argument('--pose-mode',choices=['coreml-strict','cpu'],default='coreml-strict')
    args=parser.parse_args()
    out=ROOT/f'{args.name}.json'
    assert not out.exists()
    cv2.setNumThreads(1)
    probe=inspect_recording(args.input)
    count=min(args.limit or len(probe.pts),len(probe.pts))
    assert count>0
    det,_=verify_asset(Path('models/wholebody'),'yolox-tiny')
    pacer=SourcePacer(count) if args.paced else None
    pipe=stages.StagePipeline(partial(OrtModel,det,ASSETS['yolox-tiny'].shape,
                                     'coreml-all',allow_cpu=False),
                              partial(HMRModel,args.pose_mode),pacer=pacer)
    stages.pose=hmr_pose  # This standalone process only; StageOwner lifecycle unchanged.
    decoder=RecordedDecoder(args.input,probe)
    r={'status':'running','source':probe.summary(), 'name':args.name,
       'scope':'offline headless all-frame pipeline; every-frame detector, serial people',
       'pose_mode':args.pose_mode,'paced':args.paced,'limit':count,
       'python':platform.python_version(),'platform':platform.platform(),
       'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
       'opencv_threads':cv2.getNumThreads(),'raw_frames_saved':False,
       'joint_coordinates_saved':False,'accuracy_verified':False}
    stats=collections.defaultdict(Samples)
    counts=collections.Counter()
    pixels,boxes,poses=hashlib.sha256(),hashlib.sha256(),hashlib.sha256()
    frames=people=0
    age_last=None
    started=cpu_start=None
    try:
        with pipe,decoder:
            r['models']=pipe.metadata
            started=time.perf_counter();cpu_start=time.process_time()
            for packet in pipe.packets(decoder,count,overlap=True,fast=False,
                                       advance_pose=True,verify_eof=count==len(probe.pts)):
                f=packet.detected.frame
                assert f.identity.sequence==frames
                pixels.update(f.image_bgr.tobytes())
                boxes.update(packet.detected.boxes.tobytes())
                for person in packet.people:
                    for value in person['outputs']:poses.update(value.tobytes())
                assert len(packet.people)==len(packet.detected.boxes)
                for name,value in packet.times.items():stats[name].add(value)
                if packet.detected.source_release is not None:
                    age_last=(time.perf_counter_ns()-packet.detected.source_release.due_ns)/1e6
                    stats['scheduled_source_to_validated_ms'].add(age_last)
                frames+=1;people+=len(packet.people);counts[len(packet.people)]+=1
                if frames%600==0:print(json.dumps({'name':args.name,'frames':frames,
                    'elapsed_s':time.perf_counter()-started,'last_age_ms':age_last}),flush=True)
            r['loop_s']=time.perf_counter()-started
            r['cpu_s']=time.process_time()-cpu_start
            assert frames==count
            assert count!=len(probe.pts) or decoder.complete
        r['status']='completed'
    except KeyboardInterrupt:
        r['status']='interrupted'
    except Exception as exc:
        r['status']='failed';r['error']=f'{type(exc).__name__}: {exc}'
    finally:
        r.update(frames=frames,person_observations=people,person_counts=dict(counts),
                 input_pixels_sha256=pixels.hexdigest(),boxes_raw_sha256=boxes.hexdigest(),
                 outputs_sha256=poses.hexdigest(),pipeline=pipe.snapshot(),
                 decoder_released=decoder._capture is None,decoder_eof=decoder.complete,
                 peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
                 final_age_ms=age_last,stages={k:v.summary(1000/60) for k,v in stats.items()})
        if r.get('loop_s'):r['frames_per_s']=frames/r['loop_s']
        out.write_text(json.dumps(r,indent=2))
        print(json.dumps({k:r[k] for k in ('name','status','frames','frames_per_s','final_age_ms') if k in r}),flush=True)
    return 130 if r['status']=='interrupted' else 0 if r['status']=='completed' else 1


if __name__=='__main__':
    raise SystemExit(main())
