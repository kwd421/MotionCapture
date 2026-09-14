"""Same preview with bounded state/stack observations and explicit activity A/B."""
import argparse
import faulthandler
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import AppKit
import Foundation
import objc
from motioncapture import wholebody_recorded_preview as app

p=argparse.ArgumentParser()
p.add_argument('name')
p.add_argument('--source',choices=['phone','macbook'],default='phone')
p.add_argument('--activity',choices=['off','user-initiated'],default='off')
p.add_argument('--hide-after',type=int)
p.add_argument('--max-frames',type=int,default=0)
p.add_argument('--backend',choices=['opencv','sdl'],default='opencv')
args=p.parse_args()
root=Path('sessions/dwpose-followup-20260908')
out=root/f'{args.name}.json'
assert not out.exists()
state={'in_preview':False,'completed':0,'last_progress':None,'entered':None}
stop=threading.Event()
process=Foundation.NSProcessInfo.processInfo()
token=None
observations=[]
activity={'selected':args.activity,'started':False,'ended':False,'pyobjc':objc.__version__}
original=app.Preview.__call__
trace=(root/f'{args.name}-state.jsonl').open('x')
stacks=(root/f'{args.name}-python-stacks.log').open('x')


def snapshot(self):
    native=AppKit.NSApplication.sharedApplication()
    return {'time_ns':time.perf_counter_ns(),'frames':self.frames,
            'app_active':bool(native.isActive()),'app_hidden':bool(native.isHidden()),
            'windows':[{'number':int(w.windowNumber()),'visible':bool(w.isVisible()),
                        'miniaturized':bool(w.isMiniaturized()),
                        'occlusion_state':int(w.occlusionState())} for w in native.windows()],
            'thermal_state':int(process.thermalState()),
            'low_power_mode':bool(process.isLowPowerModeEnabled()),
            'activity':args.activity}


def observed(self,packet):
    entered=time.perf_counter()
    state.update(in_preview=True,entered=entered)
    try:
        original(self,packet)
    finally:
        end=time.perf_counter()
        state.update(in_preview=False,last_progress=end,completed=self.frames)
        if self.frames==1 or self.frames%60==0 or end-entered>.25:
            row=snapshot(self);row['preview_call_ms']=(end-entered)*1000
            observations.append(row);trace.write(json.dumps(row)+'\n');trace.flush()
        if args.hide_after is not None and self.frames==args.hide_after:
            AppKit.NSApplication.sharedApplication().hide_(None)
            row=snapshot(self);row['action']='explicit test hides only this process app'
            observations.append(row);trace.write(json.dumps(row)+'\n');trace.flush()
        if self.frames==1:
            with (root/f'{args.name}-launch-info.txt').open('x') as target:
                subprocess.run(['lsappinfo','info',f'#{os.getpid()}'],stdout=target,
                               stderr=subprocess.STDOUT,timeout=5,check=False)


def watch():
    captures=0;last_capture=0.
    while not stop.wait(.25):
        last=state['entered'] if state['in_preview'] else state['last_progress']
        now=time.perf_counter()
        if last is None or now-last<1 or captures>=2 or now-last_capture<10:continue
        last_capture=now;captures+=1
        row={'stall_observed_s':now-last,'state':dict(state),'time_ns':time.perf_counter_ns()}
        stacks.write(json.dumps(row)+'\n');stacks.flush()
        faulthandler.dump_traceback(file=stacks,all_threads=True)
        with (root/f'{args.name}-sample-{captures}.log').open('x') as log:
            subprocess.run(['sample',str(os.getpid()),'1','1','-file',
                            str(root/f'{args.name}-sample-{captures}.txt')],stdout=log,
                           stderr=subprocess.STDOUT,timeout=10,check=False)


app.Preview.__call__=observed
worker=threading.Thread(target=watch,name='bounded-stall-observer')
source={'phone':'phone-1080p60-20260906_030954.mp4',
        'macbook':'macbook-720p30-20260905T175716Z.mp4'}[args.source]
config=argparse.Namespace(input=Path('benchmarks/inputs')/source,provider='coreml-all',
    allow_cpu_partitions=True,research_only=True,asset_dir=Path('models/wholebody'),
    max_frames=args.max_frames,decode_threads=0,opencv_threads=1,display_backend=args.backend,
    expected_ort_version='1.29.0',output=out,snapshot_frame=[])
worker.start()
try:
    if args.activity=='user-initiated':
        token=process.beginActivityWithOptions_reason_(
            Foundation.NSActivityUserInitiatedAllowingIdleSystemSleep,
            'MotionCapture explicitly selected recorded preview')
        assert token is not None,'native activity token missing'
        activity['started']=True
    result=app.execute(config)
finally:
    stop.set();worker.join()
    if token is not None:
        process.endActivity_(token);activity['ended']=True;token=None
    trace.close();stacks.close()
    (root/f'{args.name}-observer.json').write_text(json.dumps({
        'activity':activity,'hide_after':args.hide_after,'observations':observations,
        'observer_cost':'same frame callback in all arms; native sample only after >1s stall, at most twice',
        'source_to_photon_verified':False},indent=2))
raise SystemExit(result)
