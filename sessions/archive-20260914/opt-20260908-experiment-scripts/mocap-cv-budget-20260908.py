import json, subprocess, sys
from pathlib import Path
root=Path('.')
for label, threads in [('a1',10),('b1',1),('b2',1),('a2',10)]:
    output=root/f'sessions/opt-20260908-cv-r2-{label}.json'
    code='import cv2,runpy; cv2.setNumThreads(int(__import__("os").environ["MOCAP_CV_THREADS"])); print("opencv_threads",cv2.getNumThreads(),flush=True); runpy.run_module("motioncapture.wholebody_recorded_preview",run_name="__main__")'
    import os
    env=dict(os.environ, MOCAP_CV_THREADS=str(threads))
    argv=[sys.executable,'-c',code,str(root/'benchmarks/inputs/phone-1080p60-20260906_030954.mp4'),'--provider','coreml-all','--allow-cpu-partitions','--research-only','--max-frames','1200','--output',str(output)]
    with output.with_suffix('.log').open('x') as log:
        r=subprocess.run(argv,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
    d=json.loads(output.read_text()); p=d.get('preview',{}); row=d.get('run',{})
    print(json.dumps({'arm':label,'cv_threads':threads,'exit':r.returncode,'status':d['status'],'fps':p.get('mean_submission_rate_hz'),'compose':p.get('compose_ms'),'ui':p.get('imshow_and_event_pump_ms'),'ages':p.get('ui_return_ages'),'rss':d.get('process_peak_rss_bytes')}),flush=True)
    if r.returncode: sys.exit(r.returncode)
