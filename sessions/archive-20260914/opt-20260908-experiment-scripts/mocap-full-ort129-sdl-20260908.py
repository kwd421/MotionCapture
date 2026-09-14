import json, subprocess, sys
from pathlib import Path
root=Path('.')
for label, source in [('phone','phone-1080p60-20260906_030954.mp4'),('macbook','macbook-720p30-20260905T175716Z.mp4')]:
    output=root/f'sessions/opt-20260908-full-ort129-sdl-{label}.json'
    argv=[sys.executable,'-m','motioncapture.wholebody_recorded_preview',str(root/'benchmarks/inputs'/source),'--provider','coreml-all','--allow-cpu-partitions','--research-only','--expected-ort-version','1.29.0','--opencv-threads','1','--display-backend','sdl','--output',str(output)]
    with output.with_suffix('.log').open('x') as log:
        r=subprocess.run(argv,cwd=root,stdout=log,stderr=subprocess.STDOUT)
    d=json.loads(output.read_text()); p=d.get('preview',{}); a=p.get('ui_return_ages',{})
    print(json.dumps({'source':label,'status':d['status'],'frames':p.get('submitted_frames'),'fps':p.get('mean_submission_rate_hz'),'age_p95':a.get('source_age_ms',{}).get('p95_ms'),'age_last':a.get('last_source_age_ms'),'rss':d.get('process_peak_rss_bytes')}),flush=True)
    if r.returncode: sys.exit(r.returncode)
