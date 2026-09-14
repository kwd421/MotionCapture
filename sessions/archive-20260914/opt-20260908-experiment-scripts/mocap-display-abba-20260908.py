import json, subprocess, sys
from pathlib import Path
root=Path('.')
for label, backend in [('a1','opencv'),('b1','sdl'),('b2','sdl'),('a2','opencv')]:
    output=root/f'sessions/opt-20260908-display-{label}.json'
    argv=[sys.executable,'-m','motioncapture.wholebody_recorded_preview',str(root/'benchmarks/inputs/phone-1080p60-20260906_030954.mp4'),'--provider','coreml-all','--allow-cpu-partitions','--research-only','--max-frames','1200','--opencv-threads','1','--display-backend',backend,'--output',str(output)]
    with output.with_suffix('.log').open('x') as log:
        r=subprocess.run(argv,cwd=root,stdout=log,stderr=subprocess.STDOUT)
    d=json.loads(output.read_text()); p=d.get('preview',{}); a=p.get('ui_return_ages',{})
    print(json.dumps({'arm':label,'backend':backend,'exit':r.returncode,'status':d['status'],'fps':p.get('mean_submission_rate_hz'),'ui':p.get('display_submit_and_event_pump_ms'),'age_p95':a.get('source_age_ms',{}).get('p95_ms'),'age_last':a.get('last_source_age_ms'),'rss':d.get('process_peak_rss_bytes')}),flush=True)
    if r.returncode: sys.exit(r.returncode)
