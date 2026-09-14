import json
from pathlib import Path
root=Path(__file__).parent
results=[]
for path in sorted(root.glob('*ui-*.json')):
    d=json.loads(path.read_text())
    if 'run' not in d or 'preview' not in d: continue
    r,p=d['run'],d['preview']; stages=r['all_frames']['stages']
    states=root/f'{path.stem}-state.jsonl'
    obs=[json.loads(l) for l in states.read_text().splitlines()] if states.exists() else []
    results.append(dict(name=path.stem,status=d['status'],frames=r['all_frames']['frames'],
      submitted=p['submitted_frames'],hz=p['mean_submission_rate_hz'],
      age=p['ui_return_ages']['source_age_ms'],last_age_ms=p['ui_return_ages']['last_source_age_ms'],
      ui=p['display_submit_and_event_pump_ms'],compose=p['compose_ms'],
      detector=stages['detector_inference_ms'],pose=stages['pose_inference_ms'],
      cost=r['host_process_cost'],rss_mib=d['process_peak_rss_bytes']/2**20,
      pixels=r['pixels_sha256'],boxes=r['detector_predictions_sha256'],pose_hash=r['predictions_sha256'],
      cleanup=p['cleanup'],handoff=r['preview_handoff'],
      states=sorted({(o['app_active'],o['app_hidden'],o['thermal_state'],tuple(w['occlusion_state'] for w in o['windows'])) for o in obs})))
(root/'preview-summary.json').write_text(json.dumps(results,indent=2))
for r in results:
 print(r['name'],r['status'],r['frames'],round(r['hz'],3),'p95',round(r['age']['p95_ms'],2),'last',round(r['last_age_ms'],2),'ui',round(r['ui']['mean_ms'],2),'states',r['states'])
