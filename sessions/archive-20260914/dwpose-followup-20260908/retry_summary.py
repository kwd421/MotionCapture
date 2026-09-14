import json
from pathlib import Path
root=Path(__file__).parent
host=[json.loads(l) for l in (root/'retry-host.jsonl').read_text().splitlines()]
rows=[]
for name in ['retry-A1','retry-B1','retry-B2','retry-A2']:
 p=root/f'{name}.json'
 if not p.exists():continue
 d=json.loads(p.read_text());r=d['run'];u=d['preview'];a=u['ui_return_ages']
 states=[json.loads(l) for l in (root/f'{name}-state.jsonl').read_text().splitlines()]
 samples=[h for h in host if states[0]['time_ns'] <= h['time_ns'] <= states[-1]['time_ns']]
 peaks={}
 for h in samples:
  for process in h['top']:peaks[process['name']]=max(peaks.get(process['name'],0),process['cpu'])
 row=dict(name=name,status=d['status'],frames=r['all_frames']['frames'],consumed=u['submitted_frames'],hz=u['mean_submission_rate_hz'],age=a['source_age_ms'],last_age_ms=a['last_source_age_ms'],stages={k:v for k,v in r['all_frames']['stages'].items() if k in ['detector_inference_ms','pose_inference_ms','verification_ms','decode_read_ms']},ui=u['display_submit_and_event_pump_ms'],cleanup=u['cleanup'],pipeline=r['pipeline'],handoff=r['preview_handoff'],hashes={k:r[k] for k in ['pixels_sha256','detector_predictions_sha256','predictions_sha256']},thermal_states=sorted({s['thermal_state'] for s in states}),window_states=list({str(s['windows']) for s in states}),sampled_process_peak_cpu=peaks,cost=r['host_process_cost'],rss_mib=d['process_peak_rss_bytes']/2**20)
 rows.append(row)
 print(name,round(row['hz'],3),'p95',round(row['age']['p95_ms'],2),'last',round(row['last_age_ms'],2),'Blender',peaks.get('Blender'))
(root/'retry-summary.json').write_text(json.dumps(rows,indent=2))
