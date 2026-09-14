import json,pickle,subprocess
from pathlib import Path
import numpy as np
from motioncapture.wholebody_onnx import PARTS
runs=[]
for version,req in [('1.22.1','tools/requirements-wholebody.txt'),('1.29.0','tools/requirements-wholebody-ort129.txt')]:
 with Path(f'sessions/opt-20260908-quality-{version}.log').open('x') as log:
  p=subprocess.run(['uv','run','--with-requirements',req,'python','/tmp/mocap-runtime-quality-worker-20260908.py',version],stdout=subprocess.PIPE,stderr=log,check=True)
 # Trusted child code above; coordinate arrays stay in RAM and are never saved.
 runs.append(pickle.loads(p.stdout))
 print('completed quality worker',version,flush=True)
values={k:[] for k in PARTS};only_a=dict.fromkeys(PARTS,0);only_b=dict.fromkeys(PARTS,0)
score_max=0.;unmatched=0;matched=0;multi=0;worst=[]
for a,b in zip(runs[0]['packets'],runs[1]['packets'],strict=True):
 assert a[:3]==b[:3]
 if not np.array_equal(a[3],b[3]) or len(a[4])!=len(b[4]): unmatched+=1;continue
 matched+=1;multi+=len(a[4])>1
 for slot,(pa,pb) in enumerate(zip(a[4],b[4],strict=True)):
  xy_a,sc_a,v_a=pa;xy_b,sc_b,v_b=pb;common=v_a&v_b
  distances=np.linalg.norm(xy_a.astype(np.float64)-xy_b.astype(np.float64),axis=1)
  score_max=max(score_max,float(np.abs(sc_a-sc_b).max()))
  if common.any():worst.append({'sequence':a[0],'slot':slot,'max_pixels':float(distances[common].max())})
  for name,(start,end) in PARTS.items():
   mask=common[start:end];values[name].extend(distances[start:end][mask].tolist());only_a[name]+=int((v_a[start:end]&~v_b[start:end]).sum());only_b[name]+=int((v_b[start:end]&~v_a[start:end]).sum())
parts={}
for name,vals in values.items():
 arr=np.asarray(vals)
 parts[name]={'observations':len(arr),'mean_px':float(arr.mean()) if len(arr) else None,'p95_px':float(np.quantile(arr,.95)) if len(arr) else None,'p99_px':float(np.quantile(arr,.99)) if len(arr) else None,'max_px':float(arr.max()) if len(arr) else None,'above_1px':int((arr>1).sum()),'above_10px':int((arr>10).sum()),'only_1221':only_a[name],'only_1290':only_b[name]}
report={'scope':'original model; first 1200 phone frames; 1.22.1 vs 1.29.0; quality disagreement, NOT accuracy','correspondence':'same source PTS and byte-equal ordered detector boxes, within-frame slots only','frames_matched':matched,'multi_person_frames_matched':multi,'unmatched_box_frames':unmatched,'parts':parts,'max_abs_score_delta':score_max,'worst':sorted(worst,key=lambda w:w['max_pixels'],reverse=True)[:8],'privacy':{'coordinates_written':False,'frames_written':False,'transport':'owned subprocess pipe; transient parent RAM'},'accuracy_verified':False}
Path('sessions/opt-20260908-runtime-quality.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
