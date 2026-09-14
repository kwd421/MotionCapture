import collections,hashlib,json,tempfile,time
from pathlib import Path
import numpy as np
import onnxruntime as ort
assert ort.__version__=='1.29.0'
path=Path('models/wholebody/dwpose-m/model.onnx');rows=[]
for batch in (1,2):
 so=ort.SessionOptions();so.intra_op_num_threads=4;so.inter_op_num_threads=1;so.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
 so.add_session_config_entry('session.intra_op.allow_spinning','0');so.add_session_config_entry('session.inter_op.allow_spinning','0');so.add_free_dimension_override_by_name('batch',batch)
 options={'ModelFormat':'MLProgram','MLComputeUnits':'ALL','RequireStaticInputShapes':'1','EnableOnSubgraphs':'0','ProfileComputePlan':'1'}
 with tempfile.TemporaryDirectory(prefix='mocap-ort129-profile-') as tmp:
  so.enable_profiling=True;so.profile_file_prefix=str(Path(tmp)/'placement')
  session=ort.InferenceSession(str(path),sess_options=so,providers=[('CoreMLExecutionProvider',options),'CPUExecutionProvider']);session.disable_fallback()
  assert 'CoreMLExecutionProvider' in session.get_providers()
  start=time.perf_counter_ns();outputs=session.run(None,{session.get_inputs()[0].name:np.zeros((batch,3,256,192),np.float32)})
  ms=(time.perf_counter_ns()-start)/1e6
  assert sorted(x.shape for x in outputs)==sorted([(batch,133,384),(batch,133,512)]) and all(np.isfinite(x).all() for x in outputs)
  events=json.loads(Path(session.end_profiling()).read_text())
  nodes=[e for e in events if e.get('cat')=='Node' and e.get('args',{}).get('provider')]
  row={'batch':batch,'requested_options':options,'provider_events':dict(collections.Counter(e['args']['provider'] for e in nodes)),'cpu_ops':[e['args'].get('op_name') for e in nodes if e['args']['provider']=='CPUExecutionProvider'],'zero_inference_ms':ms,'scope':'cold synthetic preflight; not video performance','outputs_finite':True}
  assert row['provider_events'].get('CoreMLExecutionProvider',0)>0;rows.append(row);del session
report={'ort_version':ort.__version__,'original_model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':rows,'accuracy_verified':False,'live_60fps_verified':False}
Path('sessions/opt-20260908-ort129-probe.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
