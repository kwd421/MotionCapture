import argparse
import collections
import gc
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

p = argparse.ArgumentParser()
p.add_argument('mode', choices=['coreml-strict', 'coreml-hybrid', 'cpu'])
a = p.parse_args()
root = Path('sessions/instanthmr-20260908')
model = Path('models/instanthmr/instanthmr-3504446.onnx')
out = root / f'probe-{a.mode}.json'
assert not out.exists()
assert ort.__version__ == '1.29.0'
so = ort.SessionOptions()
so.intra_op_num_threads = 4
so.inter_op_num_threads = 1
so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
so.add_session_config_entry('session.intra_op.allow_spinning', '0')
so.add_session_config_entry('session.inter_op.allow_spinning', '0')
so.add_free_dimension_override_by_name('batch', 1)
so.enable_profiling = True
so.profile_file_prefix = str(root / f'profile-{a.mode}')
opts = {'ModelFormat': 'MLProgram', 'MLComputeUnits': 'ALL',
        'RequireStaticInputShapes': '1', 'EnableOnSubgraphs': '0',
        'ProfileComputePlan': '1'}
expected = 'CPUExecutionProvider' if a.mode == 'cpu' else 'CoreMLExecutionProvider'
providers = ['CPUExecutionProvider'] if a.mode == 'cpu' else [('CoreMLExecutionProvider', opts)]
if a.mode == 'coreml-strict':
    so.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
if a.mode == 'coreml-hybrid':
    providers.append('CPUExecutionProvider')
r = {'mode': a.mode, 'status': 'running', 'runtime': ort.__version__,
     'platform': platform.platform(), 'model_sha256': hashlib.sha256(model.read_bytes()).hexdigest(),
     'requested_providers': providers, 'scope': 'synthetic preflight, not video or quality'}
s = None
start = time.perf_counter()
try:
    s = ort.InferenceSession(str(model), sess_options=so, providers=providers)
    s.disable_fallback()
    r['registered_providers'] = s.get_providers()
    assert expected in s.get_providers(), 'provider substitution'
    r['load_s'] = time.perf_counter() - start
    feed = {'image': np.zeros((1, 3, 224, 224), np.float32),
            'cliff_cond': np.array([[0., 0., .8]], np.float32)}
    timings = []
    for i in range(25):
        t = time.perf_counter()
        values = s.run(None, feed)
        timings.append((time.perf_counter()-t)*1000)
        assert [v.shape for v in values] == [(1,204),(1,45),(1,3),(1,70,2),(1,70,3)]
        assert all(np.isfinite(v).all() for v in values), 'nonfinite output'
    events = json.loads(Path(s.end_profiling()).read_text())
    nodes = [e for e in events if e.get('cat')=='Node' and e.get('args',{}).get('provider')]
    r['placement_events'] = dict(collections.Counter(e['args']['provider'] for e in nodes))
    r['cpu_ops'] = dict(collections.Counter(e['args'].get('op_name') for e in nodes if e['args']['provider']=='CPUExecutionProvider'))
    r['cold_ms'] = timings[0]
    r['warm_mean_ms'] = float(np.mean(timings[5:]))
    r['warm_p95_ms'] = float(np.quantile(timings[5:], .95))
    r['status'] = 'completed'
except Exception as e:
    r['status'] = 'failed'
    r['error'] = f'{type(e).__name__}: {e}'
finally:
    s = None
    gc.collect()
    r['elapsed_s'] = time.perf_counter()-start
    out.write_text(json.dumps(r,indent=2))
    print(json.dumps(r),flush=True)
