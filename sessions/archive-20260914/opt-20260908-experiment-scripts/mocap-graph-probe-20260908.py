import hashlib,json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper,numpy_helper
from motioncapture.wholebody_catalog import verify_asset,ASSETS
from motioncapture.wholebody_onnx import OrtModel
source,asset=verify_asset(Path('models/wholebody'),'dwpose-m')
out=Path('models/wholebody/experiments/dwpose-m-hardsigmoid/model.onnx')
out.parent.mkdir(parents=True,exist_ok=False)
g=onnx.load(source)
assert next(o.version for o in g.opset_import if o.domain=='') >= 11
used={name for n in g.graph.node for name in (*n.input,*n.output)}
new=[]; transformed=[]
for i,n in enumerate(g.graph.node):
 if n.op_type!='HardSigmoid': new.append(n); continue
 attrs={a.name:helper.get_attribute_value(a) for a in n.attribute}
 prefix=f'mocap_hs_{i}_'
 names=[prefix+s for s in ('alpha','beta','lo','hi','mul','add')]
 assert not used.intersection(names)
 a,b,lo,hi,mul,add=names
 for name,value in zip(names[:4],[attrs.get('alpha',.2),attrs.get('beta',.5),0.,1.],strict=True):
  g.graph.initializer.append(numpy_helper.from_array(np.array(value,np.float32),name))
 new += [helper.make_node('Mul',[n.input[0],a],[mul],name=prefix+'Mul'),helper.make_node('Add',[mul,b],[add],name=prefix+'Add'),helper.make_node('Clip',[add,lo,hi],list(n.output),name=prefix+'Clip')]
 transformed.append({'name':n.name,'attributes':attrs})
del g.graph.node[:];g.graph.node.extend(new)
onnx.checker.check_model(g,full_check=True);onnx.save(g,out)
original=OrtModel(source,ASSETS['dwpose-m'].shape,'cpu',threads=1)
candidate=OrtModel(out,ASSETS['dwpose-m'].shape,'cpu',threads=1)
checks=[]
try:
 rng=np.random.default_rng(908)
 for seed in range(3):
  x=rng.uniform(-3,3,original.shape).astype(np.float32)
  aa,bb=original.run(x),candidate.run(x)
  checks.append({'outputs_finite':all(np.isfinite(v).all() for v in bb),'max_abs':max(float(np.abs(a-b).max()) for a,b in zip(aa,bb,strict=True)),'equal':all(np.array_equal(a,b) for a,b in zip(aa,bb,strict=True))})
finally: original.close();candidate.close()
native=OrtModel(out,ASSETS['dwpose-m'].shape,'coreml-all',allow_cpu=True)
try: meta=native.metadata
finally: native.close()
d={'original_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'candidate_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'candidate':str(out),'transformations':transformed,'cpu_synthetic_checks':checks,'native':meta,'accuracy_verified':False,'video_verified':False}
Path('sessions/opt-20260908-hardsigmoid-probe.json').write_text(json.dumps(d,indent=2)); print(json.dumps(d),flush=True)
