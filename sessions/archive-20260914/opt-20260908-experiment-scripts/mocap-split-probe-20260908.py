import hashlib,json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper,numpy_helper
from motioncapture.wholebody_catalog import ASSETS
from motioncapture.wholebody_onnx import OrtModel
source=Path('models/wholebody/experiments/dwpose-m-hardsigmoid/model.onnx')
out=Path('models/wholebody/experiments/dwpose-m-hardsigmoid-slice/model.onnx')
out.parent.mkdir(parents=True,exist_ok=False)
g=onnx.load(source);new=[];transformed=[]
used={s for n in g.graph.node for s in (*n.input,*n.output)}
for i,n in enumerate(g.graph.node):
 if n.op_type!='Split':new.append(n);continue
 attrs={a.name:helper.get_attribute_value(a) for a in n.attribute}; lengths=attrs['split'];axis=attrs['axis']
 assert len(n.input)==1 and len(lengths)==len(n.output) and all(v>0 for v in lengths)
 start=0
 for j,(length,output) in enumerate(zip(lengths,n.output,strict=True)):
  prefix=f'mocap_split_{i}_{j}_';names=[prefix+s for s in ('start','end','axes')]
  assert not used.intersection(names)
  for name,val in zip(names,([start],[start+length],[axis]),strict=True):
   g.graph.initializer.append(numpy_helper.from_array(np.array(val,np.int64),name))
  new.append(helper.make_node('Slice',[n.input[0],*names],[output],name=prefix+'Slice'));start+=length
 transformed.append({'name':n.name,'attributes':attrs})
del g.graph.node[:];g.graph.node.extend(new)
onnx.checker.check_model(g,full_check=True);onnx.save(g,out)
a=OrtModel(source,ASSETS['dwpose-m'].shape,'cpu',threads=1);b=OrtModel(out,a.shape,'cpu',threads=1)
try:
 x=np.random.default_rng(908).uniform(-3,3,a.shape).astype(np.float32)
 aa,bb=a.run(x),b.run(x); equal=all(np.array_equal(u,v) for u,v in zip(aa,bb,strict=True));assert equal
finally:a.close();b.close()
native=OrtModel(out,ASSETS['dwpose-m'].shape,'coreml-all',allow_cpu=False)
try:meta=native.metadata
finally:native.close()
d={'candidate':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'transformations':transformed,'cpu_outputs_equal_to_hardsigmoid_candidate':equal,'native':meta,'accuracy_verified':False,'video_verified':False}
Path('sessions/opt-20260908-slice-probe.json').write_text(json.dumps(d,indent=2));print(json.dumps(d),flush=True)
