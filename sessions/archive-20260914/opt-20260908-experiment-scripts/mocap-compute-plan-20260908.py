import json
from pathlib import Path
from motioncapture import wholebody_onnx as adapter
from motioncapture.wholebody_catalog import ASSETS, verify_asset
original_profile=adapter.profile_placement
profile_events=[]
def profile(events,*args,**kwargs):
    profile_events.append(events)
    return original_profile(events,*args,**kwargs)
adapter.profile_placement=profile
original=adapter.provider_plan
def plan(*args, **kwargs):
    providers,expected=original(*args,**kwargs)
    providers[0][1]['ProfileComputePlan']='1'
    return providers,expected
adapter.provider_plan=plan
for key in ['yolox-tiny','dwpose-m']:
    path, asset=verify_asset(Path('models/wholebody'),key)
    model=adapter.OrtModel(path,ASSETS[key].shape,'coreml-all',allow_cpu=True)
    try: print(json.dumps({'model':key,'requested_compute_plan':True,'metadata':model.metadata}),flush=True)
    finally: model.close()

Path('sessions/opt-20260908-compute-plan-events.json').write_text(json.dumps(profile_events,indent=2))
