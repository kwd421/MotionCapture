"""Validate two selected real frames across explicit CoreML and CPU modes."""
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
from trial import ROOT,HMRModel
from visual import overlay
from motioncapture.wholebody_onnx import OrtModel,detector_tensor,decode_people,pose_tensor,decode_pose
from motioncapture.wholebody_catalog import ASSETS,verify_asset


def main():
    cv2.setNumThreads(1)
    det=core=cpu=dw=None;rows=[]
    try:
        core=HMRModel();cpu=HMRModel('cpu')
        detpath,_=verify_asset(Path('models/wholebody'),'yolox-tiny')
        dwpath,_=verify_asset(Path('models/wholebody'),'dwpose-m')
        det=OrtModel(detpath,ASSETS['yolox-tiny'].shape,'coreml-all',allow_cpu=False)
        dw=OrtModel(dwpath,ASSETS['dwpose-m'].shape,'coreml-all',allow_cpu=False)
        manifest=json.loads((ROOT/'visual-manifest.json').read_text())['rows']
        for source,benchname in [('macbook-720p30-20260905T175716Z.mp4','macbook-paced'),
                                 ('phone-1080p60-20260906_030954.mp4','phone-paced')]:
            path=Path('benchmarks/inputs')/source
            meta=json.loads((ROOT/f'{benchname}.json').read_text())['source']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['sha256']
            pts=next(r['pts'] for r in manifest if r['source_sha256']==meta['sha256'] and r['sequence']==1200)
            cap=cv2.VideoCapture(str(path),cv2.CAP_FFMPEG)
            try:
                assert cap.isOpened() and cap.getBackendName()=='FFMPEG'
                cap.set(cv2.CAP_PROP_ORIENTATION_AUTO,0)
                for i in range(1201):
                    ok,image=cap.read();assert ok
                expected=float((pts-meta['start_pts'])*Fraction(meta['time_base'])*1000)
                assert abs(cap.get(cv2.CAP_PROP_POS_MSEC)-expected)<.001
                assert image.shape[:2]==(meta['dimensions'][1],meta['dimensions'][0])
            finally:cap.release()
            tensor,ratio=detector_tensor(image)
            boxes=decode_people(det.run(tensor)[0],ratio)
            rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);h,w=rgb.shape[:2]
            row={'source_sha256':meta['sha256'],'sequence':1200,'pts':pts,'people':len(boxes),'comparisons':[]}
            for slot,box in enumerate(boxes):
                crop,x,y,size,cond=core.codec._preprocess(rgb,box,h,w)
                feed={'image':crop[None],'cliff_cond':cond[None]}
                a=core.session.run(None,feed);b=cpu.session.run(None,feed)
                core.validate(a);cpu.validate(b)
                xya=(a[3][0]+1)*.5*size+np.array([x,y],np.float32)
                xyb=(b[3][0]+1)*.5*size+np.array([x,y],np.float32)
                def outside(xy):
                    return ((xy[:,0]<0)|(xy[:,0]>=w)|(xy[:,1]<0)|(xy[:,1]>=h))
                ta,ca,sa=pose_tensor(image,box,(192,256))
                reference=decode_pose(dw.run(ta),(192,256),ca,sa,.3)
                row['comparisons'].append({'slot':slot,'max_2d_disagreement_px':float(np.linalg.norm(xya-xyb,axis=1).max()),
                    'max_3d_joint_disagreement_m':float(np.linalg.norm(a[4][0]-b[4][0],axis=1).max()),
                    'coreml_wrist_out_of_frame':outside(xya)[[62,41]].tolist(),
                    'cpu_wrist_out_of_frame':outside(xyb)[[62,41]].tolist(),
                    'dwpose_body_wrist_in_frame_and_valid':(~outside(reference.xy)&reference.valid)[[9,10]].tolist(),
                    'left_hand_coreml_points_outside':int(outside(xya)[42:63].sum()),
                    'right_hand_coreml_points_outside':int(outside(xya)[21:42].sum())})
                # Full-hand detail centered on the union of both predictions.
                # Same source crop in every column; draw no missing-confidence mask.
                hm=overlay(image,[{'xy':xya}])
                dwi=image.copy()
                for idx in range(91,133):
                    point=reference.xy[idx]
                    if reference.valid[idx] and not outside(reference.xy)[idx]:
                        cv2.circle(dwi,tuple(np.rint(point).astype(int)),3,(255,180,30),-1)
                for side,hs,ds in [('left',slice(42,63),slice(91,112)),('right',slice(21,42),slice(112,133))]:
                    pts2=np.concatenate([xya[hs],reference.xy[ds][reference.valid[ds]]])
                    pts2=pts2[~outside(pts2)]
                    if not len(pts2):continue
                    lo=pts2.min(axis=0);hi=pts2.max(axis=0);center=(lo+hi)/2
                    radius=max(40,float((hi-lo).max())*.65)
                    x1,y1=np.maximum(0,np.floor(center-radius)).astype(int)
                    x2,y2=np.minimum([w,h],np.ceil(center+radius)).astype(int)
                    canvas=np.full((440,1200,3),22,np.uint8)
                    for column,(label,img) in enumerate([('Original',image),('InstantHMR',hm),('DWPose valid hand points',dwi)]):
                        canvas[40:,column*400:(column+1)*400]=cv2.resize(img[y1:y2,x1:x2],(400,400))
                        cv2.putText(canvas,label,(column*400+10,26),0,.55,(230,230,230),1)
                    assert cv2.imwrite(str(ROOT/'visual'/f'{benchname}-1200-{side}-detail.png'),canvas)
            rows.append(row)
    finally:
        for model in (dw,det,cpu,core):
            if model is not None:model.close()
    report={'scope':'two selected frames, same normalized input; numeric disagreement is not ground-truth accuracy',
            'modes':['coreml-strict','cpu'],'raw_coordinates_saved':False,'rows':rows}
    (ROOT/'selected-cpu-coreml-check.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
