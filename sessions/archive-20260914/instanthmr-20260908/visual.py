"""Selected-frame visual inspection only; excluded from throughput benchmarks."""
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from upstream_skeleton import JOINT_NAMES
from trial import HMRModel, hmr_pose, ROOT

from motioncapture.recording import inspect_recording, RecordedDecoder
from motioncapture.wholebody_catalog import ASSETS, verify_asset
from motioncapture.wholebody_onnx import OrtModel, pose_tensor, decode_pose
from motioncapture.wholebody_onnx import detector_tensor
from motioncapture.wholebody_recorded_preview import compose
from motioncapture.wholebody_stages import detect

# Derive full finger chains from the upstream joint names. Its default demo draws
# only wrist-to-tip fans, so that visualization is not used as anatomy authority.
edges=[('neck','left_shoulder'),('neck','right_shoulder'),
       ('left_shoulder','left_elbow'),('left_elbow','left_wrist'),
       ('right_shoulder','right_elbow'),('right_elbow','right_wrist'),
       ('left_shoulder','left_hip'),('right_shoulder','right_hip'),
       ('left_hip','right_hip'),('left_hip','left_knee'),('left_knee','left_ankle'),
       ('right_hip','right_knee'),('right_knee','right_ankle')]
for side in ('left','right'):
    for toe in ('big_toe_tip','small_toe_tip','heel'):
        edges.append((side+'_ankle',side+'_'+toe))
    for finger in ('thumb','index','middle','ring','pinky'):
        chain=[side+'_wrist']+[f'{side}_{finger}_{j}' for j in
                              ('third_joint','second_joint','first_joint','tip')]
        edges.extend(zip(chain[:-1],chain[1:]))
EDGES=[(JOINT_NAMES.index(a),JOINT_NAMES.index(b)) for a,b in edges]


def overlay(image,people):
    dst=image.copy();h,w=dst.shape[:2]
    for person in people:
        xy=person['xy']
        good=np.isfinite(xy).all(axis=1)&(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
        pts={i:tuple(np.rint(xy[i]).astype(int)) for i in np.flatnonzero(good)}
        for a,b in EDGES:
            color=(230,140,50) if 42<=a<=62 or 42<=b<=62 else (180,80,240) if 21<=a<=41 or 21<=b<=41 else (80,230,100)
            if a in pts and b in pts:cv2.line(dst,pts[a],pts[b],color,2,cv2.LINE_AA)
        for pt in pts.values():cv2.circle(dst,pt,2,(30,230,255),-1)
    return dst


def crop_hand(image,box,wrist):
    h,w=image.shape[:2]
    if not np.isfinite(wrist).all() or not(0<=wrist[0]<w and 0<=wrist[1]<h):
        blank=np.full((240,240,3),25,np.uint8)
        cv2.putText(blank,'wrist out of frame',(10,120),0,.5,(220,220,220),1)
        return blank
    radius=max(30,int(max(box[2:]-box[:2])*.10))
    x,y=np.rint(wrist).astype(int)
    return cv2.resize(image[max(0,y-radius):min(h,y+radius),max(0,x-radius):min(w,x+radius)],(240,240))


def make_sheet(frame,hmr,dw,boxes):
    source=frame.image_bgr
    h,w=source.shape[:2];width=640;height=round(h*width/w)
    left=overlay(source,hmr)
    packet=SimpleNamespace(detected=SimpleNamespace(frame=frame),people=dw)
    # Existing DWPose renderer preserves its own confidence threshold; no confidence
    # is fabricated for InstantHMR, whose outputs do not contain a joint score.
    right_full=compose(packet,[],width=w)[154:]
    canvas=np.full((height+340,1280,3),22,np.uint8)
    canvas[50:50+height,:640]=cv2.resize(left,(640,height))
    canvas[50:50+height,640:]=cv2.resize(right_full,(640,height))
    cv2.putText(canvas,'InstantHMR | 70 estimated points | no joint scores',(12,30),0,.57,(230,230,230),1)
    cv2.putText(canvas,'DWPose | valid points at threshold 0.3',(652,30),0,.57,(230,230,230),1)
    if hmr:
        # Largest per-frame box for inspection only; no actor identity inferred.
        slot=int(np.argmax((boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])))
        for side,wrist_idx,column in [('left',62,0),('right',41,1)]:
            wrist=hmr[slot]['xy'][wrist_idx]
            for img,offset in [(left,0),(right_full,640)]:
                x=offset+column*280
                canvas[height+85:height+325,x:x+240]=crop_hand(img,boxes[slot],wrist)
                cv2.putText(canvas,side+' hand, same image crop',(x+5,height+75),0,.45,(220,220,220),1)
    return canvas


def make_3d(people,boxes):
    canvas=np.full((600,1000,3),22,np.uint8)
    if not people:
        cv2.putText(canvas,'No person detected',(20,60),0,.8,(230,230,230),1)
        return canvas
    slot=int(np.argmax((boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])))
    xyz=people[slot]['outputs'][4].astype(np.float64)
    centered=xyz-(xyz[9]+xyz[10])*.5
    span=max(float(np.ptp(centered,axis=0).max()),1e-6)
    scale=430/span
    for xaxis,offset,label in [(0,0,'Front XY'),(2,500,'Side ZY')]:
        pts=np.rint(centered[:,[xaxis,1]]*scale+np.array([offset+250,310])).astype(int)
        for a,b in EDGES:
            cv2.line(canvas,tuple(pts[a]),tuple(pts[b]),(70,215,120),2,cv2.LINE_AA)
        for i,p in enumerate(pts):
            color=(230,140,50) if 42<=i<=62 else (180,80,240) if 21<=i<=41 else (70,215,120)
            cv2.circle(canvas,tuple(p),3,color,-1)
        cv2.putText(canvas,label,(offset+25,65),0,.7,(235,235,235),1)
    cv2.putText(canvas,'Estimated body-centred 3D | equal scale | no calibration or retargeting',
                (15,30),0,.59,(230,230,230),1)
    return canvas


def main():
    cv2.setNumThreads(1)
    snap=ROOT/'visual';snap.mkdir(exist_ok=False)
    det_path,_=verify_asset(Path('models/wholebody'),'yolox-tiny')
    pose_path,_=verify_asset(Path('models/wholebody'),'dwpose-m')
    hmr=det=dw=None
    rows=[]
    try:
        hmr=HMRModel()
        det=OrtModel(det_path,ASSETS['yolox-tiny'].shape,'coreml-all',allow_cpu=False)
        dw=OrtModel(pose_path,ASSETS['dwpose-m'].shape,'coreml-all',allow_cpu=False)
        sources=[('phone-1080p60-20260906_030954.mp4',(300,1200,3000,5000)),
                 ('macbook-720p30-20260905T175716Z.mp4',(200,1200)),
                 ('pose-rpi-dance.mp4',(60,180))]
        for name,indices in sources:
            src=Path('benchmarks/inputs')/name;probe=inspect_recording(src)
            with RecordedDecoder(src,probe) as decoder:
                for frame in decoder:
                    i=frame.identity.sequence
                    if i not in indices:continue
                    before=hashlib.sha256(frame.image_bgr.tobytes()).hexdigest()
                    detected=detect(det,0,frame)
                    result=hmr_pose(hmr,0,detected)
                    if not rows and result.people:
                        # A bounded isolated REAL-input control, with the other
                        # model idle. No decode/preprocess/display timing claim.
                        rgb=cv2.cvtColor(frame.image_bgr,cv2.COLOR_BGR2RGB)
                        crop,_,_,_,cond=hmr.codec._preprocess(rgb,detected.boxes[0],*rgb.shape[:2])
                        feed={'image':crop[None],'cliff_cond':cond[None]}
                        det_input,_=detector_tensor(frame.image_bgr)
                        isolated={}
                        for label,call in [('hmr',lambda:hmr.session.run(None,feed)),
                                           ('detector',lambda:det.run(det_input))]:
                            values=[]
                            for repeat in range(55):
                                began=time.perf_counter_ns();call();elapsed=(time.perf_counter_ns()-began)/1e6
                                if repeat>=5:values.append(elapsed)
                            isolated[label]={'mean_ms':float(np.mean(values)),
                                             'p95_ms':float(np.quantile(values,.95)),
                                             'max_ms':max(values),'samples':len(values)}
                        (ROOT/'isolated-real-input.json').write_text(json.dumps({
                            'scope':'repeated real phone crop, models run separately; no pipeline speed claim',
                            'source_sha256':probe.sha256,'sequence':i,'results':isolated},indent=2))
                    reference=[]
                    for box in detected.boxes:
                        tensor,center,scale=pose_tensor(frame.image_bgr,box,(192,256))
                        reference.append(decode_pose(dw.run(tensor),(192,256),center,scale,.3))
                    sheet=make_sheet(frame,result.people,reference,detected.boxes)
                    assert hashlib.sha256(frame.image_bgr.tobytes()).hexdigest()==before
                    filename=f'{src.stem}-{i:06d}.png'
                    assert cv2.imwrite(str(snap/filename),sheet)
                    three_d=f'{src.stem}-{i:06d}-3d.png'
                    assert cv2.imwrite(str(snap/three_d),make_3d(result.people,detected.boxes))
                    rows.append({'source_sha256':probe.sha256,'sequence':i,'pts':frame.identity.pts,
                                 'people':len(result.people),'snapshot':filename,'estimated_3d_snapshot':three_d,
                                 'same_boxes':True,'source_pixels_unmodified':True})
                    print(filename,flush=True)
                    if i==max(indices):break
    finally:
        for model in (dw,det,hmr):
            if model is not None:model.close()
    (ROOT/'visual-manifest.json').write_text(json.dumps({'scope':'selected visual comparisons, NOT accuracy ground truth',
        'snapshots_saved_locally':True,'coordinates_saved':False,'rows':rows},indent=2))


if __name__=='__main__':main()
