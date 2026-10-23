"""Offline conversion checks on exact C++ audit inputs; not an accuracy test."""
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

p=argparse.ArgumentParser()
p.add_argument('--original',required=True)
p.add_argument('--raw',required=True)
p.add_argument('--audit',type=Path,required=True)
a=p.parse_args()
options=ort.SessionOptions(); options.intra_op_num_threads=2
original=ort.InferenceSession(a.original,options,providers=['CPUExecutionProvider'])
raw_session=ort.InferenceSession(a.raw,options,providers=['CPUExecutionProvider'])
names=['screwdriver1','screwdriver2','hand','person']
def points(row,ratio,left,top):
    cx,cy,w,h=row[:4]
    return cv2.boxPoints(((float((cx-left)/ratio),float((cy-top)/ratio)),
                         (float(w/ratio),float(h/ratio)),float(row[8]*180/np.pi)))
def error(p,q):
    return min(float(np.linalg.norm(p-np.roll(r,k,axis=0),axis=1).max()) for r in [q,q[::-1]] for k in range(4))
reports=[]
for image in sorted(a.audit.glob('lane*.png')):
    frame=cv2.imread(str(image)); h,w=frame.shape[:2]; ratio=min(640/w,640/h)
    width,height=round(w*ratio),round(h*ratio); left,top=(640-width)//2,(640-height)//2
    canvas=np.full((640,640,3),114,np.uint8)
    canvas[top:top+height,left:left+width]=cv2.resize(frame,(width,height))
    tensor=np.ascontiguousarray(canvas[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/255
    raw=raw_session.run(None,{raw_session.get_inputs()[0].name:tensor})[0][0]
    original_rows=original.run(None,{original.get_inputs()[0].name:tensor})[0][0]
    # Verify high-confidence original end-to-end rows are present in raw predictions.
    max_tail_error=0.; tail_count=0
    for r in original_rows:
        if r[4]<.25: continue
        cls=int(r[5]); distances=np.max(np.abs(raw[:,:4]-r[:4]),axis=1)+np.abs(raw[:,8]-r[6])+np.abs(raw[:,4+cls]-r[4])
        max_tail_error=max(max_tail_error,float(distances.min())); tail_count+=1
    anchors=np.argsort(-raw[:,4:8].max(axis=1))[:300]
    candidates=sorted([(i,c) for i in anchors for c in range(4) if raw[i,4+c]>=.25],key=lambda x:-raw[x[0],4+x[1]])[:300]
    expected=[]
    for c in range(4):
        for i,cls in candidates:
            if cls!=c:continue
            poly=points(raw[i],ratio,left,top); area=abs(cv2.contourArea(poly)); accept=True
            for _,pc,prior in expected:
                if pc!=c: continue
                inter,_=cv2.intersectConvexConvex(poly,prior)
                union=area+abs(cv2.contourArea(prior))-inter
                if union>0 and inter/union>=.55:accept=False;break
            if accept:expected.append((i,c,poly))
    actual=list(csv.DictReader(image.with_suffix('.csv').open()))
    used=set(); mismatches=[]; max_corner=0.; max_score=0.
    for det in actual:
        cls=names.index(det['label']); poly=np.array([[float(det[f'x{k}']),float(det[f'y{k}'])] for k in range(4)],np.float32)
        matches=[(error(poly,points),k,i) for k,(i,c,points) in enumerate(expected) if c==cls and k not in used]
        if not matches:mismatches.append({'unmatched':det['label']});continue
        distance,k,i=min(matches); used.add(k)
        score=abs(float(det['confidence'])-float(raw[i,4+cls]))
        max_corner=max(max_corner,distance);max_score=max(max_score,score)
        # Conversion tolerance only; large boxes may magnify FP16 coordinate error.
        if distance>10 or score>.02:mismatches.append({'label':det['label'],'corner_px':distance,'score_delta':score})
    if len(actual)!=len(expected):mismatches.append({'cpp_count':len(actual),'ort_count':len(expected)})
    report={'input':image.name,'original_high_conf_count':tail_count,'raw_tail_max_error':max_tail_error,
            'ort_nms_count':len(expected),'cpp_count':len(actual),'max_corner_px':max_corner,
            'max_confidence_delta':max_score,'mismatches':mismatches,
            'status':'passed' if not mismatches and max_tail_error<.001 else 'failed'}
    reports.append(report)
result={'scope':'Raw extraction and FP16 C++ decoded-result agreement on saved frames; not ground-truth accuracy.',
        'frames':reports,'passed':bool(reports) and all(r['status']=='passed' for r in reports)}
(a.audit/'verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
if not result['passed']:raise SystemExit(1)
