"""Report throughput separately from fixed-frame output changes; no ground-truth accuracy claims."""
from pathlib import Path
import collections
import csv
import json
import math
import re
import statistics
import sys

out=Path(sys.argv[1])
def distribution(values):
    values=sorted(values)
    if not values:return {'n':0}
    return {'n':len(values),'mean':statistics.mean(values),'p50':statistics.median(values),
            'p95':values[min(len(values)-1,int(.95*(len(values)-1)))],'max':values[-1]}
result={'runs':{},'engines':{},'quality':{}}
for path in sorted(out.glob('headless_*.log'))+sorted(out.glob('gui_*.log')):
    text=path.read_text(errors='replace');lanes=[];actual=[]
    for line in text.splitlines():
        if line.startswith('status=complete'):lanes.append(dict(re.findall(r'(\w+)=([^\s]+)',line)))
        if line.startswith('actual_batch='):actual.append(dict(re.findall(r'(\w+)=([^\s]+)',line)))
    if not lanes:continue
    item={'lanes':lanes,'actual_batch_calls':actual,
          'total_steady_hz':sum(float(x['steady_job_hz']) for x in lanes),
          'mean_lane_hz':statistics.mean(float(x['steady_job_hz']) for x in lanes),
          'mosaic_fps':int(lanes[0]['mosaic_presents'])/float(lanes[0]['duration_s'])}
    groups=out/(path.stem+'_groups.csv')
    if groups.exists():
        rows=[row for row in csv.DictReader(groups.open()) if float(row['elapsed_ms'])-float(row['total_ms'])>=5000]
        item['group_stages']={key:distribution([float(row[key]) for row in rows])
                             for key in ['group_size','source_ms','detector_ms','pose_ms','total_ms']}
        item['normalized_task_ms']=distribution([float(row['total_ms'])/int(row['group_size']) for row in rows])
    result['runs'][path.stem]=item
for path in sorted(out.glob('engine_*.log')):
    text=path.read_text(errors='replace');batch=int(re.search(r'_b(\d+)$',path.stem)[1])
    q=re.search(r'Throughput: ([\d.]+)',text);gpu=re.search(r'GPU Compute Time:.*?mean = ([\d.]+)',text)
    if q and gpu:
        result['engines'][path.stem]={'batch':batch,'batches_per_second':float(q[1]),
             'images_per_second':batch*float(q[1]),'gpu_ms_per_batch':float(gpu[1]),'gpu_ms_per_image':float(gpu[1])/batch}

def read_quality(name):
    directory=out/('quality_'+name)
    if not directory.exists():return None
    ds=collections.defaultdict(list);ps=collections.defaultdict(dict)
    for row in csv.DictReader((directory/'detections.csv').open()):
        ds[row['sample']].append({'label':row['label'],'score':float(row['confidence']),
            'points':[(float(row['x'+str(i)]),float(row['y'+str(i)])) for i in range(4)]})
    for row in csv.DictReader((directory/'poses.csv').open()):
        ps[row['sample']][int(row['keypoint'])]=(float(row['x']),float(row['y']),float(row['confidence']))
    return {'ds':ds,'ps':ps,'meta':{r['sample']:r for r in csv.DictReader((directory/'samples.csv').open())}}

def corner_error(a,b):
    return min(max(math.hypot(a[i][0]-v[(i+shift)%4][0],a[i][1]-v[(i+shift)%4][1]) for i in range(4))
               for v in [b,list(reversed(b))] for shift in range(4))

for directory in sorted(out.glob('quality_*')):
    if not directory.is_dir() or not (directory/'samples.csv').exists():continue
    name=directory.name[len('quality_'):]
    measured=read_quality(name)
    for reference_name in ['legacy','square11']:
        reference=read_quality(reference_name)
        if reference is None or name==reference_name:continue
        geometries=[];scores=[];count_changes=[];pose_missing=[];label_geometry=collections.defaultdict(list)
        keypoints=collections.defaultdict(list);worst=[];person_changes=[];exact=0;visibility_changes=0
        for sample,meta in reference['meta'].items():
            current=measured['meta'].get(sample)
            if current is None:raise RuntimeError('Missing validation sample '+sample)
            if current['frame_hash']==meta['frame_hash']:exact+=1
            if current['person_count']!=meta['person_count']:person_changes.append(sample)
            a=reference['ds'][sample];b=measured['ds'][sample]
            ca=collections.Counter(d['label'] for d in a);cb=collections.Counter(d['label'] for d in b)
            if ca!=cb:count_changes.append({'sample':sample,'reference':dict(ca),'measured':dict(cb)})
            for label in sorted(set(ca)|set(cb)):
                aa=[d for d in a if d['label']==label];bb=[d for d in b if d['label']==label];used=set()
                for detection in aa:
                    candidates=[(corner_error(detection['points'],d['points']),j,d) for j,d in enumerate(bb) if j not in used]
                    if not candidates:continue
                    error,j,other=min(candidates,key=lambda x:x[0]);used.add(j)
                    geometries.append(error);scores.append(abs(detection['score']-other['score']));label_geometry[label].append(error)
            pa=reference['ps'].get(sample,{});pb=measured['ps'].get(sample,{})
            if bool(pa)!=bool(pb):pose_missing.append(sample)
            sample_errors=[]
            for k in pa.keys()&pb.keys():
                x,y,confidence=pa[k];xx,yy,other_confidence=pb[k]
                visible_a=confidence>=.35 and 0<=x<1920 and 0<=y<1080
                visible_b=other_confidence>=.35 and 0<=xx<1920 and 0<=yy<1080
                if visible_a!=visible_b:visibility_changes+=1
                if not (visible_a and visible_b):continue
                error=math.hypot(x-xx,y-yy);keypoints['all'].append(error);sample_errors.append(error)
                if 5<=k<=16:keypoints['body'].append(error)
                if 91<=k<=132:keypoints['hands'].append(error)
            if sample_errors:worst.append({'sample':sample,'keypoint_max_px':max(sample_errors)})
        result['quality'][name+'_vs_'+reference_name]={'samples':len(reference['meta']),'exact_result_hashes':exact,
            'box_corner_error_px':distribution(geometries),'box_score_difference':distribution(scores),
            'box_corner_error_by_label_px':{k:distribution(v) for k,v in label_geometry.items()},
            'detection_count_changes':count_changes,'person_count_changes':person_changes,'pose_presence_changes':pose_missing,
            'keypoint_visibility_changes':visibility_changes,
            'keypoint_visibility_rule':'score>=0.35 and within original 1920x1080 image in both outputs',
            'visible_keypoint_difference_px':{k:distribution(v) for k,v in keypoints.items()},
            'worst_samples':sorted(worst,key=lambda x:x['keypoint_max_px'],reverse=True)[:5]}
(out/'batch_analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
for name,item in result['runs'].items():
    print(name,'mean_lane_hz',round(item['mean_lane_hz'],3),'total',round(item['total_steady_hz'],3),
          'display',round(item['mosaic_fps'],2),'pose_age_p95',[x['pose_age_p95_ms'] for x in item['lanes']])
print('ENGINE',json.dumps(result['engines'],indent=2))
for name,item in result['quality'].items():
    print('QUALITY',name,'count_changes',len(item['detection_count_changes']),'person_changes',item['person_count_changes'],
          'box_max',item['box_corner_error_px'].get('max'),'keypoints',item['visible_keypoint_difference_px'])
