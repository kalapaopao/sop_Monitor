"""Summarize measured phases without adding overlapping CPU/GPU timings."""
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

out=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[2]/'output/profile_20261020'
def summary(values):
    values=sorted(values)
    return {'n':len(values),'mean':statistics.mean(values),'p50':statistics.median(values),
            'p95':values[min(len(values)-1,int((len(values)-1)*.95))]}
result={'runs':{},'phases':{},'engines':{},'audit_comparisons':[],'result_hash_comparisons':[]}
for name in ['baseline_headless','control_headless','profile_headless','control_gui','profile_gui','nsys']:
    path=out/(name+'.log')
    if not path.exists(): continue
    lanes=[]
    for line in path.read_text(errors='replace').splitlines():
        if line.startswith('status=complete'):
            lanes.append(dict(re.findall(r'(\w+)=([^\s]+)',line)))
    result['runs'][name]=lanes
for name in ['profile_headless','profile_gui']:
    groups={}
    jobs={}
    with (out/(name+'_phases.csv')).open() as f:
        for row in csv.DictReader(f):
            if not 5000<=float(row['start_ms'])<40000: continue
            value=float(row['duration_ms'])
            key=':'.join([row['domain'],row['model'],row['stage']])
            groups.setdefault(key,[]).append(value)
            if row['domain']=='gpu':
                job=(row['lane'],row['frame'])
                jobs[job]=jobs.get(job,0)+value
    result['phases'][name]={key:summary(values) for key,values in groups.items()}
    result['phases'][name]['gpu:job:sum_event_spans']=summary(list(jobs.values()))
for model in ['obb','rtmw']:
    for variant in ['plain','graph']:
        name=model+'_'+variant
        log=(out/(name+'.log')).read_text(errors='replace')
        lines=[line for line in log.splitlines() if any(t in line for t in
              ['Throughput:','Enqueue Time:','GPU Compute Time:','Host Latency:', 'CUDA Graph', 'graph capture', 'Total GPU Compute Time:'])]
        result['engines'][name]=lines
for suffix in ['headless','gui']:
    def fingerprints(name):
        found={}
        with (out/(name+'_metrics.csv')).open() as stream:
            for row in csv.DictReader(stream):
                if int(row['result_frame'])>=0:
                    found[(row['lane'],row['result_frame'])]=row['result_hash']
        return found
    reference=fingerprints('control_'+suffix); measured=fingerprints('profile_'+suffix)
    common=reference.keys()&measured.keys()
    mismatches=[key for key in common if reference[key]!=measured[key]]
    result['result_hash_comparisons'].append({'mode':suffix,'common_frames':len(common),
        'mismatches':len(mismatches),'examples':mismatches[:5]})
    for lane in range(4):
        a=out/('control_'+suffix+'_audit')/('lane'+str(lane)+'.png')
        b=out/('profile_'+suffix+'_audit')/('lane'+str(lane)+'.png')
        if not a.exists() or not b.exists(): continue
        image_equal=hashlib.sha256(a.read_bytes()).digest()==hashlib.sha256(b.read_bytes()).digest()
        csv_equal=a.with_suffix('.csv').read_bytes()==b.with_suffix('.csv').read_bytes()
        result['audit_comparisons'].append({'mode':suffix,'lane':lane,'same_input_png':image_equal,'same_detection_csv':csv_equal})
(out/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
for name,lanes in result['runs'].items():
    print(name,'Hz/lane',[x.get('steady_job_hz') for x in lanes])
for name,phases in result['phases'].items():
    print('\n'+name)
    for key,value in phases.items(): print(f"{key:47} n={value['n']:5} mean={value['mean']:8.4f} p50={value['p50']:8.4f} p95={value['p95']:8.4f}")
print('\nENGINE RESULTS',json.dumps(result['engines'],indent=2))
print('\nAUDIT',json.dumps(result['audit_comparisons']))
print('RESULT HASH',json.dumps(result['result_hash_comparisons']))
