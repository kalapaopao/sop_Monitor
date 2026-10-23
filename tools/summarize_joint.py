"""Summarize per-lane CSV; time windows and no-pose/expiry are kept separate."""
import argparse
import csv
import json
import re
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
summary={}
def percentile(values,q=.95):
    values=sorted(values)
    return values[int(q*(len(values)-1))] if values else None
for f in sorted(a.directory.glob('*.csv')):
    rows=list(csv.DictReader(f.open()))
    if not rows or 'elapsed_ms' not in rows[0]:continue
    lanes={}
    for lane in sorted(set(r['lane'] for r in rows)):
        subset=[r for r in rows if r['lane']==lane and float(r['elapsed_ms'])>=5000]
        if not subset:continue
        unique={r['result_frame']:r for r in subset if int(r['result_frame'])>=0}
        lanes[lane]={'samples':len(subset),
                     'future_results':sum(int(r['result_frame'])>int(r['frame']) for r in subset),
                     'no_pose_fraction':sum(float(r['pose_age_ms'])<0 for r in subset)/len(subset),
                     'expired_fraction':sum(float(r['pose_age_ms'])>200 for r in subset)/len(subset),
                     'pose_age_p95_ms':percentile([float(r['pose_age_ms']) for r in subset if float(r['pose_age_ms'])>=0]),
                     'detector_ms_p50':percentile([float(r['detector_ms']) for r in unique.values()],.5),
                     'pose_ms_p50':percentile([float(r['pose_ms']) for r in unique.values()],.5),
                     'observed_person_count_mean':sum(int(r['person_count']) for r in unique.values())/max(1,len(unique))}
    logs=f.with_suffix('.log')
    summary[f.stem]={'lanes':lanes,'logs':[dict(re.findall(r'(\w+)=([^ ]+)',line)) for line in logs.read_text().splitlines() if line.startswith('status=')] if logs.exists() else []}
monitor=a.directory/'tegrastats.log'
if monitor.exists():
    s=monitor.read_text()
    summary['system_peak']={
        'RAM_MB':max([int(x) for x in re.findall(r'RAM (\d+)/',s)] or [0]),
        'SWAP_MB':max([int(x) for x in re.findall(r'SWAP (\d+)/',s)] or [0]),
        'GPU_C':max([float(x) for x in re.findall(r'GPU@([\d.]+)C',s)] or [0])}
(a.directory/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
