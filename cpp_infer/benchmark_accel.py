"""Offline experiment driver. All actual decode/inference/display remains C++."""
import argparse
import csv
import json
import re
import subprocess
import time
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--phase',choices=['headless','gui','soak'],required=True)
p.add_argument('--duration',type=int,default=30)
p.add_argument('--workers',type=int,default=1)
p.add_argument('--output',type=Path,default=Path('output/accel_20261019/benchmark'))
a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
configs=[('baseline',None,'opencv','cpu')]
if a.phase=='headless': configs += [('hw_cpu1',1,'gst-h265','cpu')]
if a.phase!='soak': configs += [(f'hw_cuda{w}',w,'gst-h265','cuda') for w in (1,2,4)]
else: configs=[(f'soak_cuda{a.workers}',a.workers,'gst-h265','cuda')]
summary=[]
for label,workers,decoder,preprocess in configs:
    name=f'{a.phase}_{label}'
    exe='./build/sop_nx_joint_probe' if workers is None else './build/sop_nx_accel_probe'
    args=[exe,'--source','../test_vedio/03.mp4','--second-source','../test_vedio/03.mp4',
          '--third-source','../test_vedio/07.mp4','--fourth-source','../test_vedio/07.mp4',
          '--duration',str(a.duration),'--metrics',str(a.output/(name+'.csv'))]
    if workers is not None: args+=['--workers',str(workers),'--decoder',decoder,'--preprocess',preprocess]
    if a.phase=='headless': args+=['--headless']
    print('START',name,flush=True)
    with (a.output/(name+'.system.log')).open('w') as monitor_log, (a.output/(name+'.log')).open('w') as log:
        monitor=subprocess.Popen(['tegrastats','--interval','1000'],stdout=monitor_log,stderr=subprocess.STDOUT)
        started=time.monotonic()
        try:
            result=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,timeout=a.duration+90)
            code=result.returncode
        except subprocess.TimeoutExpired:
            code=124
        finally:
            monitor.terminate()
            try: monitor.wait(timeout=5)
            except subprocess.TimeoutExpired: monitor.kill(); monitor.wait()
    text=(a.output/(name+'.log')).read_text()
    lanes=[dict(re.findall(r'(\w+)=([^ ]+)',line)) for line in text.splitlines() if line.startswith('status=complete')]
    system=(a.output/(name+'.system.log')).read_text()
    rows=list(csv.DictReader((a.output/(name+'.csv')).open())) if (a.output/(name+'.csv')).exists() else []
    steady=[r for r in rows if float(r['elapsed_ms'])>=5000]
    item={'name':name,'exit_code':code,'wall_seconds':time.monotonic()-started,'command':args,'lanes':lanes,
          'future_results':sum(int(r['result_frame'])>int(r['frame']) for r in steady),
          'system_peak_ram_mb':max([int(x) for x in re.findall(r'RAM (\d+)/',system)] or [0]),
          'system_peak_swap_mb':max([int(x) for x in re.findall(r'SWAP (\d+)/',system)] or [0]),
          'system_peak_gpu_c':max([float(x) for x in re.findall(r'GPU@([\d.]+)C',system)] or [0])}
    summary.append(item)
    (a.output/(a.phase+'_summary.json')).write_text(json.dumps(summary,indent=2))
    print('DONE',name,'exit',code,'Hz',[l['steady_job_hz'] for l in lanes],
          'ageP95',[l['pose_age_p95_ms'] for l in lanes],'RAM',item['system_peak_ram_mb'],flush=True)
    if code or len(lanes)!=4 or item['future_results']: raise SystemExit('Invalid benchmark; inspect logs')
