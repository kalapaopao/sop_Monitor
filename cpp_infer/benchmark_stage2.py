"""Sequential C++ four-stream ablation runs; no Python model inference."""
import argparse,csv,json,re,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--phase',choices=['headless','gui','soak'],required=True)
p.add_argument('--duration',type=int,default=30)
p.add_argument('--reuse-io',choices=['on','off'],default='on')
p.add_argument('--output',type=Path,default=Path('output/stage2_20261019/benchmark'))
a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
configs=[('baseline',None,None)]
if a.phase=='headless': configs += [('control','cpu','off'),('reuse_only','cpu','on')]
configs += [('warp_only','cuda','off'),('warp_reuse','cuda','on')]
if a.phase=='soak': configs=[('soak','cuda',a.reuse_io)]
summary=[]
for label,pose,reuse in configs:
 name=a.phase+'_'+label
 args=['./build/sop_nx_accel_probe' if pose is None else './build/sop_nx_stage2_probe',
       '--source','../test_vedio/03.mp4','--second-source','../test_vedio/03.mp4',
       '--third-source','../test_vedio/07.mp4','--fourth-source','../test_vedio/07.mp4',
       '--duration',str(a.duration),'--workers','1','--decoder','gst-h265','--preprocess','cuda',
       '--metrics',str(a.output/(name+'.csv'))]
 if pose is not None: args+=['--pose-preprocess',pose,'--reuse-io',reuse,'--source-layout','bgr']
 if a.phase=='headless': args+=['--headless']
 print('START',name,flush=True)
 with (a.output/(name+'.log')).open('w') as log,(a.output/(name+'.system.log')).open('w') as system_log:
  monitor=subprocess.Popen(['tegrastats','--interval','1000'],stdout=system_log,stderr=subprocess.STDOUT)
  start=time.monotonic()
  try:
   result=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,timeout=a.duration+90); code=result.returncode
  except subprocess.TimeoutExpired: code=124
  finally:
   monitor.terminate()
   try: monitor.wait(timeout=5)
   except subprocess.TimeoutExpired: monitor.kill();monitor.wait()
 text=(a.output/(name+'.log')).read_text()
 lanes=[dict(re.findall(r'(\w+)=([^ ]+)',line)) for line in text.splitlines() if line.startswith('status=complete')]
 io=[dict(re.findall(r'(\w+)=([^ ]+)',line)) for line in text.splitlines() if line.startswith('stage2_io=')]
 rows=list(csv.DictReader((a.output/(name+'.csv')).open())) if (a.output/(name+'.csv')).exists() else []
 steady=[r for r in rows if float(r['elapsed_ms'])>=5000]
 system=(a.output/(name+'.system.log')).read_text()
 item={'name':name,'exit_code':code,'command':args,'wall_seconds':time.monotonic()-start,'lanes':lanes,'io':io,
       'future_results':sum(int(r['result_frame'])>int(r['frame']) for r in steady),
       'system_peak_ram_mb':max([int(x) for x in re.findall(r'RAM (\d+)/',system)] or [0]),
       'system_peak_swap_mb':max([int(x) for x in re.findall(r'SWAP (\d+)/',system)] or [0]),
       'system_peak_gpu_c':max([float(x) for x in re.findall(r'GPU@([\d.]+)C',system)] or [0])}
 summary.append(item); (a.output/(a.phase+'_summary.json')).write_text(json.dumps(summary,indent=2))
 print('DONE',name,'exit',code,'Hz',[x['steady_job_hz'] for x in lanes],
       'P95',[x['pose_age_p95_ms'] for x in lanes],'RAM',item['system_peak_ram_mb'],flush=True)
 if code or len(lanes)!=4 or item['future_results']: raise SystemExit('Invalid run; inspect logs')
