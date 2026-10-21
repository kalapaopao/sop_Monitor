"""Sequential experiments: no overlapping inference runs, no environment changes."""
from pathlib import Path
import json
import os
import subprocess
import time

root=Path(__file__).resolve().parent
os.chdir(root)
out=root/'output/profile_20261020'
out.mkdir(parents=True,exist_ok=True)
manifest=[]
def run(name,args,environment=None,timeout=150,cwd=None):
    start=time.time()
    with (out/(name+'.log')).open('w') as log:
        result=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,env=environment,timeout=timeout,cwd=cwd)
    manifest.append({'name':name,'exit':result.returncode,'wall_seconds':time.time()-start,'args':args})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(name,'exit',result.returncode,'seconds',round(time.time()-start,2),flush=True)
    if result.returncode:
        print((out/(name+'.log')).read_text()[-5000:],flush=True)
        raise RuntimeError(name+' failed')

video=['--source','../test_vedio/03.mp4','--second-source','../test_vedio/03.mp4',
       '--third-source','../test_vedio/07.mp4','--fourth-source','../test_vedio/07.mp4']
telemetry_log=(out/'telemetry.log').open('w')
telemetry=subprocess.Popen(['tegrastats','--interval','1000'],stdout=telemetry_log,stderr=subprocess.STDOUT)
try:
    smoke_env=os.environ.copy(); smoke_env['SOP_PROFILE_CSV']=str(out/'smoke_phases.csv')
    run('smoke',['./build/sop_nx_profile_probe']+video+['--headless','--duration','8'],smoke_env)
    for name,binary,gui,profile in [
        ('control_headless','sop_nx_stage2_probe',False,False),
        ('profile_headless','sop_nx_profile_probe',False,True),
        ('control_gui','sop_nx_stage2_probe',True,False),
        ('profile_gui','sop_nx_profile_probe',True,True)]:
        audit=out/(name+'_audit'); audit.mkdir(exist_ok=True)
        args=['./build/'+binary]+video+['--duration','40','--metrics',str(out/(name+'_metrics.csv')),
                                     '--audit-dir',str(audit)]
        if not gui: args+=['--headless']
        env=os.environ.copy()
        if profile: env['SOP_PROFILE_CSV']=str(out/(name+'_phases.csv'))
        run(name,args,env)
    for model,plan in [('obb','models/joint_20261018/joint_fp16.plan'),('rtmw','build/rtmw.plan')]:
        for graph in (False,True):
            name=model+('_graph' if graph else '_plain')
            args=['/usr/src/tensorrt/bin/trtexec','--loadEngine='+plan,
                  '--plugins=build/libsop_silu_plugin.so','--duration=6','--warmUp=1500',
                  '--iterations=100','--noDataTransfers','--exportTimes='+str(out/(name+'_times.json'))]
            if graph: args+=['--useCudaGraph']
            else: args+=['--separateProfileRun','--dumpProfile','--exportProfile='+str(out/(name+'_layers.json')),
                         '--exportLayerInfo='+str(out/(name+'_layerinfo.json'))]
            run(name,args,timeout=120)
    run('nsys',['nsys','profile','--sample=none','--cpuctxsw=none','--trace=cuda,nvtx,osrt',
                '--delay=8','--duration=10','--kill=none','--output='+str(out/'four_headless'),
                './build/sop_nx_stage2_probe']+video+['--headless','--duration','25'],timeout=120)
    run('nsys_stats',['nsys','stats','--report','cudaapisum,gpukernsum,gpumemtimesum,gpumemsizesum',
                     '--format','csv','--output',str(out/'nsys_stats'),str(out/'four_headless.nsys-rep')],timeout=120)
    run('ctest',['ctest','--output-on-failure'],timeout=150,cwd=str(root/'build'))
finally:
    telemetry.terminate()
    telemetry.wait(timeout=10)
    telemetry_log.close()
print('ALL_EXPERIMENTS_DONE',flush=True)
