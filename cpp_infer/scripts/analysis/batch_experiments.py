"""Build and test batch experiments sequentially on NX, preserving current engines."""
from pathlib import Path
import json
import os
import signal
import subprocess
import time

root=Path(__file__).resolve().parents[2];os.chdir(root)
out=root/'output/batch_20261020';models=root/'models/batch_20261020'
out.mkdir(parents=True,exist_ok=True);models.mkdir(parents=True,exist_ok=True)
manifest=[]
def run(name,args,timeout=180,cwd=None,check=True):
    start=time.time()
    with (out/(name+'.log')).open('w') as log:
        value=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,timeout=timeout,cwd=cwd)
    manifest.append({'name':name,'exit':value.returncode,'seconds':time.time()-start,'args':args,'cwd':str(cwd or root)})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(name,'exit',value.returncode,'seconds',round(time.time()-start,2),flush=True)
    if value.returncode and check:
        print((out/(name+'.log')).read_text(errors='replace')[-6000:],flush=True)
        raise RuntimeError(name+' failed')
    return value.returncode

def prepare_samples():
    import cv2
    directory=out/'samples';directory.mkdir(exist_ok=True)
    index=0;items=[]
    for frame_id in [0,30,60,300,600,900]:
        for video in ['03','07']:
            capture=cv2.VideoCapture('../test_vedio/'+video+'.mp4')
            if frame_id and not capture.set(cv2.CAP_PROP_POS_FRAMES,frame_id):raise RuntimeError('Cannot seek samples')
            ok,image=capture.read();capture.release()
            if not ok:raise RuntimeError('Cannot decode sample')
            name=f'sample{index:02}_{video}_f{frame_id:04}.png'
            if not cv2.imwrite(str(directory/name),image):raise RuntimeError('Cannot save sample')
            items.append({'sample':name,'video':video,'frame':frame_id});index+=1
    (out/'samples.json').write_text(json.dumps(items,indent=2))

def main():
    run('configure',['cmake','-S','.','-B','build','-DSOP_BUILD_ACCEL=ON'])
    run('build',['cmake','--build','build','--target','sop_nx_batch_probe','batch_preprocess_test','-j2'],timeout=600)
    run('ctest',['ctest','--output-on-failure'],cwd=root/'build')
    prepare_samples()
    base=['./build/sop_nx_batch_probe','--duration','30','--source','../test_vedio/03.mp4',
          '--second-source','../test_vedio/03.mp4','--third-source','../test_vedio/07.mp4',
          '--fourth-source','../test_vedio/07.mp4']
    legacy=['--yolo-engine','models/joint_20261018/joint_fp16.plan','--rtmw-engine','build/rtmw.plan']
    directory=out/'quality_legacy';directory.mkdir(exist_ok=True)
    run('quality_legacy',base+legacy+['--headless','--sample-dir',str(out/'samples'),'--validation-dir',str(directory)])
    for height in [384,640]:
        plan=models/f'obb_640x{height}_b1_4.plan'
        if not plan.exists():
            run('build_obb_'+str(height),['python3','-u','scripts/build/build_batch_engine.py',str(models/f'obb_640x{height}_silu.onnx'),
                 str(plan),'--kind','obb','--height',str(height),'--width','640','--plugin-library','build/libsop_silu_plugin.so',
                 '--cache',str(models/'obb.timing.cache')],timeout=2400)
        if height==384:
            directory=out/'quality_rect11_early';directory.mkdir(exist_ok=True)
            run('quality_rect11_early',base+['--det-height','384','--headless','--sample-dir',str(out/'samples'),
                 '--validation-dir',str(directory)])
            for name,options in [('early_legacy',legacy),('early_rect11',['--det-height','384'])]:
                run(name,base+options+['--headless','--metrics',str(out/(name+'_metrics.csv')),
                    '--batch-metrics',str(out/(name+'_groups.csv'))])
    for batch in [2,4]:
        plan=models/f'rtmw_b{batch}.plan'
        if not plan.exists():
            run('build_rtmw_b'+str(batch),['python3','-u','scripts/build/build_batch_engine.py',
                str(models/f'rtmw_b{batch}_static.onnx'),str(plan),
                '--kind','rtmw','--height','256','--width','192','--fixed-batch',str(batch),
                '--cache',str(models/'rtmw_fixed.timing.cache')],timeout=2400)
    telemetry_log=(out/'telemetry.log').open('w')
    telemetry=subprocess.Popen(['tegrastats','--interval','1000'],stdout=telemetry_log,stderr=subprocess.STDOUT)
    try:
        for model,height,width,plan,binding in [('obb640',640,640,models/'obb_640x640_b1_4.plan','images'),
               ('obb384',384,640,models/'obb_640x384_b1_4.plan','images'),('rtmw',256,192,None,'input')]:
            for batch in [1,2,4]:
                selected_plan=plan if plan is not None else (root/'build/rtmw.plan' if batch==1 else models/f'rtmw_b{batch}.plan')
                name=f'engine_{model}_b{batch}'
                run(name,['/usr/src/tensorrt/bin/trtexec','--loadEngine='+str(selected_plan),'--plugins=build/libsop_silu_plugin.so',
                    f'--shapes={binding}:{batch}x3x{height}x{width}','--noDataTransfers','--warmUp=1500','--duration=5',
                    '--iterations=100','--exportTimes='+str(out/(name+'.json'))])
        cases=[('legacy',640,1,1,legacy),('square11',640,1,1,[]),('rect11',384,1,1,[]),
               ('square21',640,2,1,[]),('square41',640,4,1,[]),('square12',640,1,2,[]),('square14',640,1,4,[]),
               ('square22',640,2,2,[]),('square44',640,4,4,[]),('rect22',384,2,2,[]),('rect44',384,4,4,[])]
        for name,height,db,pb,extra in cases:
            options=['--det-height',str(height),'--det-batch',str(db),'--pose-batch',str(pb)]+extra
            directory=out/('quality_'+name);directory.mkdir(exist_ok=True)
            if name!='legacy':
                run('quality_'+name,base+options+['--headless','--sample-dir',str(out/'samples'),'--validation-dir',str(directory)])
            run('headless_'+name,base+options+['--headless','--metrics',str(out/('headless_'+name+'_metrics.csv')),
                '--batch-metrics',str(out/('headless_'+name+'_groups.csv'))])
        # Same-shape and rectangle candidates both remain reviewable; do not auto-select on FPS alone.
        for name,height,db,pb,extra in [cases[0],cases[2],cases[-2],cases[-1]]:
            run('gui_'+name,base+['--det-height',str(height),'--det-batch',str(db),'--pose-batch',str(pb)]+extra+
                ['--metrics',str(out/('gui_'+name+'_metrics.csv')),'--batch-metrics',str(out/('gui_'+name+'_groups.csv'))])
    finally:
        telemetry.terminate();telemetry.wait(timeout=10);telemetry_log.close()
    print('ALL_BATCH_EXPERIMENTS_DONE',flush=True)

if __name__=='__main__':main()
