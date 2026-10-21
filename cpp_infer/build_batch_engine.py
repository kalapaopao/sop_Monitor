"""Build separate experiment engines; never overwrite current plans."""
import argparse
import ctypes
from pathlib import Path
import time
import tensorrt as trt

p=argparse.ArgumentParser()
p.add_argument('source',type=Path); p.add_argument('destination',type=Path)
p.add_argument('--kind',choices=['obb','rtmw'],required=True)
p.add_argument('--height',type=int,required=True); p.add_argument('--width',type=int,required=True)
p.add_argument('--plugin-library',type=Path)
p.add_argument('--cache',type=Path,required=True)
p.add_argument('--fixed-batch',type=int,choices=[1,2,4])
a=p.parse_args()
if a.kind=='rtmw' and a.fixed_batch:
    specialized=a.destination.parent/f'rtmw_b{a.fixed_batch}_static.onnx'
    if specialized.exists():a.source=specialized
print('SOURCE_ONNX',a.source,flush=True)
if a.destination.exists():raise SystemExit('Refusing to overwrite existing plan')
if a.plugin_library:ctypes.CDLL(str(a.plugin_library.resolve()),mode=ctypes.RTLD_GLOBAL)
logger=trt.Logger(trt.Logger.INFO)
builder=trt.Builder(logger); builder.max_threads=6
network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser=trt.OnnxParser(network,logger)
if not parser.parse(a.source.read_bytes()):raise RuntimeError([str(parser.get_error(i)) for i in range(parser.num_errors)])
if network.num_inputs!=1:raise RuntimeError('Expected one model input')
if a.fixed_batch:network.get_input(0).shape=(a.fixed_batch,3,a.height,a.width)
config=builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,(2048 if a.kind=='rtmw' else 1024)*1024*1024)
config.set_flag(trt.BuilderFlag.FP16)
config.avg_timing_iterations=1
if a.kind=='obb':
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    allowed={trt.LayerType.CONVOLUTION,trt.LayerType.ACTIVATION,trt.LayerType.POOLING,
             trt.LayerType.ELEMENTWISE,trt.LayerType.CONCATENATION,trt.LayerType.MATRIX_MULTIPLY,
             trt.LayerType.SOFTMAX,trt.LayerType.UNARY,trt.LayerType.REDUCE}
    count=0
    for layer in network:
        if layer.type not in allowed or '/model.23/' in layer.name:continue
        floating=[j for j in range(layer.num_outputs) if layer.get_output(j).dtype==trt.float32]
        if not floating:continue
        layer.precision=trt.float16
        for j in floating:layer.set_output_type(j,trt.float16)
        count+=1
    print('CONSTRAINED_BACKBONE_LAYERS',count,flush=True)
name=network.get_input(0).name
if not a.fixed_batch:
    profile=builder.create_optimization_profile()
    profile.set_shape(name,(1,3,a.height,a.width),(2,3,a.height,a.width),(4,3,a.height,a.width))
    if not profile:raise RuntimeError('Invalid batch1..4 profile')
    config.add_optimization_profile(profile)
cache=config.create_timing_cache(a.cache.read_bytes() if a.cache.exists() else b'')
config.set_timing_cache(cache,False)
print('BUILD_START',a.kind,a.height,a.width,'fixed batch '+str(a.fixed_batch) if a.fixed_batch else 'batch 1/2/4 opt=2',flush=True)
start=time.monotonic()
engine=builder.build_serialized_network(network,config)
if engine is None:raise RuntimeError('Build returned None')
a.destination.write_bytes(engine)
a.cache.write_bytes(config.get_timing_cache().serialize())
print('BUILD_COMPLETE',a.destination,'seconds',time.monotonic()-start,flush=True)
