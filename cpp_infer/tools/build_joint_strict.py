"""Build on Jetson: constrain floating backbone ops to FP16, leave head decode mixed."""
import argparse
from pathlib import Path
import tensorrt as trt
import ctypes

p=argparse.ArgumentParser()
p.add_argument('onnx',type=Path);p.add_argument('engine',type=Path)
p.add_argument('--first-tactic',action='store_true',help='Diagnostic engine: restrict each layer to its first available tactic')
p.add_argument('--linear-only',action='store_true',help='Restrict tensor format masks to LINEAR')
p.add_argument('--plugin-library',type=Path)
a=p.parse_args()
if a.engine.exists():raise SystemExit('Refusing to overwrite existing engine')
if a.plugin_library:ctypes.CDLL(str(a.plugin_library.resolve()),mode=ctypes.RTLD_GLOBAL)
log=trt.Logger(trt.Logger.VERBOSE)
builder=trt.Builder(log)
builder.max_threads=6
network=builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser=trt.OnnxParser(network,log)
if not parser.parse(a.onnx.read_bytes()):
    raise RuntimeError([str(parser.get_error(i)) for i in range(parser.num_errors)])
config=builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,1024*1024*1024)
config.set_flag(trt.BuilderFlag.FP16)
config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
types={trt.LayerType.CONVOLUTION,trt.LayerType.ACTIVATION,trt.LayerType.POOLING,
       trt.LayerType.ELEMENTWISE,trt.LayerType.CONCATENATION,trt.LayerType.MATRIX_MULTIPLY,
       trt.LayerType.SOFTMAX,trt.LayerType.UNARY,trt.LayerType.REDUCE}
count=0
for layer in network:
    if a.linear_only:
        for j in range(layer.num_outputs):
            layer.get_output(j).allowed_formats=1 << int(trt.TensorFormat.LINEAR)
    if layer.type not in types or layer.name.startswith('/model.23/'):
        continue
    floating=[j for j in range(layer.num_outputs) if layer.get_output(j).dtype==trt.float32]
    if not floating:continue
    layer.precision=trt.float16
    for j in floating:layer.set_output_type(j,trt.float16)
    count+=1
print('CONSTRAINED_BACKBONE_LAYERS',count,flush=True)
if a.linear_only:
    for i in range(network.num_inputs):network.get_input(i).allowed_formats=1 << int(trt.TensorFormat.LINEAR)
config.avg_timing_iterations=1
class First(trt.IAlgorithmSelector):
    def __init__(self): trt.IAlgorithmSelector.__init__(self)
    def select_algorithms(self,context,choices):
        print('SELECT_FIRST',context.name,'choices',len(choices),flush=True)
        return [0] if choices else []
    def report_algorithms(self,contexts,choices):
        print('SELECTED_LAYERS',len(choices),flush=True)
selector=First()
if a.first_tactic:config.algorithm_selector=selector
engine=builder.build_serialized_network(network,config)
if engine is None:raise RuntimeError('TensorRT build returned None')
a.engine.write_bytes(engine)
print('BUILD_COMPLETE',a.engine,flush=True)
