"""Replace exact x*sigmoid(x) pairs with a registered TensorRT CUDA plugin."""
import argparse
from pathlib import Path
import onnx
from onnx import helper

p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');a=p.parse_args()
m=onnx.load(a.source)
consumers={}
for node in m.graph.node:
    for x in node.input:consumers.setdefault(x,[]).append(node)
replacement={};skip=set();count=0
for node in m.graph.node:
    if node.op_type!='Sigmoid':continue
    uses=consumers.get(node.output[0],[])
    if len(uses)!=1:continue
    mul=uses[0]
    if mul.op_type!='Mul' or sorted(mul.input)!=sorted([node.input[0],node.output[0]]):continue
    replacement[mul.name]=helper.make_node('SopSiLU',[node.input[0]],list(mul.output),name=mul.name,domain='sop')
    skip.add(node.name);count+=1
nodes=[replacement.get(n.name,n) for n in m.graph.node if n.name not in skip]
del m.graph.node[:];m.graph.node.extend(nodes)
m.opset_import.append(helper.make_opsetid('sop',1))
assert count>0
onnx.checker.check_model(m)
onnx.save(m,a.destination)
print('REPLACED_EXACT_SILU_PAIRS',count)
