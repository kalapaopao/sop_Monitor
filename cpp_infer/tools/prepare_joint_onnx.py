"""Extract pre-TopK predictions; does not alter weights or original ONNX."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import onnx

p = argparse.ArgumentParser()
p.add_argument('source', type=Path)
p.add_argument('destination', type=Path)
a = p.parse_args()
m = onnx.load(str(a.source))
metadata = {x.key: x.value for x in m.metadata_props}
assert ast.literal_eval(metadata['names']) == {0:'screwdriver1',1:'screwdriver2',2:'hand',3:'person'}
shaped = onnx.shape_inference.infer_shapes(m)
output = '/model.23/Transpose_output_0'
shape = next(v for v in shaped.graph.value_info if v.name == output)
assert [d.dim_value for d in shape.type.tensor_type.shape.dim] == [1,8400,9]
onnx.utils.extract_model(str(a.source), str(a.destination), [m.graph.input[0].name], [output])
raw = onnx.load(str(a.destination))
assert not any(n.op_type == 'TopK' for n in raw.graph.node)
onnx.checker.check_model(raw)
info = {'source_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),
        'raw_sha256':hashlib.sha256(a.destination.read_bytes()).hexdigest(),
        'classes':ast.literal_eval(metadata['names']), 'output':[1,8400,9],
        'change':'Remove end-to-end TopK tail only; C++ uses classwise rotated NMS.'}
a.destination.with_suffix('.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
print(json.dumps(info,indent=2))
