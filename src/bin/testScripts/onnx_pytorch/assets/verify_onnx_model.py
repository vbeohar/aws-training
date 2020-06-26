import os

import onnx

curr_dir = os.path.dirname(__file__)
model_proto = onnx.load(curr_dir + "/pytorch_model.onnx")
onnx.checker.check_model(model_proto)

graph = model_proto.graph
inputs = []
for i in graph.input:
    inputs.append(i.name)
assert inputs == ['input']

params = []
for tensor_vals in graph.initializer:
    params.append(tensor_vals.name)
assert params == ['conv.weight']

nodes = []
for node in graph.node:
    nodes.append(node.op_type)
assert nodes == ['Conv', 'ReduceMean']
