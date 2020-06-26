#!/usr/bin/env python

import io
import onnx
import os
import tempfile
from onnx import AttributeProto, NodeProto, GraphProto, ModelProto, IR_VERSION
 
# Create a model proto.
model = ModelProto()
model.ir_version = IR_VERSION
model_string = model.SerializeToString()
 
# Test if input is string
loaded_model = onnx.load_from_string(model_string)
assert model == loaded_model
 
# Test if input has a read function
f = io.BytesIO(model_string)
loaded_model = onnx.load(f)
assert model == loaded_model
 
# Test if input is a file name
f = tempfile.NamedTemporaryFile(delete=False)
f.write(model_string)
f.close()
loaded_model = onnx.load(f.name)
assert model == loaded_model
os.remove(f.name)
 
try:
    AttributeProto
    NodeProto
    GraphProto
    ModelProto
except Exception as e:
    assert False, 'Did not find proper onnx protobufs. Error is: {}'.format(e)
 
model = ModelProto()
# When we create it, graph should not have a version string.
assert not model.HasField('ir_version')
# We should touch the version so it is annotated with the current
# ir version of the running ONNX
model.ir_version = IR_VERSION
model_string = model.SerializeToString()
model.ParseFromString(model_string)
assert model.HasField('ir_version')
# Check if the version is correct.
assert model.ir_version == IR_VERSION
