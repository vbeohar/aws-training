#!/usr/bin/env python

import os
import mxnet as mx
from mxnet.contrib import onnx as onnx_mxnet
import onnx
import numpy as np
from onnx import TensorProto
from onnx import numpy_helper

curr_dir = os.path.dirname(__file__)

sym, arg_params, aux_params = onnx_mxnet.import_model(curr_dir + "/model.onnx")

input_tensor = TensorProto()
with open(curr_dir + "/input_0.pb", 'rb') as proto_file:
	input_tensor.ParseFromString(proto_file.read())
input_array = numpy_helper.to_array(input_tensor)

x = mx.nd.array(input_array)

mod = mx.mod.Module(symbol=sym, data_names=['0'], context=mx.cpu(), label_names=None)
mod.bind(for_training=False, data_shapes=[('0', (2, 4, 6, 6))], label_shapes=None)
mod.set_params(arg_params=arg_params, aux_params=aux_params)
mod.forward(mx.io.DataBatch([x]))
result = mod.get_outputs()[0].asnumpy()

output_tensor = TensorProto()
with open(curr_dir + "/output_0.pb", 'rb') as proto_file:
	output_tensor.ParseFromString(proto_file.read())
output_array = numpy_helper.to_array(output_tensor)

np.testing.assert_allclose(result, output_array, rtol=1e-3, atol=1e-3)
