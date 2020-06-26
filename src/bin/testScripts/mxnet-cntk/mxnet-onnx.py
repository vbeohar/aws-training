# Convert a MXNet model to ONNX format

import numpy as np
from mxnet.contrib import onnx as onnx_mxnet
import mxnet as mx
import os.path

curr_dir = os.path.dirname(os.path.abspath(__file__))

path='http://data.mxnet.io/models/imagenet/'

# Download vgg16 pretrained model files
mx.test_utils.download(path+'vgg/vgg16-0000.params', curr_dir + '/vgg16-0000.params')
mx.test_utils.download(path+'vgg/vgg16-symbol.json', curr_dir + '/vgg16-symbol.json')

# Export the model to a .onnx file
out = onnx_mxnet.export_model(curr_dir + '/vgg16-symbol.json', curr_dir + '/vgg16-0000.params',
                              [(1,3,224,224)], np.float32, curr_dir + '/vgg16.onnx')

# Check that the newly created model is valid and meets ONNX specification.
import onnx
model_proto = onnx.load(out)
onnx.checker.check_model(model_proto)
