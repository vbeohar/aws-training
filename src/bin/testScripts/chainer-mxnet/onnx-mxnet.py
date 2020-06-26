# Convert an ONNX model to MXNet model

import mxnet as mx
from mxnet.contrib import onnx as onnx_mxnet
import numpy as np
import os

curr_dir = os.path.dirname(__file__)

# Import the ONNX model into MXNet's symbolic interface
sym, arg, aux = onnx_mxnet.import_model(curr_dir + "/vgg16.onnx")
print("Loaded vgg16.onnx!")
print(sym.get_internals())
