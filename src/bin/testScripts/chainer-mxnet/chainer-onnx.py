# Convert Chainer model to ONNX model

import numpy as np
import chainer
import chainercv.links as L
import onnx_chainer
import os

curr_dir = os.path.dirname(__file__)

# Fetch a vgg16 model
model = L.VGG16(pretrained_model='imagenet')

# Prepare an input tensor
x = np.random.rand(1, 3, 224, 224).astype(np.float32) * 255

# Run the model on the data
with chainer.using_config('train', False):
    chainer_out = model(x).array

# Export the model to a .onnx file
out = onnx_chainer.export(model, x, filename=curr_dir + '/vgg16.onnx')

# Check that the newly created model is valid and meets ONNX specification.
import onnx
model_proto = onnx.load(curr_dir + "/vgg16.onnx")
onnx.checker.check_model(model_proto)
