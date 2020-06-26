import os
import numpy as np

import chainer
import chainercv.links as C
import onnx_chainer
import onnx

curr_dir = os.path.dirname(__file__)

model = C.VGG16(pretrained_model='imagenet')

# Pseudo input
x = np.zeros((1, 3, 224, 224), dtype=np.float32)

out = onnx_chainer.export(model, x, filename=curr_dir + '/vgg16.onnx')

model_proto = onnx.load(curr_dir + "/vgg16.onnx")
onnx.checker.check_model(model_proto)
