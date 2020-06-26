#!/usr/bin/env python

import os
import cntk as C
import onnx

curr_dir = os.path.dirname(__file__)

z = C.Function.load(curr_dir + "/conv.onnx", device=C.device.cpu(), format=C.ModelFormat.ONNX)
z.save(curr_dir + "/exported_model.onnx", format=C.ModelFormat.ONNX)


model_proto = onnx.load(curr_dir + "/exported_model.onnx")
onnx.checker.check_model(model_proto)
