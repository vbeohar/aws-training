import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.onnx as torch_onnx

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), stride=1, padding=0, bias=False)

    def forward(self, inputs):
        x = self.conv(inputs)
        return torch.mean(x, dim=2)

input_shape = (3, 100, 100)

curr_dir = os.path.dirname(__file__)
model_onnx_path = curr_dir + "/pytorch_model.onnx"

model = Model()
model.train(False)

dummy_input = Variable(torch.randn(1, *input_shape))
torch_onnx.export(model, 
                  dummy_input, 
                  model_onnx_path, 
                  verbose=True)

