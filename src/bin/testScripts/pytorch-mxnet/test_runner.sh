#!/bin/bash
set -e
set -x

source activate pytorch_p36
python $(dirname "$0")/pytorch-onnx.py
source deactivate

source activate mxnet_p36
python $(dirname "$0")/onnx-mxnet.py
source deactivate

exit
