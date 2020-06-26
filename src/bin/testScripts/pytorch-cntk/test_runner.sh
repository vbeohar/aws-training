#!/bin/bash
set -e
set -x

source activate pytorch_p36
python $(dirname "$0")/pytorch-onnx.py
source deactivate

source activate cntk_p36
python $(dirname "$0")/onnx-cntk.py
source deactivate

exit
