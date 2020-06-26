#!/bin/bash
set -e
set -x

python $(dirname "$0")/assets/onnx_mxnet.py

exit
