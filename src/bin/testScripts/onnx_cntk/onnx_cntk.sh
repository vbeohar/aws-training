#!/bin/bash
set -e
set -x

python $(dirname "$0")/assets/cntk_onnx.py
rm $(dirname "$0")/assets/exported_model.onnx

exit

