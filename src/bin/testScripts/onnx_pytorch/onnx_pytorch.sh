#!/bin/bash
set -e
set -x

python $(dirname "$0")/assets/pytorch_export.py
python $(dirname "$0")/assets/verify_onnx_model.py

exit
