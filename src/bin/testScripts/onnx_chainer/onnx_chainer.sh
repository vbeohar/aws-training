#!/bin/bash
set -e
set -x

echo "Loading a vgg16 network running mock data through it and exporting the model to vgg16.onnx file."
python $(dirname "$0")/assets/chainer_onnx.py

echo "Removing the exported file."
rm $(dirname "$0")/assets/vgg16.onnx

exit
