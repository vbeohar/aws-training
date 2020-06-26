#!/bin/bash
set -e
set -x

source activate mxnet_p36
python $(dirname "$0")/mxnet-onnx.py
source deactivate

source activate cntk_p36
wget https://s3.amazonaws.com/onnx-mxnet/dlami-blogpost/Siberian_Husky_bi-eyed_Flickr.jpg -P $(dirname "$0")/
wget https://s3.amazonaws.com/onnx-mxnet/dlami-blogpost/imagenet1000_clsid_to_human.pkl -P $(dirname "$0")/
python $(dirname "$0")/onnx-cntk.py
source deactivate

exit
