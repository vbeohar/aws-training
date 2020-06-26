#!/bin/bash
set -e
set -x

source activate chainer_p36
python $(dirname "$0")/chainer-onnx.py
source deactivate

source activate cntk_p36
wget "https://upload.wikimedia.org/wikipedia/commons/b/b5/Siberian_Husky_bi-eyed_Flickr.jpg"
wget "https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/d133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee/imagenet1000_clsid_to_human.pkl"
python $(dirname "$0")/onnx-cntk.py
source deactivate

exit
