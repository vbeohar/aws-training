# Convert an ONNX model to CNTK model.

import cntk as C
import numpy as np
from PIL import Image
import pickle
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))

# Import the Chainer model into CNTK via CNTK's import API
z = C.Function.load(curr_dir + "/vgg16.onnx", device=C.device.cpu(), format=C.ModelFormat.ONNX)
print("Loaded vgg16.onnx!")
img = Image.open(curr_dir + "/Siberian_Husky_bi-eyed_Flickr.jpg")
img = img.resize((224,224))
rgb_img = np.asarray(img, dtype=np.float32) - 128
bgr_img = rgb_img[..., [2,1,0]]
img_data = np.ascontiguousarray(np.rollaxis(bgr_img,2))
predictions = np.squeeze(z.eval({z.arguments[0]:[img_data]}))
top_class = np.argmax(predictions)
assert top_class==248
labels_dict = pickle.load(open(curr_dir + "/imagenet1000_clsid_to_human.pkl", "rb"))
assert labels_dict[top_class] == 'Eskimo dog, husky'
