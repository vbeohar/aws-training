# Convert ONNX model to CNTK

import cntk as C
import os

curr_dir = os.path.dirname(__file__)

# Import the PyTorch model into CNTK via CNTK's import API
z = C.Function.load(curr_dir + "/torch_model.onnx", device=C.device.cpu(), format=C.ModelFormat.ONNX)

# Export the model to ONNX via CNTK's export API
z.save(curr_dir + "/cntk_model.onnx", format=C.ModelFormat.ONNX)
