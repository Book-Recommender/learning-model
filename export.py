# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "onnxruntime",
#     "pandas",
#     "skl2onnx",
# ]
# ///

import model
import numpy as np
from skl2onnx import to_onnx


if __name__ == "__main__":
    onx = to_onnx(model.clr, model.X_train[:1].astype(np.float32))

    with open("model.onnx", "wb") as f:
        f.write(onx.SerializeToString())
