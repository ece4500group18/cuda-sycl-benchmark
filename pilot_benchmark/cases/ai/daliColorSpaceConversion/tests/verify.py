#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    b, h, w, c = meta["input"]["sizes"]
    x = V.gen_hash01(b * h * w * c, meta["input"]["seed"]).reshape(-1, 3).astype(np.float32)
    r, g, bb = x[:, 0], x[:, 1], x[:, 2]
    out = np.empty_like(x)
    out[:, 0] = np.float32(0.299) * r + np.float32(0.587) * g + np.float32(0.114) * bb
    out[:, 1] = np.float32(-0.168736) * r - np.float32(0.331264) * g + np.float32(0.5) * bb + np.float32(0.5)
    out[:, 2] = np.float32(0.5) * r - np.float32(0.418688) * g - np.float32(0.081312) * bb + np.float32(0.5)
    return out.reshape(-1)


if __name__ == "__main__":
    V.run(reference)
