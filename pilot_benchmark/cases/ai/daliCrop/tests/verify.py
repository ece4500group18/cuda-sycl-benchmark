#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    b, h, w, c, oh, ow = meta["input"]["sizes"]
    x = V.gen_hash01(b * h * w * c, meta["input"]["seed"]).reshape(b, h, w, c)
    out = np.empty((b, oh, ow, c), dtype=np.float32)
    anchors = [(5, 7), (9, 3)]
    for n in range(b):
        y0, x0 = anchors[n]
        out[n] = x[n, y0:y0 + oh, x0:x0 + ow, :]
    return out.reshape(-1)


if __name__ == "__main__":
    V.run(reference)
