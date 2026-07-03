#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    b, d, h, w, od, oh, ow = meta["input"]["sizes"]
    x = V.gen_hash01(b * d * h * w, meta["input"]["seed"]).reshape(b, d, h, w)
    out = np.empty((b, od, oh, ow), dtype=np.float32)
    anchors = [(1, 2, 3), (2, 1, 4)]
    for n in range(b):
        z0, y0, x0 = anchors[n]
        out[n] = x[n, z0:z0 + od, y0:y0 + oh, x0:x0 + ow]
    return out.reshape(-1)


if __name__ == "__main__":
    V.run(reference)
