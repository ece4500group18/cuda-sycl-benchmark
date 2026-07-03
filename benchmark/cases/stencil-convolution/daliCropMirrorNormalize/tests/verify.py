#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    b, h, w, c, oh, ow = meta["input"]["sizes"]
    x = V.gen_hash01(b * h * w * c, meta["input"]["seed"]).reshape(b, h, w, c).astype(np.float32)
    out = np.empty((b, c, oh, ow), dtype=np.float32)
    means = np.asarray([0.45, 0.50, 0.55], dtype=np.float32)
    stds = np.asarray([0.20, 0.25, 0.30], dtype=np.float32)
    anchors = [(4, 6), (8, 5)]
    for n in range(b):
        y0, x0 = anchors[n]
        crop = x[n, y0:y0 + oh, x0:x0 + ow, :]
        if n == 1:
            crop = crop[:, ::-1, :]
        out[n] = ((crop - means) / stds).transpose(2, 0, 1)
    return out.reshape(-1)


if __name__ == "__main__":
    V.run(reference)
