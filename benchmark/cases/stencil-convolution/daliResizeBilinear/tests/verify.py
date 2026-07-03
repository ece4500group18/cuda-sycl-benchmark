#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    b, ih, iw, c, oh, ow = meta["input"]["sizes"]
    x = V.gen_hash01(b * ih * iw * c, meta["input"]["seed"]).reshape(b, ih, iw, c).astype(np.float32)
    out = np.empty((b, oh, ow, c), dtype=np.float32)
    for n in range(b):
        for oy in range(oh):
            fy = (np.float32(oy) + np.float32(0.5)) * ih / oh - np.float32(0.5)
            y_floor = np.floor(fy)
            y0 = min(max(int(y_floor), 0), ih - 1)
            y1 = min(y0 + 1, ih - 1)
            wy = np.float32(fy - y_floor)
            for ox in range(ow):
                fx = (np.float32(ox) + np.float32(0.5)) * iw / ow - np.float32(0.5)
                x_floor = np.floor(fx)
                x0 = min(max(int(x_floor), 0), iw - 1)
                x1 = min(x0 + 1, iw - 1)
                wx = np.float32(fx - x_floor)
                out[n, oy, ox, :] = (np.float32(1) - wy) * ((np.float32(1) - wx) * x[n, y0, x0, :] + wx * x[n, y0, x1, :]) + wy * ((np.float32(1) - wx) * x[n, y1, x0, :] + wx * x[n, y1, x1, :])
    return out.reshape(-1)


if __name__ == "__main__":
    V.run(reference)
