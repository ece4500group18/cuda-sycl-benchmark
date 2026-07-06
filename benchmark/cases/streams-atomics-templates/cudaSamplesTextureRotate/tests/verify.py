#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../../../tools")
import verify_lib as V

W = H = 128
THETA = np.float32(0.5)


def reference(meta):
    x = np.arange(W, dtype=np.float64)
    y = np.arange(H, dtype=np.float64)
    img = (0.5 + 0.25 * np.sin(2 * np.pi * x[None, :] / W)
               + 0.25 * np.cos(2 * np.pi * y[:, None] / H))  # img[y, x]

    xs = np.arange(W, dtype=np.float64)
    ys = np.arange(H, dtype=np.float64)
    u = xs[None, :] - W / 2.0
    v = ys[:, None] - H / 2.0
    ct, st = np.cos(float(THETA)), np.sin(float(THETA))
    tu = (u * ct - v * st) / W + 0.5
    tv = (v * ct + u * st) / H + 0.5

    # CUDA tex2D with normalized coords, linear filter, wrap addressing:
    # sample at (tu*W - 0.5, tv*H - 0.5) with bilinear weights and wrap.
    fx = tu * W - 0.5
    fy = tv * H - 0.5
    x0 = np.floor(fx).astype(np.int64)
    y0 = np.floor(fy).astype(np.int64)
    ax = fx - x0
    ay = fy - y0
    x0w, x1w = x0 % W, (x0 + 1) % W
    y0w, y1w = y0 % H, (y0 + 1) % H
    out = ((1 - ax) * (1 - ay) * img[y0w, x0w] + ax * (1 - ay) * img[y0w, x1w]
           + (1 - ax) * ay * img[y1w, x0w] + ax * ay * img[y1w, x1w])
    return out.reshape(-1).astype(V.F32)


if __name__ == "__main__":
    V.run(reference)
