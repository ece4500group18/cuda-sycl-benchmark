#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def _sample(img, y, x, ch):
    h, w, _ = img.shape
    if x < 0 or y < 0 or x > w - 1 or y > h - 1:
        return np.float32(0)
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    wx, wy = np.float32(x - x0), np.float32(y - y0)
    return (np.float32(1) - wy) * ((np.float32(1) - wx) * img[y0, x0, ch] + wx * img[y0, x1, ch]) + wy * ((np.float32(1) - wx) * img[y1, x0, ch] + wx * img[y1, x1, ch])

def reference(meta):
    b, h, w, c = meta["input"]["sizes"]
    img = V.gen_hash01(b * h * w * c, meta["input"]["seed"]).reshape(h, w, c).astype(np.float32)
    out = np.empty_like(img)
    for y in range(h):
        for x in range(w):
            sx = np.float32(0.92) * x + np.float32(0.12) * y - np.float32(2.3)
            sy = -np.float32(0.08) * x + np.float32(1.04) * y + np.float32(1.7)
            for ch in range(c):
                out[y, x, ch] = _sample(img, sy, sx, ch)
    return out.reshape(-1)


if __name__ == "__main__":
    V.run(reference)
