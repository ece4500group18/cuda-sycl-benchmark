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
    angle = np.float32(-17.0 * np.pi / 180.0)
    co, si = np.float32(np.cos(angle)), np.float32(np.sin(angle))
    cx, cy = np.float32(0.5 * (w - 1)), np.float32(0.5 * (h - 1))
    for y in range(h):
        for x in range(w):
            dx, dy = np.float32(x) - cx, np.float32(y) - cy
            sx, sy = co * dx - si * dy + cx, si * dx + co * dy + cy
            for ch in range(c):
                out[y, x, ch] = _sample(img, sy, sx, ch)
    return out.reshape(-1)


if __name__ == "__main__":
    V.run(reference)
