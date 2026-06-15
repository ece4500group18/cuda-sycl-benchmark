#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    b, h, w, c = meta["input"]["sizes"]
    x = V.gen_hash01(b * h * w * c, meta["input"]["seed"]).reshape(-1, 3).astype(np.float32)
    brightness, contrast, saturation, hue = np.float32(0.08), np.float32(1.15), np.float32(0.85), np.float32(0.25)
    rgb = (x - np.float32(0.5)) * contrast + np.float32(0.5) + brightness
    r, g, bb = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    y = np.float32(0.299) * r + np.float32(0.587) * g + np.float32(0.114) * bb
    u, v = r - y, bb - y
    co, si = np.float32(np.cos(hue)), np.float32(np.sin(hue))
    ru = saturation * (co * u - si * v)
    rv = saturation * (si * u + co * v)
    out = np.empty_like(rgb)
    out[:, 0] = y + ru
    out[:, 1] = y - np.float32(0.509) * ru - np.float32(0.194) * rv
    out[:, 2] = y + rv
    return out.reshape(-1)


if __name__ == "__main__":
    V.run(reference)
