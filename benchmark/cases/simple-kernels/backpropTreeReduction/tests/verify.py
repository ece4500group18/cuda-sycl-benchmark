#!/usr/bin/env python3
"""backpropTreeReduction: CPU reference for the layer-forward weighted partial
sum, performing the identical power-of-two-stride tree reduction (same float32
add order) as the GPU kernel.

Deterministic inputs:
  input_node[i]       = ((i % 7) - 3) * 0.5
  weight_matrix[i][j] = (((i + j) % 5) - 2) * 0.1
Output: 512 x 16 = 8192 hidden_partial_sum floats.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V  # noqa: E402

WIDTH = 16
HEIGHT = 16


def reference(meta):
    n_in = int(meta["input"]["sizes"][0])   # 8192
    hid = WIDTH
    num_blocks = n_in // HEIGHT              # 512
    f32 = np.float32

    # input_node[i] = ((i % 7) - 3) * 0.5, i in [0, in]
    idx = np.arange(n_in + 1)
    inp = (((idx % 7) - 3).astype(f32)) * f32(0.5)

    out = np.empty(num_blocks * hid, dtype=f32)
    for by in range(num_blocks):
        for tx in range(hid):
            col = np.empty(HEIGHT, dtype=f32)
            for ty in range(HEIGHT):
                row = HEIGHT * by + ty + 1
                w = f32((((row + (tx + 1)) % 5) - 2)) * f32(0.1)   # gen_weight(row, tx+1)
                col[ty] = f32(w * inp[row])
            stride = 2
            while stride <= HEIGHT:
                for ty in range(0, HEIGHT, stride):
                    col[ty] = f32(col[ty] + col[ty + stride // 2])
                stride *= 2
            out[by * hid + tx] = col[0]
    return out


if __name__ == "__main__":
    V.run(reference)
