#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    tokens, dim, vocab = meta["input"]["sizes"]
    idx = np.floor(V.gen_hash01(tokens, 77) * vocab).astype(np.int64)
    grad = (V.F32(0.01) * V.gen_hashsigned(tokens * dim, 123)).reshape(tokens, dim)
    out = np.zeros((vocab, dim), dtype=np.float32)
    for t in range(tokens):
        out[idx[t]] += grad[t]
    return out.reshape(-1)

if __name__ == "__main__":
    V.run(reference)
