#!/usr/bin/env python3
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V
"""attention: softmax(QK^T/sqrt(d))V."""


def reference(meta):
    seq, dim = meta["input"]["sizes"]
    Q = V.gen_hash01(seq * dim, 11).reshape(seq, dim).astype(np.float64)
    K = V.gen_hash01(seq * dim, 22).reshape(seq, dim).astype(np.float64)
    Vv = V.gen_hash01(seq * dim, 33).reshape(seq, dim).astype(np.float64)
    s = (Q @ K.T) / np.sqrt(dim)
    s = s - s.max(axis=1, keepdims=True)
    e = np.exp(s)
    p = e / e.sum(axis=1, keepdims=True)
    return p @ Vv


if __name__ == "__main__":
    V.run(reference)
