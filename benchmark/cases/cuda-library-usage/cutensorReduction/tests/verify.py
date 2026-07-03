#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../../../tools")
import verify_lib as V

EM, EH, EK, EV = 48, 32, 16, 16


def reference(meta):
    a = (V.gen_hash01(EM * EH * EK * EV, 51) - V.F32(0.5)).astype(np.float64)
    A = a.reshape(EV, EK, EH, EM)          # modes A: m,h,k,v -> np [v,k,h,m]
    # C_{m,v} = 1.1 * sum_{h,k} A_{m,h,k,v}
    C = 1.1 * A.sum(axis=(1, 2))           # np [v,m], m fastest
    return C.reshape(-1).astype(V.F32)


if __name__ == "__main__":
    V.run(reference)
