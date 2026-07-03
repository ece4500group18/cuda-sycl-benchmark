#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../../../tools")
import verify_lib as V

EW, EH, EC, EN = 8, 16, 16, 16


def reference(meta):
    a = V.gen_hash01(EW * EH * EC * EN, 61) - V.F32(0.5)
    A = a.reshape(EN, EC, EH, EW)          # modes A: w,h,c,n -> np [n,c,h,w]
    # C_{c,w,h,n}: np layout [n,h,w,c], c fastest
    C = A.transpose(0, 2, 3, 1)
    return np.ascontiguousarray(C).reshape(-1).astype(V.F32)


if __name__ == "__main__":
    V.run(reference)
