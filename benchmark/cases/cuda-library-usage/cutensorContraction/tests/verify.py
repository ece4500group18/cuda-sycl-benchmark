#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../../../tools")
import verify_lib as V

EM, EN, EU, EV, EH, EK = 16, 16, 16, 8, 8, 8


def reference(meta):
    # cuTENSOR default layout: first listed mode is fastest-varying
    # (column-major style), so the numpy shape is the reversed mode order.
    a = (V.gen_hash01(EM * EH * EK * EN, 41) - V.F32(0.5)).astype(np.float64)
    b = (V.gen_hash01(EU * EK * EV * EH, 42) - V.F32(0.5)).astype(np.float64)
    A = a.reshape(EN, EK, EH, EM)          # modes A: m,h,k,n -> np [n,k,h,m]
    B = b.reshape(EH, EV, EK, EU)          # modes B: u,k,v,h -> np [h,v,k,u]
    # C_{m,u,n,v} = alpha * sum_{h,k} A_{m,h,k,n} B_{u,k,v,h}
    C = 1.1 * np.einsum("nkhm,hvku->vnum", A, B)   # np [v,n,u,m], m fastest
    return C.reshape(-1).astype(V.F32)


if __name__ == "__main__":
    V.run(reference)
