#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../../../tools")
import verify_lib as V

IN, HID = 4096, 16
NB = IN // 16
ETA, MOM = np.float32(0.3), np.float32(0.3)


def reference(meta):
    wsize = (IN + 1) * (HID + 1)
    inputs = V.gen_hash01(IN + 1, 91)
    W = (np.float32(2.0) * V.gen_hash01(wsize, 92) - np.float32(1.0))
    oldw = np.float32(0.1) * (np.float32(2.0) * V.gen_hash01(wsize, 93) - np.float32(1.0))
    delta = np.float32(0.1) * (np.float32(2.0) * V.gen_hash01(HID + 1, 94) - np.float32(1.0))

    # --- kernel_layerforward: per block, wm[ty,tx] = W[row,col]*input[row],
    # then an in-block tree reduction over ty (exact f32 order).
    rows = np.arange(1, IN + 1)
    Wmat = W.reshape(IN + 1, HID + 1)
    P = (Wmat[rows, 1:] * inputs[rows][:, None]).astype(V.F32)   # (IN, HID)
    P = P.reshape(NB, 16, HID)
    # NB: upstream's loop starts at i=1 with power_two/2 == 0, i.e. every
    # element first adds ITSELF (doubling) - replicated exactly.
    i = 1
    while i <= 16:
        half = i // 2
        idx = np.arange(0, 16, i)
        src = idx + half
        valid = src < 16
        P[:, idx[valid], :] = (P[:, idx[valid], :] + P[:, src[valid], :]).astype(V.F32)
        i *= 2
    partial = P[:, 0, :]                                          # (NB, HID)

    # --- host squash (sequential f32 accumulation over blocks)
    hidden = np.empty(HID, dtype=V.F32)
    for j in range(1, HID + 1):
        s = np.float32(0.0)
        for k in range(NB):
            s = np.float32(s + partial[k, j - 1])
        s = np.float32(s + W[j])
        hidden[j - 1] = np.float32(1.0 / (1.0 + np.exp(-np.float64(s))))

    # --- kernel_adjust_weights (weights restored before this kernel)
    Wadj = W.copy().reshape(IN + 1, HID + 1)
    oldw2 = oldw.reshape(IN + 1, HID + 1)
    upd = (ETA * delta[1:][None, :] * inputs[rows][:, None] + MOM * oldw2[rows, 1:]).astype(V.F32)
    Wadj[rows, 1:] = (Wadj[rows, 1:] + upd).astype(V.F32)
    bias_upd = (ETA * delta[1:] + MOM * oldw2[0, 1:]).astype(V.F32)
    Wadj[0, 1:] = (Wadj[0, 1:] + bias_upd).astype(V.F32)

    return np.concatenate([hidden, Wadj.reshape(-1)]).astype(V.F32)


if __name__ == "__main__":
    V.run(reference)
