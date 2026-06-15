#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def _boxes(n, seed, gt=False):
    vals = V.gen_hash01(n * 4 + 16, seed).astype(np.float32)
    boxes = []
    for i in range(n):
        if gt:
            cx = np.float32(0.15) + np.float32(0.7) * vals[i * 4 + 11]
            cy = np.float32(0.15) + np.float32(0.7) * vals[i * 4 + 12]
            ww = np.float32(0.18) + np.float32(0.25) * vals[i * 4 + 13]
            hh = np.float32(0.18) + np.float32(0.25) * vals[i * 4 + 14]
        else:
            cx = np.float32(0.1) + np.float32(0.8) * vals[i * 4 + 0]
            cy = np.float32(0.1) + np.float32(0.8) * vals[i * 4 + 1]
            ww = np.float32(0.08) + np.float32(0.22) * vals[i * 4 + 2]
            hh = np.float32(0.08) + np.float32(0.22) * vals[i * 4 + 3]
        boxes.append([max(np.float32(0), cx - ww * np.float32(0.5)), max(np.float32(0), cy - hh * np.float32(0.5)),
                      min(np.float32(1), cx + ww * np.float32(0.5)), min(np.float32(1), cy + hh * np.float32(0.5))])
    return np.asarray(boxes, dtype=np.float32)

def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(np.float32(0), ix2 - ix1), max(np.float32(0), iy2 - iy1)
    inter = iw * ih
    aa = max(np.float32(0), a[2] - a[0]) * max(np.float32(0), a[3] - a[1])
    ab = max(np.float32(0), b[2] - b[0]) * max(np.float32(0), b[3] - b[1])
    return inter / (aa + ab - inter + np.float32(1e-6))

def reference(meta):
    anchors_n, gts_n = meta["input"]["sizes"]
    anchors = _boxes(anchors_n, 123, False)
    gts = _boxes(gts_n, 321, True)
    out = np.empty((anchors_n, 5), dtype=np.float32)
    for i, a in enumerate(anchors):
        scores = np.asarray([_iou(a, g) for g in gts], dtype=np.float32)
        best = int(np.argmax(scores))
        gt = gts[best]
        aw, ah = a[2] - a[0], a[3] - a[1]
        acx, acy = np.float32(0.5) * (a[0] + a[2]), np.float32(0.5) * (a[1] + a[3])
        gw, gh = gt[2] - gt[0], gt[3] - gt[1]
        gcx, gcy = np.float32(0.5) * (gt[0] + gt[2]), np.float32(0.5) * (gt[1] + gt[3])
        out[i, 0] = (gcx - acx) / aw
        out[i, 1] = (gcy - acy) / ah
        out[i, 2] = np.log(gw / aw)
        out[i, 3] = np.log(gh / ah)
        out[i, 4] = np.float32(best + 1) if scores[best] >= np.float32(0.35) else np.float32(0)
    return out.reshape(-1)


if __name__ == "__main__":
    V.run(reference)
