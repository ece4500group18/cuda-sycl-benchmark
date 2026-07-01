#!/usr/bin/env python3
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V

def reference(meta):
    nodes, deg = meta["input"]["sizes"]
    dist = np.full(nodes, 1000000, dtype=np.int32)
    frontier = np.zeros(nodes, dtype=np.int32)
    for i in range(nodes):
        if i % 97 == 0:
            frontier[i] = 1
            dist[i] = 0
    nextf = np.zeros(nodes, dtype=np.int32)
    for i in range(nodes):
        dests = [(i + 1) % nodes, (i + 17) % nodes, (i * 13 + 7) % nodes]
        if frontier[i]:
            for v in dests:
                if dist[v] > dist[i] + 1:
                    dist[v] = dist[i] + 1
                    nextf[v] = 1
    out = np.where(dist < 1000000, dist, -1).astype(np.float32) + V.F32(0.01) * nextf.astype(np.float32)
    return out

if __name__ == "__main__":
    V.run(reference)
