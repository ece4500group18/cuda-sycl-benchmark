#!/usr/bin/env python3
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V
"""bfs: BFS distances from node 0 (exact)."""


from collections import deque


def reference(meta):
    N = meta["input"]["sizes"][0]
    dist = [-1] * N
    dist[0] = 0
    q = deque([0])
    while q:
        u = q.popleft()
        for v in ((u + 1) % N, (2 * u + 1) % N, (7 * u + 13) % N):
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return np.array(dist, dtype=np.float32)


if __name__ == "__main__":
    V.run(reference)
