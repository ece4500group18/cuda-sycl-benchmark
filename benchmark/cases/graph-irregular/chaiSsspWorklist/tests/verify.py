#!/usr/bin/env python3
import heapq
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../../../tools")
import verify_lib as V

N, PICKS, SOURCE = 4096, 4, 0
INF = -2147483647


def build_graph():
    adj = {}
    for k in range(PICKS):
        w = V.gen_index(N, N, 600 + k)
        for u in range(N):
            if w[u] != u:
                adj.setdefault(u, set()).add(int(w[u]))
    # CSR order (ascending targets) determines each edge's weight-hash index
    edges = {}
    for u in range(N):
        targets = sorted(adj.get(u, ()))
        for j, t in enumerate(targets):
            wgt = 1 + int(np.float32(V.gen_hash01(u * 16 + j + 1, 610)[-1]) * np.float32(9.0))
            edges.setdefault(u, []).append((t, wgt))
    return edges


def reference(meta):
    edges = build_graph()
    # Dijkstra ground truth; the worklist kernel's atomicMax relaxation over
    # negated costs converges to the same shortest distances.
    dist = {SOURCE: 0}
    pq = [(0, SOURCE)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, 1 << 60):
            continue
        for t, w in edges.get(u, ()):
            nd = d + w
            if nd < dist.get(t, 1 << 60):
                dist[t] = nd
                heapq.heappush(pq, (nd, t))
    out = np.full(N, INF, dtype=np.int64)
    for v, d in dist.items():
        out[v] = -d  # costs are stored negated
    return out.astype(np.float64).astype(V.F32)


if __name__ == "__main__":
    V.run(reference)
