#!/usr/bin/env python3
"""matrixMul: C must equal A*B (max_rel_error)."""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V  # noqa: E402


def reference(meta):
    N = meta["input"]["sizes"][0]
    A = V.gen_a(N * N).reshape(N, N)
    B = V.gen_b(N * N).reshape(N, N)
    return A @ B


if __name__ == "__main__":
    V.run(reference)
