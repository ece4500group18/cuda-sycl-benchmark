#!/usr/bin/env python3
"""transpose: out must equal in^T exactly."""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V  # noqa: E402


def reference(meta):
    rows, cols = meta["input"]["sizes"]
    return V.gen_a(rows * cols).reshape(rows, cols).T


if __name__ == "__main__":
    V.run(reference)
