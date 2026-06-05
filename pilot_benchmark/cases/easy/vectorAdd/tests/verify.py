#!/usr/bin/env python3
"""vectorAdd: C must equal A + B (max_abs_error)."""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V  # noqa: E402


def reference(meta):
    n = meta["input"]["sizes"][0]
    return V.gen_a(n) + V.gen_b(n)


if __name__ == "__main__":
    V.run(reference)
