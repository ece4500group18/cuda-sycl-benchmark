#!/usr/bin/env python3
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V
"""embedding: gather table rows by ids."""


def reference(meta):
    vocab, dim, num_ids = meta["input"]["sizes"]
    table = V.gen_hash01(vocab * dim, 123).reshape(vocab, dim)
    ids = V.gen_index(num_ids, vocab, 777)
    return table[ids].reshape(-1)


if __name__ == "__main__":
    V.run(reference)
