#!/usr/bin/env python3
import os
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../../../tools")
import verify_lib as V

CHUNK, COUNT = 4096, 8


def reference(meta):
    # byte i = (unsigned char)(h01(global_index, 71) * 256.0f)
    data = (V.gen_hash01(COUNT * CHUNK, 71) * np.float32(256.0)).astype(np.uint8)
    out = []
    for c in range(COUNT):
        chunk = data[c * CHUNK:(c + 1) * CHUNK].tobytes()
        # nvcompCRC32 spec is PKZIP CRC32, identical to zlib.crc32
        out.append(zlib.crc32(chunk) & 0xFFFFFFFF)
    return np.array(out, dtype=np.float64).astype(V.F32)


if __name__ == "__main__":
    V.run(reference)
