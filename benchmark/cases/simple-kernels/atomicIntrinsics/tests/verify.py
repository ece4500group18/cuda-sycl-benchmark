#!/usr/bin/env python3
"""atomicIntrinsics: verify the 11 atomic output slots against a CPU reference.

Nine slots (add/sub/max/min/inc/dec/and/or/xor) have a single,
order-independent final value, recomputed exactly on the CPU here (the same
formulas as the upstream sample's computeGold). The two order-dependent slots
(atomicExch, atomicCAS) have no single correct value -- whichever thread runs
last "wins" -- so they are only range-checked for being a valid thread id in
[0, len), exactly as the original NVIDIA sample does.

Metric: atomic_slot_mismatch = number of slots that fail (tolerance 0).
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V  # noqa: E402

# slot layout (matches original/main.cu testKernel):
#   0 add  1 sub  2 exch*  3 max  4 min  5 inc  6 dec  7 cas*  8 and  9 or  10 xor
#   (* = order-dependent, range-checked; the other 9 are exact)
EXACT_SLOTS = [0, 1, 3, 4, 5, 6, 8, 9, 10]
RANGE_SLOTS = [2, 7]


def _to_int32(x):
    """Interpret a 32-bit result as a signed int, matching C `int` output."""
    x &= 0xFFFFFFFF
    return x - 0x100000000 if x & 0x80000000 else x


def expected_exact(length):
    exp = {}
    exp[0] = 10 * length                                  # atomicAdd
    exp[1] = -10 * length                                 # atomicSub
    v = -(1 << 8)
    for i in range(length):
        v = v if v > i else i
    exp[3] = v                                            # atomicMax
    v = 1 << 8
    for i in range(length):
        v = v if v < i else i
    exp[4] = v                                            # atomicMin
    v = 0
    for i in range(length):
        v = 0 if v >= 17 else v + 1
    exp[5] = v                                            # atomicInc (limit 17)
    v = 0
    for i in range(length):
        v = 137 if (v == 0 or v > 137) else v - 1
    exp[6] = v                                            # atomicDec (limit 137)
    v = 0xff
    for i in range(length):
        v &= (2 * i + 7)
    exp[8] = _to_int32(v)                                 # atomicAnd
    v = 0
    for i in range(length):
        v |= (1 << (i & 31))   # shift amount masked mod 32, as in hardware
    exp[9] = _to_int32(v)                                 # atomicOr
    v = 0xff
    for i in range(length):
        v ^= i
    exp[10] = _to_int32(v)                                # atomicXor
    return exp


def _read_ints(path):
    with open(path, "r", encoding="utf-8") as fh:
        return [int(float(tok)) for tok in fh.read().split()]


def check(meta, output, selftest):
    length = int(meta["input"]["sizes"][0])
    exp = expected_exact(length)
    if selftest:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        vals = [0] * 11
        for slot, val in exp.items():
            vals[slot] = val
        for slot in RANGE_SLOTS:
            vals[slot] = length - 1  # any valid thread id in [0, len)
        with open(output, "w", encoding="utf-8") as fh:
            fh.write("\n".join(str(v) for v in vals) + "\n")

    got = _read_ints(output)
    mismatches = 11
    if len(got) == 11:
        mismatches = 0
        for slot in EXACT_SLOTS:
            if got[slot] != exp[slot]:
                mismatches += 1
        for slot in RANGE_SLOTS:
            if not (0 <= got[slot] < length):
                mismatches += 1
    tol = meta["correctness"]["tolerance"] or 0
    return mismatches <= tol, "atomic_slot_mismatch", float(mismatches), float(tol)


if __name__ == "__main__":
    V.run_custom(check)
