#!/usr/bin/env python3
"""systemScopeAtomicAdd: CPU reference for the 10 atomic slots of the *system*
array, mirroring the upstream sample's own verify() generalized to len/LOOP_NUM.

Eight slots have a single order-independent value (add/max/min/inc/dec/and/or/xor)
and are matched exactly; atomicExch (slot 1) and atomicCAS (slot 6) are
execution-order dependent and only range-checked to a valid contributor index
in [0, len).

Slot layout: 0 add, 1 exch*, 2 max, 3 min, 4 inc, 5 dec, 6 cas*, 7 and, 8 or, 9 xor.
sizes = [len, LOOP_NUM] = [32768, 50]. Metric: atomic_slot_mismatch (tol 0).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "tools"))
import verify_lib as V  # noqa: E402

EXACT_SLOTS = [0, 2, 3, 4, 5, 7, 8, 9]
RANGE_SLOTS = [1, 6]


def _to_int32(x):
    x &= 0xFFFFFFFF
    return x - 0x100000000 if x & 0x80000000 else x


def _sawtooth(steps, transition):
    """Simulate a periodic recurrence from state 0 for `steps` steps, using the
    period so we never loop more than 2*period times."""
    seen = {}
    v = 0
    seq = [0]
    while v not in seen:
        seen[v] = len(seq) - 1
        v = transition(v)
        seq.append(v)
        if len(seq) > 100000:
            break
    # seq[k] = state after k steps until it first repeats a state
    # find period: state seq[len(seq)-1] equals some earlier seq index
    repeat_state = seq[-1]
    start = seen[repeat_state]
    period = (len(seq) - 1) - start
    if steps < len(seq):
        return seq[steps]
    # fast-forward whole periods
    rem = start + (steps - start) % period
    return seq[rem]


def expected_exact(length, loop):
    N = length * loop
    exp = {}
    exp[0] = 10 * N                                   # atomicAdd
    exp[2] = length - 1                               # atomicMax
    exp[3] = 0                                        # atomicMin
    exp[4] = _sawtooth(N, lambda v: 0 if v >= 17 else v + 1)          # atomicInc
    exp[5] = _sawtooth(N, lambda v: 137 if (v == 0 or v > 137) else v - 1)  # atomicDec
    i = np.arange(length, dtype=np.int64)
    exp[7] = _to_int32(int(np.bitwise_and.reduce((2 * i + 7))) & 0xff)       # atomicAnd (init 0xff)
    or_val = 0
    for k in range(length):
        or_val |= (1 << (k & 31))
    exp[8] = _to_int32(or_val)                        # atomicOr
    exp[9] = _to_int32(int(np.bitwise_xor.reduce(i)) ^ 0xff)                 # atomicXor (init 0xff)
    return exp


def _read_ints(path):
    with open(path, "r", encoding="utf-8") as fh:
        return [int(float(tok)) for tok in fh.read().split()]


def check(meta, output, selftest):
    length, loop = (int(x) for x in meta["input"]["sizes"][:2])
    exp = expected_exact(length, loop)
    if selftest:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        vals = [0] * 10
        for slot, val in exp.items():
            vals[slot] = val
        for slot in RANGE_SLOTS:
            vals[slot] = length - 1  # any valid contributor id in [0, len)
        with open(output, "w", encoding="utf-8") as fh:
            fh.write("\n".join(str(v) for v in vals) + "\n")

    got = _read_ints(output)
    mism = 10
    if len(got) == 10:
        mism = 0
        for slot in EXACT_SLOTS:
            if got[slot] != exp[slot]:
                mism += 1
        for slot in RANGE_SLOTS:
            if not (0 <= got[slot] < length):
                mism += 1
    tol = meta["correctness"]["tolerance"] or 0
    return mism <= tol, "atomic_slot_mismatch", float(mism), float(tol)


if __name__ == "__main__":
    V.run_custom(check)
