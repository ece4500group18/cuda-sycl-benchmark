// reference.h
//
// CPU reference implementation for the atomicIntrinsics case.
// Ported directly from the original sample's own computeGold()
// (NVIDIA/cuda-samples cpp/0_Introduction/simpleAtomicIntrinsics/
// simpleAtomicIntrinsics_cpu.cpp), split into:
//   - reference_check_exact(): the 9 slots with a single, order-independent
//     final value (add, sub, max, min, inc, dec, and, or, xor).
//   - reference_check_range(): the 2 slots that are inherently
//     execution-order-dependent (exch, cas) -- only checked for being a
//     valid thread id in [0, len), exactly as the original sample does.
#ifndef ATOMICINTRINSICS_REFERENCE_H
#define ATOMICINTRINSICS_REFERENCE_H

// gpuData layout (11 ints), matching main.cu's testKernel:
//   0: atomicAdd     1: atomicSub     2: atomicExch    3: atomicMax
//   4: atomicMin     5: atomicInc     6: atomicDec     7: atomicCAS
//   8: atomicAnd     9: atomicOr     10: atomicXor
// Initial state (set by main.cu before launch): all 0, except
// gpuData[8] = gpuData[10] = 0xff.

inline int reference_check_exact(const int *gpuData, int len) {
  int val;

  val = 0;
  for (int i = 0; i < len; ++i) val += 10;
  if (val != gpuData[0]) return 0;  // atomicAdd

  val = 0;
  for (int i = 0; i < len; ++i) val -= 10;
  if (val != gpuData[1]) return 0;  // atomicSub

  val = -(1 << 8);
  for (int i = 0; i < len; ++i) val = (val > i) ? val : i;
  if (val != gpuData[3]) return 0;  // atomicMax

  val = 1 << 8;
  for (int i = 0; i < len; ++i) val = (val < i) ? val : i;
  if (val != gpuData[4]) return 0;  // atomicMin

  int limit = 17;
  val = 0;
  for (int i = 0; i < len; ++i) val = (val >= limit) ? 0 : val + 1;
  if (val != gpuData[5]) return 0;  // atomicInc

  limit = 137;
  val = 0;
  for (int i = 0; i < len; ++i) val = ((val == 0) || (val > limit)) ? limit : val - 1;
  if (val != gpuData[6]) return 0;  // atomicDec

  val = 0xff;
  for (int i = 0; i < len; ++i) val &= (2 * i + 7);
  if (val != gpuData[8]) return 0;  // atomicAnd

  val = 0;
  for (int i = 0; i < len; ++i) val |= (1 << (i & 31));  // shift amount is masked mod 32 in hardware
  if (val != gpuData[9]) return 0;  // atomicOr

  val = 0xff;
  for (int i = 0; i < len; ++i) val ^= i;
  if (val != gpuData[10]) return 0;  // atomicXor

  return 1;
}

// atomicExch (slot 2) and atomicCAS (slot 7) are inherently
// execution-order-dependent: whichever of the `len` threads happens to
// perform the operation last determines the final value. The only
// invariant that holds regardless of scheduling is that the final value
// must be *some* thread's id, i.e. a member of [0, len).
inline int reference_check_range(const int *gpuData, int len) {
  int found_exch = 0, found_cas = 0;
  for (int i = 0; i < len; ++i) {
    if (i == gpuData[2]) found_exch = 1;
    if (i == gpuData[7]) found_cas = 1;
  }
  return found_exch && found_cas;
}

#endif  // ATOMICINTRINSICS_REFERENCE_H
