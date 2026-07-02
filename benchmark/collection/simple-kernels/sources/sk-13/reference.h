// reference.h
//
// CPU reference implementation for the systemScopeAtomicAdd case.
// Ported directly from the original sample's own verify() function
// (NVIDIA/cuda-samples cpp/0_Introduction/systemWideAtomics/
// systemWideAtomics.cu), generalized so it can check either the
// "system" array (GPU atomicKernel + host atomicKernel_CPU, combined
// via *_system atomics) or the "device" array (a single GPU kernel
// using ordinary, non-_system atomics over the same total index range)
// against the same closed-form / range oracle.
#ifndef SYSTEMSCOPEATOMICADD_REFERENCE_H
#define SYSTEMSCOPEATOMICADD_REFERENCE_H

// atom_arr layout (10 ints), matching main.cu's atomicKernel /
// atomicKernel_device / atomicKernel_CPU:
//   0: add   1: exch   2: max   3: min   4: inc
//   5: dec   6: cas    7: and   8: or    9: xor
// Initial state (set by main.cu before launch): all 0, except
// atom_arr[7] = atom_arr[9] = 0xff.
//
// `len` is the total number of *logical* contributors across both
// halves of the computation:
//   - system path:  numBlocks*numThreads GPU threads (tid in
//     [0, numBlocks*numThreads)) plus a host-side mirror loop covering
//     i in [numBlocks*numThreads, 2*numBlocks*numThreads) -- together
//     exactly [0, len).
//   - device path:  a single kernel launched over 2*numBlocks*numThreads
//     threads, tid in [0, len) directly.
// `loop_num` is the inner repeat count (LOOP_NUM in main.cu); it only
// changes the *count* seen by add/inc/dec (whose effect accumulates
// across repeats) -- max/min/exch/cas/and/or/xor apply the same
// idempotent update every repeat, so repeating them changes nothing.

// The 8 slots with a single, order-independent final value regardless
// of how contributions from different threads (or host vs. device) are
// interleaved: add, max, min, inc, dec, and, or, xor.
inline bool reference_check_exact(const int *arr, int len, int loop_num) {
  long N = (long)len * (long)loop_num;

  // atomicAdd: net effect is +10 per (thread, repeat).
  long addv = N * 10;
  if (addv != arr[0]) return false;

  // atomicMax: converges to the largest contributor index, len-1.
  int val = -(1 << 8);
  for (int i = 0; i < len; ++i) val = (val > i) ? val : i;
  if (val != arr[2]) return false;

  // atomicMin: converges to the smallest contributor index, 0.
  val = 1 << 8;
  for (int i = 0; i < len; ++i) val = (val < i) ? val : i;
  if (val != arr[3]) return false;

  // atomicInc (modulo 17+1): deterministic counter over N repeats.
  int limit = 17;
  val = 0;
  for (long i = 0; i < N; ++i) val = (val >= limit) ? 0 : val + 1;
  if (val != arr[4]) return false;

  // atomicDec: deterministic counter over N repeats.
  limit = 137;
  val = 0;
  for (long i = 0; i < N; ++i)
    val = ((val == 0) || (val > limit)) ? limit : val - 1;
  if (val != arr[5]) return false;

  // atomicAnd: bitwise AND over all contributor indices, commutative
  // and associative, order-independent.
  val = 0xff;
  for (int i = 0; i < len; ++i) val &= (2 * i + 7);
  if (val != arr[7]) return false;

  // atomicOr: bitwise OR over all contributor indices. The original
  // sample's own verify() does `val |= (1 << i)` for `i` up to
  // `len - 1`, which relies on GPU/x86 hardware masking the shift
  // amount to 5 bits (`i mod 32`) -- technically undefined behavior
  // for a 32-bit shift count `>= 32` in portable C++, even though it
  // is exactly what both the CUDA `atomicOr`/`atomicOr_system`
  // instruction and an x86 `shl` instruction actually do. This is made
  // explicit here (`1 << (i & 31)`) so the reference is well-defined
  // C++ while computing the identical value (same fix used in this
  // repo's atomicIntrinsics/reference.h).
  val = 0;
  for (int i = 0; i < len; ++i) val |= (1 << (i & 31));
  if (val != arr[8]) return false;

  // atomicXor: bitwise XOR over all contributor indices, order-independent.
  val = 0xff;
  for (int i = 0; i < len; ++i) val ^= i;
  if (val != arr[9]) return false;

  return true;
}

// atomicExch (slot 1) and atomicCAS (slot 6) are inherently
// execution-order-dependent: whichever contributor happens to perform
// the operation last determines the final value. The only invariant
// that holds regardless of scheduling (GPU warp order, or GPU-vs-host
// interleaving in the system path) is that the final value must be
// *some* contributor's index, i.e. a member of [0, len).
inline bool reference_check_range(const int *arr, int len) {
  bool found_exch = false, found_cas = false;
  for (int i = 0; i < len; ++i) {
    if (i == arr[1]) found_exch = true;
    if (i == arr[6]) found_cas = true;
  }
  return found_exch && found_cas;
}

#endif  // SYSTEMSCOPEATOMICADD_REFERENCE_H
