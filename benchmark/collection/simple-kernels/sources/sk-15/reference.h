// reference.h
//
// CPU reference implementation for the threadFenceSinglePassReduction case
// (single-precision sum reduction).
//
// A sum reduction's result depends on the *order* in which partial sums are
// combined (float addition is not associative), so a naive linear-order CPU
// sum is not guaranteed to match either GPU kernel bit-for-bit. Instead,
// each of the two reduction functions below is a straight-line, single-
// threaded re-implementation of the *exact* same binary tree of additions
// that the corresponding GPU kernel performs -- same grouping, same order,
// same float32 (not double) arithmetic -- so comparing against the GPU
// output is an exact (max_abs_error == 0) check, not a tolerance check.
#ifndef THREADFENCESINGLEPASSREDUCTION_REFERENCE_H
#define THREADFENCESINGLEPASSREDUCTION_REFERENCE_H

#include <cstring>

// Deterministic input generator: input[i] = ((i % 29) - 14) * 0.25
inline float gen_input(long i) {
  return (float)((i % 29) - 14) * 0.25f;
}

// ---------------------------------------------------------------------
// reduceBlockRef: a single-threaded, bit-exact re-implementation of the
// __device__ function `reduceBlock<blockSize>()` in
// threadFenceReduction_kernel.cuh (see main.cu for the verbatim CUDA
// source). That function:
//   1. Seeds a `blockSize`-element shared array with each thread's partial
//      sum (`mySum`).
//   2. Reduces each 32-lane warp independently via a stride-halving tree
//      (16, 8, 4, 2, 1), where every lane's read of `sdata[tid+i]` in a
//      given round sees the *pre-round* state of that array (real SIMT
//      hardware performs all of a warp's reads for one instruction before
//      any of its writes retire) -- reproduced below via an explicit
//      `snap` (snapshot) of the shared array taken before each round's
//      writes.
//   3. Combines the `blockSize/32` per-warp results with a simple
//      left-to-right scalar sum (thread 0 walks `sdata[0], sdata[32], ...`).
//
// This has been verified (see scratch harness used during development) to
// be bit-identical, for float32 inputs, to a literal thread-by-thread SIMT
// simulation of the CUDA source with explicit barriers.
inline float reduceBlockRef(const float *mySum, unsigned int blockSize) {
  float sdata[256];
  for (unsigned int tid = 0; tid < blockSize; ++tid) sdata[tid] = mySum[tid];

  const unsigned int VEC = 32;
  for (unsigned int warpBase = 0; warpBase < blockSize; warpBase += VEC) {
    float beta[VEC];
    for (unsigned int lane = 0; lane < VEC; ++lane) beta[lane] = mySum[warpBase + lane];
    for (int step = VEC / 2; step > 0; step >>= 1) {
      float snap[VEC];
      for (unsigned int lane = 0; lane < VEC; ++lane) snap[lane] = sdata[warpBase + lane];
      for (int lane = 0; lane < step; ++lane) {
        beta[lane] = beta[lane] + snap[lane + step];
        sdata[warpBase + lane] = beta[lane];
      }
    }
  }

  float finalSum = 0.0f;
  for (unsigned int i = 0; i < blockSize; i += VEC) finalSum = finalSum + sdata[i];
  return finalSum;
}

// ---------------------------------------------------------------------
// Phase 1 / level 1, common to both GPU variants: each of the `B` blocks
// of `T` threads reduces its 2*T-element slice of `input` (elements
// [b*2*T, (b+1)*2*T)) to a single partial sum, exactly mirroring
// reduceBlocks<T, true>()'s first-level load (`g_idata[i] + g_idata[i +
// blockSize]`, i = b*(T*2) + tid) followed by reduceBlock<T>().
inline void reference_level1_partials(const float *input, int N, int T, int B, float *partial /* size B */) {
  float mySum[256];
  for (int b = 0; b < B; ++b) {
    int base = b * 2 * T;
    for (int tid = 0; tid < T; ++tid) {
      mySum[tid] = input[base + tid] + input[base + tid + T];
    }
    partial[b] = reduceBlockRef(mySum, (unsigned int)T);
  }
}

// reduceSinglePass<T, true>()'s Phase 2 (the last-arriving block, detected
// via __threadfence()+atomicInc()): since here B == T, each of the T
// threads picks up exactly one partial sum (`g_odata[tid]`, the while-loop
// body executes exactly once because gridDim.x == blockSize), then the
// same reduceBlock<T>() tree combines them.
inline float reference_single_pass(const float *input, int N, int T, int B) {
  float partial[256];
  reference_level1_partials(input, N, T, B, partial);
  return reduceBlockRef(partial, (unsigned int)T);
}

// The classic two-kernel-launch reduction: kernel launch #1 is the same
// level-1 reduceBlocks<T,true>() as above (producing `partial[0..B)`);
// kernel launch #2 re-invokes reduceMultiPass<T2,true>() with a single
// block of T2 = B/2 threads, each combining two partial sums
// (`g_idata[tid] + g_idata[tid+T2]`), then reduceBlock<T2>() over the
// result -- a differently-shaped tree from the single-pass kernel's Phase
// 2 (T2-wide vs. T-wide), which is exactly the point: the two techniques
// only need to be *individually* exact, not bit-identical to each other.
inline float reference_two_launch(const float *input, int N, int T, int B, int T2) {
  float partial[256];
  reference_level1_partials(input, N, T, B, partial);

  float mySum2[256];
  for (int tid = 0; tid < T2; ++tid) {
    mySum2[tid] = partial[tid] + partial[tid + T2];
  }
  return reduceBlockRef(mySum2, (unsigned int)T2);
}

#endif  // THREADFENCESINGLEPASSREDUCTION_REFERENCE_H
