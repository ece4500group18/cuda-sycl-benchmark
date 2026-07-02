// reference.h
//
// CPU reference implementation for the fp16PackedScalarProduct case
// (packed half2 dot product, native half2 operators vs. explicit
// __hfma2/__hadd2 intrinsics).
#ifndef FP16PACKEDSCALARPRODUCT_REFERENCE_H
#define FP16PACKEDSCALARPRODUCT_REFERENCE_H

#include <cstddef>

// Deterministic input generators, used both to build the half2 input
// arrays in main.cu and by the CPU reference below:
//   a[i].x = a[i].y = i % 4
//   b[i].x = b[i].y = i % 2
// Both are small non-negative integers, exactly representable in fp16.
inline double gen_a_lane(size_t i) { return (double)(i % 4); }
inline double gen_b_lane(size_t i) { return (double)(i % 2); }

// CPU reference for the per-block partial dot product, computed with
// *exactly* the same block/thread decomposition and grid-stride
// iteration order as scalarProductKernel_native / _intrinsics in
// main.cu: block `b`'s partial is the sum, over its `numThreads`
// threads, of each thread's grid-stride multiply-accumulate over both
// half2 lanes.
//
// Because every value involved is a small, exactly-representable fp16
// integer, and (as shown in main.cu's header comment) no partial sum
// anywhere in the per-thread accumulation or the block's shared-memory
// tree reduction ever exceeds fp16's exact-integer range of
// [-2048, 2048], this plain double-precision computation reproduces
// the GPU's fp16 result bit-for-bit: no floating-point tolerance is
// needed to compare against it.
inline void reference_block_partials(double *out, size_t size, int numBlocks, int numThreads) {
  int stride = numBlocks * numThreads;
  for (int b = 0; b < numBlocks; ++b) {
    double block_sum = 0.0;
    for (int t = 0; t < numThreads; ++t) {
      double thread_sum = 0.0;
      for (size_t i = (size_t)(t + numThreads * b); i < size; i += (size_t)stride) {
        double av = gen_a_lane(i);
        double bv = gen_b_lane(i);
        thread_sum += av * bv;  // x lane: a[i].x * b[i].x
        thread_sum += av * bv;  // y lane: a[i].y * b[i].y (identical formula)
      }
      block_sum += thread_sum;
    }
    out[b] = block_sum;
  }
}

#endif  // FP16PACKEDSCALARPRODUCT_REFERENCE_H
