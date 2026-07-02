// reference.h
//
// CPU reference implementation for the bankConflictReduction case.
//
// Both CUDA kernels (sum_cudakernel: sequential addressing,
// sum_cudakernel_bc: interleaved addressing) compute the same
// per-block sum reduction; this header provides a CPU equivalent
// for verifying either kernel's output.
#ifndef BANKCONFLICTREDUCTION_REFERENCE_H
#define BANKCONFLICTREDUCTION_REFERENCE_H

// Deterministic input generator: x[i] = ((i % 17) - 8) * 0.25f
inline float gen_x(long i) {
  return (float)((i % 17) - 8) * 0.25f;
}

// Reference per-block sum reduction.
// x has n elements, grouped into blocks of `threads_per_block`.
// result[b] = sum of x[b*threads_per_block .. b*threads_per_block + threads_per_block - 1]
inline void reference_block_sum(const float *x, float *result, int n,
                                 int threads_per_block) {
  int num_blocks = (n + threads_per_block - 1) / threads_per_block;
  for (int b = 0; b < num_blocks; ++b) {
    float s = 0.0f;
    for (int t = 0; t < threads_per_block; ++t) {
      int idx = b * threads_per_block + t;
      if (idx < n) s += x[idx];
    }
    result[b] = s;
  }
}

#endif  // BANKCONFLICTREDUCTION_REFERENCE_H
