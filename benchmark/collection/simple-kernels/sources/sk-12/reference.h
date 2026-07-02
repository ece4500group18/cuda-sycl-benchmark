// reference.h
//
// CPU reference implementation for the streamOrderedAllocVectorAdd case
// (elementwise vector add, c[i] = a[i] + b[i]).
#ifndef STREAMORDEREDALLOCVECTORADD_REFERENCE_H
#define STREAMORDEREDALLOCVECTORADD_REFERENCE_H

// Deterministic input generators:
//   a[i] = (i % 23) - 11   (range [-11, 11])
//   b[i] = (i % 19) - 9    (range [-9, 9])
// Both are small integers exactly representable in float, so their sum
// is also exactly representable -- no rounding is involved anywhere in
// generating or checking this case's data.
inline float gen_a(long i) {
  return (float)((i % 23) - 11);
}

inline float gen_b(long i) {
  return (float)((i % 19) - 9);
}

// c[i] = a[i] + b[i], for i in [0, n). Each element is an independent
// addition, so the result is exact regardless of which allocation
// strategy (cudaMalloc/cudaFree vs. cudaMallocAsync/cudaFreeAsync) or
// which stream fed the buffers touched by the kernel.
inline void reference_vector_add(const float *a, const float *b, float *c_out, long n) {
  for (long i = 0; i < n; ++i) {
    c_out[i] = a[i] + b[i];
  }
}

#endif  // STREAMORDEREDALLOCVECTORADD_REFERENCE_H
