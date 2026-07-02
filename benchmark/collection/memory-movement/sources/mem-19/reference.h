// reference.h
//
// CPU reference implementation for the unifiedMemoryAccess case
// (strided gather, y[j] = a*x[j*stride]).
#ifndef UNIFIEDMEMORYACCESS_REFERENCE_H
#define UNIFIEDMEMORYACCESS_REFERENCE_H

// Deterministic input generator:
//   x[i] = ((i % 17) - 8) * 0.25
inline float gen_x(long i) {
  return (float)((i % 17) - 8) * 0.25f;
}

// y[j] = a * x[j*stride], for j in [0, n/stride). Single-precision
// multiply, computed in float to match the GPU kernel's precision
// exactly (no accumulation order to worry about, so the result is
// bit-exact regardless of how the input array was provisioned).
inline void reference_strided_axpy(const float *x, float *y_out, long n, float a, int stride) {
  long n_out = n / stride;
  for (long j = 0; j < n_out; ++j) {
    y_out[j] = a * x[j * stride];
  }
}

#endif  // UNIFIEDMEMORYACCESS_REFERENCE_H
