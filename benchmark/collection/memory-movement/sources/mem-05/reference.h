// reference.h
//
// CPU reference implementation for the constMemMatrixDims case
// (elementwise 2D matrix add, C = A + B, single precision).
#ifndef CONSTMEMMATRIXDIMS_REFERENCE_H
#define CONSTMEMMATRIXDIMS_REFERENCE_H

// Deterministic input generators (M x N row-major matrices):
//   A[i] = ((i % 17) - 8) * 0.5
//   B[i] = ((i % 13) - 6) * 0.25
inline float gen_a(long i) {
  return (float)((i % 17) - 8) * 0.5f;
}

inline float gen_b(long i) {
  return (float)((i % 13) - 6) * 0.25f;
}

// C = A + B, all M x N row-major, linear index i = row*N + col. Each
// element is an independent single-precision add, so the result is exact
// regardless of which thread computes it or where the bounds (M, N) used
// to compute that thread's index came from (kernel argument vs.
// __constant__ memory).
inline void reference_add(const float *A, const float *B, float *C, int M, int N) {
  for (int i = 0; i < M * N; ++i) {
    C[i] = A[i] + B[i];
  }
}

#endif  // CONSTMEMMATRIXDIMS_REFERENCE_H
