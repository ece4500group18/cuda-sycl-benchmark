// reference.h
//
// CPU reference implementation for the tiledMatmulShmem case
// (square matrix multiplication, C = A * B).
#ifndef TILEDMATMULSHMEM_REFERENCE_H
#define TILEDMATMULSHMEM_REFERENCE_H

// Deterministic input generators (n x n row-major matrices):
//   A[i] = ((i % 17) - 8) * 0.25
//   B[i] = ((i % 23) - 11) * 0.5
inline double gen_a(long i) {
  return (double)((i % 17) - 8) * 0.25;
}

inline double gen_b(long i) {
  return (double)((i % 23) - 11) * 0.5;
}

// C = A * B, all n x n row-major.
inline void reference_matmul(const double *A, const double *B, double *C, int n) {
  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < n; ++col) {
      double sum = 0.0;
      for (int k = 0; k < n; ++k) {
        sum += A[row * n + k] * B[k * n + col];
      }
      C[row * n + col] = sum;
    }
  }
}

#endif  // TILEDMATMULSHMEM_REFERENCE_H
