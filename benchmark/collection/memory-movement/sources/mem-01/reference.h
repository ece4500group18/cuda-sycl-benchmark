// reference.h
//
// CPU reference implementation for the asyncCopySingleStage case
// (square matrix multiplication, C = A * B, single precision).
#ifndef ASYNCCOPYSINGLESTAGE_REFERENCE_H
#define ASYNCCOPYSINGLESTAGE_REFERENCE_H

// Deterministic input generators (n x n row-major matrices):
//   A[i] = ((i % 17) - 8) * 0.25
//   B[i] = ((i % 23) - 11) * 0.5
inline float gen_a(long i) {
  return (float)((i % 17) - 8) * 0.25f;
}

inline float gen_b(long i) {
  return (float)((i % 23) - 11) * 0.5f;
}

// C = A * B, all n x n row-major. Accumulates in double for a stable
// reference regardless of the single-precision GPU accumulation order.
inline void reference_matmul(const float *A, const float *B, float *C, int n) {
  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < n; ++col) {
      double sum = 0.0;
      for (int k = 0; k < n; ++k) {
        sum += (double)A[row * n + k] * (double)B[k * n + col];
      }
      C[row * n + col] = (float)sum;
    }
  }
}

#endif  // ASYNCCOPYSINGLESTAGE_REFERENCE_H
