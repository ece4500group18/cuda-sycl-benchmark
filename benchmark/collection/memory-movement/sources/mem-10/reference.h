// reference.h
//
// CPU reference implementation for the memAlign case (DAXPY).
//
// Shared by the CUDA and SYCL variants so that both can be checked
// against exactly the same expected output.
//
// y[i] += a * x[i]   for i in [1, n)   (note: index 0 is intentionally
//                                       left untouched, matching the
//                                       original CUDAMicroBench kernel)
#ifndef MEMALIGN_REFERENCE_H
#define MEMALIGN_REFERENCE_H

#include <cstddef>

// Deterministic input generator shared by host and reference code.
// x[i] = ((i % 17) - 8) * 0.25
// y[i] = ((i % 23) - 11) * 0.5
// (same family of formulas used elsewhere in this repo's easy/medium cases)
inline double gen_x(long i) {
  return (double)((i % 17) - 8) * 0.25;
}

inline double gen_y(long i) {
  return (double)((i % 23) - 11) * 0.5;
}

// Computes the reference y array (in place) for y[i] += a * x[i], i in [1,n).
inline void reference_axpy(double *x, double *y, long n, double a) {
  for (long i = 1; i < n; ++i) {
    y[i] += a * x[i];
  }
}

#endif  // MEMALIGN_REFERENCE_H
