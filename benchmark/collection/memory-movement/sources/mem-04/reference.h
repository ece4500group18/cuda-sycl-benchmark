// reference.h
//
// CPU reference implementation for the coalescedAxpyDistribution case
// (AXPY, y[i] += a*x[i]).
#ifndef COALESCEDAXPYDISTRIBUTION_REFERENCE_H
#define COALESCEDAXPYDISTRIBUTION_REFERENCE_H

// Deterministic input generators:
//   x[i] = ((i % 17) - 8) * 0.25
//   y[i] = ((i % 23) - 11) * 0.5
inline double gen_x(long i) {
  return (double)((i % 17) - 8) * 0.25;
}

inline double gen_y(long i) {
  return (double)((i % 23) - 11) * 0.5;
}

// y[i] = gen_y(i) + a * gen_x(i), for i in [0, n). Each element is an
// independent multiply-add, so the result is exact regardless of the
// order in which indices are processed.
inline void reference_axpy(const double *x, double *y_out, long n, double a) {
  for (long i = 0; i < n; ++i) {
    y_out[i] = gen_y(i) + a * x[i];
  }
}

#endif  // COALESCEDAXPYDISTRIBUTION_REFERENCE_H
