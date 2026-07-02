// reference.h
//
// CPU reference implementation for the warpDivergence case.
//
// Both kernels compute, for each i:
//   if i is even: z[i] = 2*x[i] + 3*y[i]
//   if i is odd:  z[i] = 3*x[i] + 2*y[i]
#ifndef WARPDIVERGENCE_REFERENCE_H
#define WARPDIVERGENCE_REFERENCE_H

// Deterministic input generators:
//   x[i] = ((i % 17) - 8) * 0.25f
//   y[i] = ((i % 23) - 11) * 0.5f
inline float gen_x(long i) {
  return (float)((i % 17) - 8) * 0.25f;
}

inline float gen_y(long i) {
  return (float)((i % 23) - 11) * 0.5f;
}

inline void reference_z(const float *x, const float *y, float *z, int n) {
  for (int i = 0; i < n; ++i) {
    if (i % 2 == 0) {
      z[i] = 2.0f * x[i] + 3.0f * y[i];
    } else {
      z[i] = 3.0f * x[i] + 2.0f * y[i];
    }
  }
}

#endif  // WARPDIVERGENCE_REFERENCE_H
