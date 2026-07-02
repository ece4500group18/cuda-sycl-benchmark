// reference.h
//
// CPU reference implementation for the zeroCopyMappedVectorAdd case
// (elementwise vector add, c[i] = a[i] + b[i], single precision).
#ifndef ZEROCOPYMAPPEDVECTORADD_REFERENCE_H
#define ZEROCOPYMAPPEDVECTORADD_REFERENCE_H

// Deterministic input generators:
//   a[i] = (i % 23) - 11
//   b[i] = (i % 19) - 9
inline float gen_a(int i) {
  return (float)((i % 23) - 11);
}

inline float gen_b(int i) {
  return (float)((i % 19) - 9);
}

// c[i] = gen_a(i) + gen_b(i), for i in [0, n). Each element is an
// independent add with no accumulation across elements, so the result is
// exact (bit-identical) regardless of which thread computes it, in what
// order, or via which host<->device memory-movement mechanism the
// operands arrived.
inline void reference_vector_add(float *c_out, int n) {
  for (int i = 0; i < n; ++i) {
    c_out[i] = gen_a(i) + gen_b(i);
  }
}

#endif  // ZEROCOPYMAPPEDVECTORADD_REFERENCE_H
