// reference.h
//
// CPU reference implementation for the dynamicSharedMinReduction case.
//
// Both device kernels compute the minimum of the same N = 2*NUM_THREADS
// input floats (every block reads from the same input buffer starting at
// offset 0 -- there is no per-block indexing into `input`, exactly as in
// the upstream sample -- so every block is expected to produce the same
// minimum value).
#ifndef DYNAMICSHAREDMINREDUCTION_REFERENCE_H
#define DYNAMICSHAREDMINREDUCTION_REFERENCE_H

// Deterministic input generator: input[i] = ((i % 37) - 18) * 0.5f
// Range: i in [0, 37) maps to values in [-9.0f, +9.0f] in steps of 0.5f,
// repeating every 37 indices.
inline float gen_input(long i) {
  return (float)((i % 37) - 18) * 0.5f;
}

// Reference minimum over input[0 .. n) computed with a straightforward
// sequential scan. Since `min` is commutative/associative and every
// candidate value is an exact float (no rounding involved in computing
// gen_input's small integer * 0.5f products), the result is bit-exact
// regardless of the order in which elements are compared -- a tree
// reduction, a linear scan, or any other traversal order all must land
// on the exact same minimum float value.
inline float reference_min(long n) {
  float m = gen_input(0);
  for (long i = 1; i < n; ++i) {
    float v = gen_input(i);
    if (v < m) m = v;
  }
  return m;
}

#endif  // DYNAMICSHAREDMINREDUCTION_REFERENCE_H
