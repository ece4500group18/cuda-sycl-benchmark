// reference.h
//
// CPU reference implementation for the occupancyTunedLaunch case
// (elementwise square, array[i] *= array[i]).
#ifndef OCCUPANCYTUNEDLAUNCH_REFERENCE_H
#define OCCUPANCYTUNEDLAUNCH_REFERENCE_H

#include <cstdint>

// Deterministic input generator:
//   array[i] = i % 1000
inline uint32_t gen_array(int i) {
  return (uint32_t)(i % 1000);
}

// out[i] = gen_array(i) * gen_array(i), for i in [0, arrayCount). Each
// element is an independent, order-independent integer multiply (values
// are at most 999*999 = 998001, so this never overflows uint32_t; even if
// it did, unsigned overflow is well-defined modulo 2^32 and would still
// match the GPU kernel's identical arithmetic), so the result is exact
// regardless of which thread/block processes which index or how many
// threads-per-block the launch configuration uses.
inline void reference_square(uint32_t *out, int arrayCount) {
  for (int i = 0; i < arrayCount; ++i) {
    uint32_t v = gen_array(i);
    out[i] = v * v;
  }
}

#endif  // OCCUPANCYTUNEDLAUNCH_REFERENCE_H
