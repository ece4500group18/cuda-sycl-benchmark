// reference.h
//
// CPU reference implementation for the sharedScanVsShuffleScan case
// (256-element unsigned-int exclusive prefix sum / scan).
#ifndef SHAREDSCANVSSHUFFLESCAN_REFERENCE_H
#define SHAREDSCANVSSHUFFLESCAN_REFERENCE_H

typedef unsigned int uint;

// Deterministic input generator: in[i] = (i % 7) + 1, i in [0, n).
// Values are in [1, 7], so the running sum over 256 elements stays well
// under 2^16, let alone the 2^32 wraparound point of `uint` -- no
// overflow, so unsigned-integer addition is exact and its result does
// not depend on the order pairs are summed in (commutative, associative,
// no overflow).
inline uint gen_in(uint i) {
  return (i % 7) + 1;
}

// Exclusive prefix sum: out[i] = sum(in[0..i-1]), out[0] = 0. A plain
// sequential left-to-right accumulation; since addition here can't
// overflow (see gen_in's comment), this is an exact reference for any
// reduction-tree shape a GPU scan kernel might use (shared-memory
// Hillis-Steele doubling, warp-shuffle, etc.) -- they all must produce
// the same sum of the same set of addends.
inline void reference_exclusive_scan(const uint *in, uint *out, uint n) {
  uint running = 0;
  for (uint i = 0; i < n; ++i) {
    out[i] = running;
    running += in[i];
  }
}

#endif  // SHAREDSCANVSSHUFFLESCAN_REFERENCE_H
