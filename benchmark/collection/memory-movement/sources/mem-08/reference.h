// reference.h
//
// CPU reference implementation for the histogramSharedPrivatization case
// (64-bin byte histogram).
#ifndef HISTOGRAMSHAREDPRIVATIZATION_REFERENCE_H
#define HISTOGRAMSHAREDPRIVATIZATION_REFERENCE_H

#include <cstdint>

// Deterministic input generator: byte i of the input buffer is
//   data[i] = (i * 31 + 7) % 256
inline unsigned char gen_byte(long i) {
  return (unsigned char)((i * 31L + 7L) % 256L);
}

// 64-bin histogram: each byte contributes to bin (byte >> 2) & 0x3F, i.e.
// its top 6 bits -- exactly the bin selection done on-device by addWord()
// in histogram64Kernel (see main.cu) and by the naive per-byte atomicAdd
// kernel. Every byte increments exactly one bin by exactly 1; integer
// counting is commutative and associative, so summing in any order (any
// thread/block interleaving, any partial-histogram merge order) produces
// the exact same 64 counts as this single sequential CPU pass.
inline void reference_histogram64(const unsigned char *data, long byteCount,
                                   unsigned int hist[64]) {
  for (int b = 0; b < 64; ++b) hist[b] = 0;
  for (long i = 0; i < byteCount; ++i) {
    unsigned int bin = ((unsigned int)data[i] >> 2) & 0x3FU;
    hist[bin]++;
  }
}

#endif  // HISTOGRAMSHAREDPRIVATIZATION_REFERENCE_H
