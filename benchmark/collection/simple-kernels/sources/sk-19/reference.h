// reference.h
//
// CPU reference implementation for the warpShuffleReduction case.
//
// Both kernels compute a per-block sum reduction over n input floats:
//   result[b] = sum over the elements assigned to block b.
//
// reduce2 launches n/blockSize blocks, one input element per thread
// (block b owns x[b*blockSize .. b*blockSize+blockSize-1]).
//
// reduce4 launches n/(2*blockSize) blocks, two input elements per
// thread (block b owns x[b*2*blockSize .. b*2*blockSize+2*blockSize-1]).
#ifndef WARPSHUFFLEREDUCTION_REFERENCE_H
#define WARPSHUFFLEREDUCTION_REFERENCE_H

// Deterministic input generator: x[i] = ((i % 17) - 8) * 0.25f
inline float gen_x(long i) {
  return (float)((i % 17) - 8) * 0.25f;
}

// Reference per-block sum reduction over contiguous chunks of `chunk`
// elements each (chunk = blockSize for reduce2, chunk = 2*blockSize
// for reduce4).
inline void reference_block_sum(const float *x, float *result, int n, int chunk) {
  int num_blocks = n / chunk;  // n is an exact multiple of chunk in this case
  for (int b = 0; b < num_blocks; ++b) {
    float s = 0.0f;
    for (int t = 0; t < chunk; ++t) {
      s += x[b * chunk + t];
    }
    result[b] = s;
  }
}

#endif  // WARPSHUFFLEREDUCTION_REFERENCE_H
