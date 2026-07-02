// reference.h
//
// CPU reference implementation for the shflScanWarpPrefixSum case.
//
// The GPU kernel (shfl_scan_test, reproduced verbatim in main.cu from
// NVIDIA/cuda-samples' shfl_scan sample) computes, independently for each
// thread block, an inclusive prefix sum ("scan") of that block's own
// contiguous chunk of the input array using warp-shuffle instructions
// (plus a shared-memory cross-warp broadcast when a block spans more
// than one warp). It is launched here with `partial_sums = NULL`, so
// there is no carry propagated between blocks -- each block's segment
// is scanned independently. This is exactly reproduced on the CPU as a
// "segmented" inclusive prefix sum with a fixed segment length equal to
// the block size used for that launch (32 for the single-warp
// configuration, 256 for the multi-warp configuration).
//
// Since all values are non-negative 32-bit integers and integer addition
// is associative/commutative, the CPU's simple left-to-right running sum
// within each segment produces bit-for-bit the same result as the GPU's
// hierarchical (intra-warp shuffle, then inter-warp shared-memory
// broadcast) accumulation order, regardless of how the additions are
// grouped -- so an exact match (max_abs_error == 0) is the correct
// oracle, not a tolerance.
#ifndef SHFLSCANWARPPREFIXSUM_REFERENCE_H
#define SHFLSCANWARPPREFIXSUM_REFERENCE_H

// Deterministic input generator: in[i] = (i % 9) + 1, i.e. values in
// [1, 9] cycling with period 9 -- small enough that even a 256-element
// segment sum (max 256 * 9 = 2304) stays far below INT_MAX, so there is
// no risk of overflow-order-dependence either.
inline int gen_in(long i) {
  return (int)(i % 9) + 1;
}

// Segmented inclusive prefix sum: for each contiguous, non-overlapping
// segment of length `seg` in `in[0, n)`, write the running (inclusive)
// sum of that segment into `out` at the same indices. `n` must be a
// multiple of `seg` (true for both configurations used by main.cu).
inline void reference_segmented_scan(const int *in, int *out, long n, long seg) {
  for (long base = 0; base < n; base += seg) {
    int running = 0;
    for (long j = 0; j < seg; ++j) {
      running += in[base + j];
      out[base + j] = running;
    }
  }
}

#endif  // SHFLSCANWARPPREFIXSUM_REFERENCE_H
