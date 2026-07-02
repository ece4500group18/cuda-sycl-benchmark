// reference.h
//
// CPU reference implementation for the warpAggregatedAtomicCompaction case
// (stream compaction: keep every src[i] > 0, pack them into a contiguous
// array).
#ifndef WARPAGGREGATEDATOMICCOMPACTION_REFERENCE_H
#define WARPAGGREGATEDATOMICCOMPACTION_REFERENCE_H

#include <algorithm>

// Deterministic input generator:
//   src[i] = (i % 13) - 6, for i in [0, n)
// 13-periodic, values in [-6, 6]; exactly 6 of every 13 values (1..6) are
// > 0, so the expected compacted count is deterministic and computable
// without running the GPU kernels.
inline int gen_src(long i) {
  return (int)(i % 13) - 6;
}

// CPU reference: filter src[i] > 0, i in [0, n), into out_sorted[] in
// ascending order. Returns the number of elements kept.
//
// Both GPU kernels (atomicAggInc-based filter_arr and the naive
// filter_arr_naive) visit every index in [0, n) exactly once and are
// guaranteed to keep exactly the same *set* (multiset) of values -- but
// block/warp scheduling order on the GPU is unspecified, so the *position*
// each kept value lands at in the compacted output is not reproducible
// run-to-run or between the two kernels. The correctness oracle therefore
// sorts ascending before comparing (see main.cu), turning this into a
// well-defined multiset-equality check rather than a positional one.
inline int reference_filter_sorted(const int *src, long n, int *out_sorted) {
  int count = 0;
  for (long i = 0; i < n; ++i) {
    if (src[i] > 0) out_sorted[count++] = src[i];
  }
  std::sort(out_sorted, out_sorted + count);
  return count;
}

#endif  // WARPAGGREGATEDATOMICCOMPACTION_REFERENCE_H
