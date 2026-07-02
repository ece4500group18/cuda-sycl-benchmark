// reference.h
//
// CPU reference implementation for the bitonicVsOddEvenMergeSort case.
#ifndef BITONICVSODDEVENMERGESORT_REFERENCE_H
#define BITONICVSODDEVENMERGESORT_REFERENCE_H

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

// Deterministic key/value generators (index-formula based, no rand()):
//
//   key[i] = (i * 40503u) mod 65536     (uint32_t arithmetic, wraps)
//   val[i] = i
//
// 40503 is odd, so gcd(40503, 65536) == 1 (65536 == 2^16). Because the
// multiplier is therefore invertible modulo 65536, the map
// i -> (i * 40503u) % 65536u is a *bijection* on Z/65536Z and hence
// injective on any subset of it -- in particular on i in
// [0, arrayLength) for arrayLength = 1024. All 1024 generated keys are
// guaranteed pairwise distinct, so the sorted order of (key, val) pairs
// is unambiguous for any correct comparison sort: there is no
// tie-breaking ambiguity to paper over, and a plain position-by-position
// array comparison against a CPU std::sort is a valid *exact* oracle.
inline uint32_t gen_key(int i) {
  return ((uint32_t)i * 40503u) % 65536u;
}

inline uint32_t gen_val(int i) { return (uint32_t)i; }

// CPU reference: ascending sort of (key, val) pairs by key.
inline void reference_sort(const uint32_t *key_in, const uint32_t *val_in,
                            int n, uint32_t *key_out, uint32_t *val_out) {
  std::vector<std::pair<uint32_t, uint32_t>> v(n);
  for (int i = 0; i < n; ++i) v[i] = std::make_pair(key_in[i], val_in[i]);
  std::sort(v.begin(), v.end(),
            [](const std::pair<uint32_t, uint32_t> &a,
               const std::pair<uint32_t, uint32_t> &b) {
              return a.first < b.first;
            });
  for (int i = 0; i < n; ++i) {
    key_out[i] = v[i].first;
    val_out[i] = v[i].second;
  }
}

#endif  // BITONICVSODDEVENMERGESORT_REFERENCE_H
