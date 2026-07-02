// reference.h
//
// CPU reference implementation for the chunkedStreamPipelineIncrement case
// (per-element increment, out[i] = in[i] + 1).
#ifndef CHUNKEDSTREAMPIPELINEINCREMENT_REFERENCE_H
#define CHUNKEDSTREAMPIPELINEINCREMENT_REFERENCE_H

// Deterministic input generator: in[i] = i.
//
// Upstream's own test harness zero-initializes the whole input buffer
// (`h_data_source[i] = 0`), which cannot distinguish "processed the wrong
// chunk" from "processed the right chunk" (0 + 1 == 1 either way). Using
// an index-dependent value here means any chunk-offset or stream-buffer
// mixup produces a detectable, non-zero error.
inline int gen_in(long i) {
  return (int)i;
}

// out[i] = in[i] + 1, for i in [0, n). Each element is an independent
// update (matches incKernel's `g_out[idx] = g_in[idx] + 1` body with
// inner_reps == 1), so the result is exact regardless of how the index
// range [0, n) is split into chunks, staged into pinned buffers, or
// scheduled across CUDA streams.
inline void reference_inc(const int *in, int *out, long n) {
  for (long i = 0; i < n; ++i) {
    out[i] = in[i] + 1;
  }
}

#endif  // CHUNKEDSTREAMPIPELINEINCREMENT_REFERENCE_H
