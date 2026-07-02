// reference.h
//
// CPU reference implementation for the voteAnyAll case (per-warp
// any/all of a 0/1 predicate).
#ifndef VOTEANYALL_REFERENCE_H
#define VOTEANYALL_REFERENCE_H

// Deterministic test pattern, adapted from the original sample's
// genVoteTestPattern (NVIDIA/cuda-samples simpleVoteIntrinsics.cu):
// four groups of `size/4` lanes each (one warp per group, for
// warp_size=32, size=128):
//   group 0: all lanes 0                       (Any=false, All=false)
//   group 1: odd lane index nonzero, even 0     (Any=true,  All=false)
//   group 2: even lane index nonzero, odd 0     (Any=true,  All=false)
//   group 3: all lanes 0xffffffff               (Any=true,  All=true)
inline void gen_vote_pattern(unsigned int *input, int size) {
  for (int i = 0; i < size / 4; ++i) {
    input[i] = 0x00000000u;
  }
  for (int i = 2 * size / 8; i < 4 * size / 8; ++i) {
    input[i] = (i & 0x01) ? (unsigned int)i : 0u;
  }
  for (int i = 2 * size / 4; i < 3 * size / 4; ++i) {
    input[i] = (i & 0x01) ? 0u : (unsigned int)i;
  }
  for (int i = 3 * size / 4; i < size; ++i) {
    input[i] = 0xffffffffu;
  }
}

// Per-warp any()/all() of (input[lane] != 0), broadcast to every lane
// in that warp -- exactly what __any_sync/__all_sync compute in
// hardware, computed independently on the host for verification.
inline void reference_vote(const unsigned int *input, unsigned int *ref_any,
                            unsigned int *ref_all, int size, int warp_size) {
  for (int base = 0; base < size; base += warp_size) {
    int any_true = 0;
    int all_true = 1;
    for (int lane = 0; lane < warp_size; ++lane) {
      int pred = (input[base + lane] != 0);
      any_true |= pred;
      all_true &= pred;
    }
    for (int lane = 0; lane < warp_size; ++lane) {
      ref_any[base + lane] = any_true ? 1u : 0u;
      ref_all[base + lane] = all_true ? 1u : 0u;
    }
  }
}

#endif  // VOTEANYALL_REFERENCE_H
