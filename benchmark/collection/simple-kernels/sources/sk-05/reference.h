// reference.h
//
// CPU reference implementation for the cgThreadBlockRankSum case.
#ifndef CGTHREADBLOCKRANKSUM_REFERENCE_H
#define CGTHREADBLOCKRANKSUM_REFERENCE_H

// Closed-form sum of ranks 0..blockDim-1 within one cooperative_groups
// thread_block partition -- exactly the analytical formula the upstream
// sample itself uses to self-check ("(n-1)(n)/2", noting indexing starts
// at 0), evaluated here in plain 64-bit integer arithmetic. Since every
// block's input is simply its own threads' ranks 0..blockDim-1, every
// block has this same expected sum, with no floating-point involvement
// and no accumulation-order sensitivity.
inline long long reference_block_rank_sum(int blockDim) {
  return (long long)(blockDim - 1) * (long long)blockDim / 2;
}

#endif  // CGTHREADBLOCKRANKSUM_REFERENCE_H
