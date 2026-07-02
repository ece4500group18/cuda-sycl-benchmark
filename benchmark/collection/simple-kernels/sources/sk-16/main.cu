// voteAnyAll: per-warp "does any/all lane satisfy a predicate" using the
// hardware vote intrinsics __any_sync / __all_sync.
//
// The two __global__ kernels below are reproduced, unmodified, from:
//   NVIDIA/cuda-samples, cpp/0_Introduction/simpleVoteIntrinsics/simpleVote_kernel.cuh
//   Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//   SPDX-License-Identifier: BSD-3-Clause
//
// Each kernel evaluates, per 32-lane warp, whether the input predicate
// (`input[tx] != 0`) holds for any lane ("VoteAnyKernel1") or for all
// lanes ("VoteAllKernel2"), and broadcasts the same boolean result to
// every lane in that warp -- a single hardware instruction replacing what
// would otherwise be a shared-memory reduction + __syncthreads() (as used
// by warpShuffleReduction's tree reduction, or a naive loop over the warp).
//
// This is a "simple but not trivial" kernel: the operation itself
// ("did any/all of these 32 booleans come out true") is a one-line
// reduction, but doing it *within a single warp instruction*, with no
// shared memory and no explicit synchronization, is the non-trivial,
// widely-used GPU technique it demonstrates.
//
// Deterministic input (replicated by reference.h, following the same
// deterministic test pattern as the original sample's genVoteTestPattern):
//   128 threads = 4 warps, launched as a single block.
//   - warp 0: all lanes predicate-false
//   - warp 1: predicate true for odd lane indices only (mixed)
//   - warp 2: predicate true for even lane indices only (mixed)
//   - warp 3: all lanes predicate-true
//
// Output: argv[1] (default output/cuda_output.txt), 128 lines, one 0/1
// per line: 1 if VoteAnyKernel1's result for that lane is nonzero, else 0.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "reference.h"

#define CHECK(call)                                                          \
  do {                                                                       \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), \
              __FILE__, __LINE__);                                          \
      return 2;                                                             \
    }                                                                        \
  } while (0)

// --- begin: verbatim from NVIDIA/cuda-samples simpleVote_kernel.cuh ---

// Kernel #1 tests the across-the-warp vote(any) intrinsic.
// If ANY one of the threads (within the warp) of the predicated condition
// returns a non-zero value, then all threads within this warp will return a
// non-zero value
__global__ void VoteAnyKernel1(unsigned int *input, unsigned int *result, int size)
{
    int tx = threadIdx.x;

    int mask   = 0xffffffff;
    result[tx] = __any_sync(mask, input[tx]);
}

// Kernel #2 tests the across-the-warp vote(all) intrinsic.
// If ALL of the threads (within the warp) of the predicated condition returns
// a non-zero value, then all threads within this warp will return a non-zero
// value
__global__ void VoteAllKernel2(unsigned int *input, unsigned int *result, int size)
{
    int tx = threadIdx.x;

    int mask   = 0xffffffff;
    result[tx] = __all_sync(mask, input[tx]);
}

// --- end: verbatim from NVIDIA/cuda-samples ---

int main(int argc, char **argv) {
  const int warp_size = 32;
  const int vote_data_group = 4;
  const int size = vote_data_group * warp_size;  // 128
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";

  unsigned int *h_input = (unsigned int *)malloc(size * sizeof(unsigned int));
  unsigned int *h_any = (unsigned int *)malloc(size * sizeof(unsigned int));
  unsigned int *h_all = (unsigned int *)malloc(size * sizeof(unsigned int));

  gen_vote_pattern(h_input, size);

  unsigned int *d_input, *d_any, *d_all;
  CHECK(cudaMalloc(&d_input, size * sizeof(unsigned int)));
  CHECK(cudaMalloc(&d_any, size * sizeof(unsigned int)));
  CHECK(cudaMalloc(&d_all, size * sizeof(unsigned int)));

  CHECK(cudaMemcpy(d_input, h_input, size * sizeof(unsigned int), cudaMemcpyHostToDevice));

  VoteAnyKernel1<<<1, size>>>(d_input, d_any, size);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  VoteAllKernel2<<<1, size>>>(d_input, d_all, size);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  CHECK(cudaMemcpy(h_any, d_any, size * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(h_all, d_all, size * sizeof(unsigned int), cudaMemcpyDeviceToHost));

  // CPU reference check: exact boolean-truth-value match, per warp.
  unsigned int ref_any[128], ref_all[128];
  reference_vote(h_input, ref_any, ref_all, size, warp_size);

  int mismatches = 0;
  for (int i = 0; i < size; ++i) {
    int gpu_any_true = (h_any[i] != 0);
    int gpu_all_true = (h_all[i] != 0);
    int ref_any_true = (ref_any[i] != 0);
    int ref_all_true = (ref_all[i] != 0);
    if (gpu_any_true != ref_any_true) mismatches++;
    if (gpu_all_true != ref_all_true) mismatches++;
  }

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (int i = 0; i < size; ++i) fprintf(f, "%d\n", h_any[i] != 0 ? 1 : 0);
  fclose(f);

  printf("voteAnyAll done: size=%d, mismatches=%d -> %s\n", size, mismatches, out);
  printf("%s\n", (mismatches == 0) ? "PASS" : "FAIL");

  cudaFree(d_input);
  cudaFree(d_any);
  cudaFree(d_all);
  free(h_input);
  free(h_any);
  free(h_all);
  return 0;
}
