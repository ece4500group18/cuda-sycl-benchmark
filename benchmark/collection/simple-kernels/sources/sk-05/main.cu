// cgThreadBlockRankSum: cooperative_groups::this_thread_block() partition +
// a shared-memory tree reduction of each thread's rank within that block.
//
// The __device__ reduction helper below is reproduced, unmodified, from:
//   NVIDIA/cuda-samples, cpp/0_Introduction/simpleCooperativeGroups/simpleCooperativeGroups.cu
//   Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//   SPDX-License-Identifier: BSD-3-Clause
//
// `sumReduction(thread_group g, int *x, int val)` is the sample's generic
// reduction: at each step it halves the number of "active" lanes and has
// lane `lane < i` add its partner's value (`x[lane + i]`) into its own,
// synchronizing the group (`g.sync()`) both before and after the partial
// sum is visible in shared memory. This is the "simple but not trivial"
// idiom this case isolates: a whole-thread-block cooperative_groups
// partition (`this_thread_block()`), used purely for its `.sync()` /
// `.thread_rank()` / `.size()` API (a drop-in typed replacement for raw
// `__syncthreads()` + `threadIdx.x` arithmetic), driving a shared-memory
// tree reduction whose correctness depends on every thread in the block
// reaching each `g.sync()` call -- a non-trivial synchronization/control-flow
// pattern distinct from the independent per-element kernels elsewhere in
// this benchmark set.
//
// `cgkernel` itself is *adapted* from the upstream sample: the original
// kernel additionally creates 16-thread `tiled_partition<16>` sub-groups of
// the block and reduces each of those too (a warp-level CG partition, a
// distinct idiom covered by a separate case in this pick set), and reports
// both results only via `printf`, self-checking against the analytical
// formula `(n-1)*n/2` inline in the kernel with no host-visible output
// buffer. Here `cgkernel` keeps only the whole-thread-block partition and
// its `sumReduction` call, and writes each block's reduced sum to a
// `g_odata[blockIdx.x]` global array element instead of printing it, so the
// result can be read back on the host and diffed against a CPU reference --
// the reduction algorithm itself (the `sumReduction` body, and how
// `cgkernel` calls it for the block-wide group) is byte-identical to
// upstream.
//
// Deterministic setup (replicated by reference.h):
//   blockDim.x = 256 (threadsPerBlock, a power of two so g.size()/2 halves
//   evenly at every step of sumReduction's loop), gridDim.x = 16 blocks.
//   Each block's input is simply its threads' ranks 0..255, so the exact
//   expected sum is the closed-form triangular number
//   blockDim*(blockDim-1)/2 = 32640, identical for every block, computed in
//   plain 64-bit integer arithmetic (no floating-point, no order
//   sensitivity).
//
// Output: argv[1] (default output/cuda_output.txt), gridDim.x = 16 lines,
// one integer per line: block i's reduced sum of ranks 0..255.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "reference.h"

using namespace cooperative_groups;

#define CHECK(call)                                                          \
  do {                                                                       \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), \
              __FILE__, __LINE__);                                          \
      return 2;                                                             \
    }                                                                        \
  } while (0)

// --- begin: verbatim from NVIDIA/cuda-samples simpleCooperativeGroups.cu ---

/**
 * CUDA device function
 *
 * calculates the sum of val across the group g. The workspace array, x,
 * must be large enough to contain g.size() integers.
 */
__device__ int sumReduction(thread_group g, int *x, int val)
{
    // rank of this thread in the group
    int lane = g.thread_rank();

    // for each iteration of this loop, the number of threads active in the
    // reduction, i, is halved, and each active thread (with index [lane])
    // performs a single summation of it's own value with that
    // of a "partner" (with index [lane+i]).
    for (int i = g.size() / 2; i > 0; i /= 2) {
        // store value for this thread in temporary array
        x[lane] = val;

        // synchronize all threads in group
        g.sync();

        if (lane < i)
            // active threads perform summation of their value with
            // their partner's value
            val += x[lane + i];

        // synchronize all threads in group
        g.sync();
    }

    // master thread in group returns result, and others return -1.
    if (g.thread_rank() == 0)
        return val;
    else
        return -1;
}

// --- end: verbatim from NVIDIA/cuda-samples ---

// --- begin: adapted from NVIDIA/cuda-samples simpleCooperativeGroups.cu's
// cgkernel() -- same this_thread_block() partition + sumReduction() call,
// but writes the result to g_odata[blockIdx.x] instead of printf, and omits
// the original's additional tiled_partition<16> reduction (a warp-level CG
// idiom covered separately). ---

__global__ void cgkernel(int *g_odata)
{
    // threadBlockGroup includes all threads in the block
    thread_block threadBlockGroup = this_thread_block();

    // workspace array in shared memory required for reduction
    extern __shared__ int workspace[];

    // input to reduction, for each thread, is its' rank in the group
    int input = threadBlockGroup.thread_rank();

    // perform reduction
    int output = sumReduction(threadBlockGroup, workspace, input);

    // master thread in group writes out the block's result
    if (threadBlockGroup.thread_rank() == 0) {
        g_odata[blockIdx.x] = output;
    }
}

// --- end: adapted from NVIDIA/cuda-samples ---

int main(int argc, char **argv) {
  const int threadsPerBlock = 256;  // power of two: sumReduction halves evenly
  const int blocksPerGrid   = 16;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  const size_t memSize = (size_t)blocksPerGrid * sizeof(int);

  int *hOData = (int *)malloc(memSize);

  int *dOData;
  CHECK(cudaMalloc(&dOData, memSize));

  cgkernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(dOData);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  CHECK(cudaMemcpy(hOData, dOData, memSize, cudaMemcpyDeviceToHost));

  // CPU reference check: every block reduces ranks 0..threadsPerBlock-1, so
  // every block's expected sum is the same closed-form triangular number.
  long long expected = reference_block_rank_sum(threadsPerBlock);
  int exact_ok = 1;
  for (int b = 0; b < blocksPerGrid; ++b) {
    if ((long long)hOData[b] != expected) exact_ok = 0;
  }

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (int b = 0; b < blocksPerGrid; ++b) fprintf(f, "%d\n", hOData[b]);
  fclose(f);

  printf("cgThreadBlockRankSum done: blocksPerGrid=%d, threadsPerBlock=%d, "
         "expected_sum=%lld, exact_ok=%d -> %s\n",
         blocksPerGrid, threadsPerBlock, expected, exact_ok, out);
  printf("%s\n", exact_ok ? "PASS" : "FAIL");

  cudaFree(dOData);
  free(hOData);
  return 0;
}
