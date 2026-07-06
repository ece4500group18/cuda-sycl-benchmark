// cgThreadBlockRankSum: cooperative_groups::this_thread_block() partition + a
// shared-memory tree reduction of each thread's rank within its block.
//
// The __device__ helper sumReduction below is reproduced, unmodified, from:
//   NVIDIA/cuda-samples, cpp/0_Introduction/simpleCooperativeGroups/simpleCooperativeGroups.cu
//   Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//   SPDX-License-Identifier: BSD-3-Clause
//
// cgkernel is adapted from the sample's cgkernel(): it keeps only the
// whole-thread-block partition and its sumReduction call, and writes each
// block's reduced sum to g_odata[blockIdx.x] instead of printf (the original
// also reduced tiled_partition<16> sub-groups and self-checked via printf).
//
// Deterministic setup: blockDim.x = 256, gridDim.x = 16. Each block reduces its
// threads' ranks 0..255, so every block's sum is the triangular number
// 255*256/2 = 32640.
//
// Output: argv[1] (default output/cuda_output.txt), 16 lines, block i's sum.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;

#define CHECK(call)                                                          \
  do {                                                                       \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), \
              __FILE__, __LINE__);                                           \
      return 2;                                                              \
    }                                                                        \
  } while (0)

// --- begin: verbatim from NVIDIA/cuda-samples simpleCooperativeGroups.cu ---

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
// cgkernel() -- same this_thread_block() partition + sumReduction() call, but
// writes to g_odata[blockIdx.x] instead of printf, and omits the original's
// additional tiled_partition<16> reduction. ---

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
  const char *out = (argc > 1) ? argv[1] : "output/cuda_output.txt";
  const size_t memSize = (size_t)blocksPerGrid * sizeof(int);

  int *hOData = (int *)malloc(memSize);

  int *dOData;
  CHECK(cudaMalloc(&dOData, memSize));

  cgkernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(dOData);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  CHECK(cudaMemcpy(hOData, dOData, memSize, cudaMemcpyDeviceToHost));

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (int b = 0; b < blocksPerGrid; ++b) fprintf(f, "%d\n", hOData[b]);
  fclose(f);

  cudaFree(dOData);
  free(hOData);
  return 0;
}
