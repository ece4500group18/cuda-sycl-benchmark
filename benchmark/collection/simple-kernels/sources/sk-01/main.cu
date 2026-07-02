// atomicIntrinsics: a battery of global-memory atomic read-modify-write
// operations, all racing on the same handful of output slots.
//
// The __global__ kernel below is reproduced, unmodified, from:
//   NVIDIA/cuda-samples, cpp/0_Introduction/simpleAtomicIntrinsics/simpleAtomicIntrinsics_kernel.cuh
//   Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//   SPDX-License-Identifier: BSD-3-Clause
//
// Every one of `numThreads * numBlocks` threads calls all eleven atomic
// intrinsics (add, sub, exch, max, min, inc, dec, cas, and, or, xor) on
// the same 11-element output array, with no coordination between threads
// beyond the atomics themselves. This is "simple but not trivial": each
// individual atomic call is a one-line, well-documented primitive, but
// reasoning about what the *final* value in each slot must be after
// thousands of threads race on it -- and recognizing which operations
// have a single well-defined outcome (add/sub/max/min/inc/dec/and/or/xor,
// all associative or otherwise order-independent) vs. which are
// inherently execution-order-dependent (exch, cas: whichever thread
// happens to run last "wins") -- is the non-trivial part.
//
// Deterministic setup (replicated by reference.h, following the
// original sample's own computeGold reference exactly):
//   numThreads = 256, numBlocks = 64 (len = 16384 threads total)
//   initial output values: all 0, except slot 8 (AND) and slot 10 (XOR)
//   initialized to 0xff so those two tests produce a non-trivial result.
//
// Output: argv[1] (default output/cuda_output.txt), 11 lines, one final
// integer value per output slot.

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

// --- begin: verbatim from NVIDIA/cuda-samples simpleAtomicIntrinsics_kernel.cuh ---

__global__ void testKernel(int *g_odata)
{
    // access thread id
    const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // Test various atomic instructions

    // Arithmetic atomic instructions

    // Atomic addition
    atomicAdd(&g_odata[0], 10);

    // Atomic subtraction (final should be 0)
    atomicSub(&g_odata[1], 10);

    // Atomic exchange
    atomicExch(&g_odata[2], tid);

    // Atomic maximum
    atomicMax(&g_odata[3], tid);

    // Atomic minimum
    atomicMin(&g_odata[4], tid);

    // Atomic increment (modulo 17+1)
    atomicInc((unsigned int *)&g_odata[5], 17);

    // Atomic decrement
    atomicDec((unsigned int *)&g_odata[6], 137);

    // Atomic compare-and-swap
    atomicCAS(&g_odata[7], tid - 1, tid);

    // Bitwise atomic instructions

    // Atomic AND
    atomicAnd(&g_odata[8], 2 * tid + 7);

    // Atomic OR
    atomicOr(&g_odata[9], 1 << tid);

    // Atomic XOR
    atomicXor(&g_odata[10], tid);
}

// --- end: verbatim from NVIDIA/cuda-samples ---

int main(int argc, char **argv) {
  const unsigned int numThreads = 256;
  const unsigned int numBlocks = 64;
  const unsigned int numData = 11;
  const size_t memSize = sizeof(int) * numData;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";

  int *hOData = (int *)malloc(memSize);
  for (unsigned int i = 0; i < numData; ++i) hOData[i] = 0;
  // To make the AND and XOR tests generate something other than 0.
  hOData[8] = hOData[10] = 0xff;

  int *dOData;
  CHECK(cudaMalloc(&dOData, memSize));
  CHECK(cudaMemcpy(dOData, hOData, memSize, cudaMemcpyHostToDevice));

  testKernel<<<numBlocks, numThreads>>>(dOData);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  CHECK(cudaMemcpy(hOData, dOData, memSize, cudaMemcpyDeviceToHost));

  // CPU reference check (mirrors the original sample's own computeGold).
  int exact_ok = reference_check_exact(hOData, (int)(numThreads * numBlocks));
  int range_ok = reference_check_range(hOData, (int)(numThreads * numBlocks));

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (unsigned int i = 0; i < numData; ++i) fprintf(f, "%d\n", hOData[i]);
  fclose(f);

  printf("atomicIntrinsics done: len=%u, exact_ok=%d, range_ok=%d -> %s\n",
         numThreads * numBlocks, exact_ok, range_ok, out);
  printf("%s\n", (exact_ok && range_ok) ? "PASS" : "FAIL");

  cudaFree(dOData);
  free(hOData);
  return 0;
}
