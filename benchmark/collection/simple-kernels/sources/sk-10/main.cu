// occupancyTunedLaunch: CUDA's cudaOccupancyMaxPotentialBlockSize-driven
// automatic launch-configuration API vs. a naive, hand-picked fixed block
// size, both launching the identical elementwise-square kernel.
//
// The __global__ kernel below is reproduced, unmodified, from:
//   NVIDIA/cuda-samples, cpp/0_Introduction/simpleOccupancy/simpleOccupancy.cu
//   Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//   SPDX-License-Identifier: BSD-3-Clause
//
// "square" multiplies each array element by itself (uint32_t, so any
// overflow would wrap modulo 2^32 with well-defined C/CUDA semantics --
// though with the inputs used here it never actually overflows) and is
// launched two different ways:
//
// "manual": a fixed, hand-picked block size (32 threads/block -- the
//   original sample's own `manualBlockSize`, deliberately "too small" per
//   the original sample's own comment) with a simple round-up grid size.
//
// "automatic": `cudaOccupancyMaxPotentialBlockSize` queries the CUDA
//   runtime/driver for a block size (and minimum grid size) that
//   maximizes theoretical occupancy for *this* kernel on *this* GPU, then
//   the kernel is launched with that suggested configuration -- runtime
//   occupancy introspection, not a compile-time constant or a value
//   hand-picked by the programmer.
//
// Both configurations launch the exact same kernel, with the exact same
// `idx = threadIdx.x + blockIdx.x * blockDim.x` addressing and bounds
// check, over the exact same input array. So regardless of how many
// blocks/threads-per-block are used to cover `arrayCount`, every index is
// touched by exactly one thread doing the exact same integer multiply --
// this isolates occupancy-driven automatic launch configuration (a
// CUDA-runtime API with no direct SYCL analog) as the sole non-trivial
// technique under test, independent of the (deliberately trivial)
// per-element arithmetic.
//
// Deterministic input (replicated by reference.h):
//   array[i] = i % 1000
// arrayCount = 1048576 (1 << 20)
//
// Output: argv[1] (default output/cuda_output.txt), one uint32_t per
// line, array[] after the occupancy-tuned ("automatic") launch's square.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
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

// --- begin: verbatim from NVIDIA/cuda-samples cpp/0_Introduction/simpleOccupancy/simpleOccupancy.cu ---

__global__ void square(uint32_t *array, int arrayCount)
{
    extern __shared__ int dynamicSmem[];
    int                   idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < arrayCount) {
        array[idx] *= array[idx];
    }
}

// --- end: verbatim from NVIDIA/cuda-samples ---

int main(int argc, char **argv) {
  const int arrayCount = 1 << 20;         // 1,048,576
  const int manualBlockSize = 32;         // upstream's own "too small" manualBlockSize
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  const size_t bytes = (size_t)arrayCount * sizeof(uint32_t);

  uint32_t *hArray = (uint32_t *)malloc(bytes);
  for (int i = 0; i < arrayCount; ++i) hArray[i] = gen_array(i);

  uint32_t *dArrayManual, *dArrayAuto;
  CHECK(cudaMalloc(&dArrayManual, bytes));
  CHECK(cudaMalloc(&dArrayAuto, bytes));
  CHECK(cudaMemcpy(dArrayManual, hArray, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dArrayAuto, hArray, bytes, cudaMemcpyHostToDevice));

  // Manual: fixed, hand-picked block size, simple round-up grid size.
  int manualGridSize = (arrayCount + manualBlockSize - 1) / manualBlockSize;
  square<<<manualGridSize, manualBlockSize>>>(dArrayManual, arrayCount);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  // Automatic: cudaOccupancyMaxPotentialBlockSize-driven configuration.
  // (Passed as (void*), matching the original sample's own call site.)
  int autoBlockSize = 0;
  int autoMinGridSize = 0;
  CHECK(cudaOccupancyMaxPotentialBlockSize(&autoMinGridSize, &autoBlockSize,
                                            (void *)square, 0, arrayCount));
  int autoGridSize = (arrayCount + autoBlockSize - 1) / autoBlockSize;
  square<<<autoGridSize, autoBlockSize>>>(dArrayAuto, arrayCount);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  uint32_t *hArrayManual = (uint32_t *)malloc(bytes);
  uint32_t *hArrayAuto = (uint32_t *)malloc(bytes);
  CHECK(cudaMemcpy(hArrayManual, dArrayManual, bytes, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(hArrayAuto, dArrayAuto, bytes, cudaMemcpyDeviceToHost));

  // CPU reference check.
  uint32_t *href = (uint32_t *)malloc(bytes);
  reference_square(href, arrayCount);

  double max_abs_err_manual = 0.0, max_abs_err_auto = 0.0;
  for (int i = 0; i < arrayCount; ++i) {
    double d1 = (double)hArrayManual[i] - (double)href[i];
    double d2 = (double)hArrayAuto[i] - (double)href[i];
    if (d1 < 0) d1 = -d1;
    if (d2 < 0) d2 = -d2;
    if (d1 > max_abs_err_manual) max_abs_err_manual = d1;
    if (d2 > max_abs_err_auto) max_abs_err_auto = d2;
  }

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (int i = 0; i < arrayCount; ++i) fprintf(f, "%u\n", hArrayAuto[i]);
  fclose(f);

  printf("occupancyTunedLaunch done: arrayCount=%d, "
         "manual(blockSize=%d, gridSize=%d), "
         "automatic(blockSize=%d, gridSize=%d, minGridSize=%d), "
         "max_abs_error(manual vs ref)=%.3e, max_abs_error(auto vs ref)=%.3e -> %s\n",
         arrayCount, manualBlockSize, manualGridSize, autoBlockSize, autoGridSize,
         autoMinGridSize, max_abs_err_manual, max_abs_err_auto, out);
  printf("%s\n", (max_abs_err_manual == 0.0 && max_abs_err_auto == 0.0) ? "PASS" : "FAIL");

  cudaFree(dArrayManual);
  cudaFree(dArrayAuto);
  free(hArray);
  free(hArrayManual);
  free(hArrayAuto);
  free(href);
  return 0;
}
