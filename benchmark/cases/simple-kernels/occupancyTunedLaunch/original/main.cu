// occupancyTunedLaunch: cudaOccupancyMaxPotentialBlockSize-driven automatic
// launch-configuration vs. a naive hand-picked fixed block size, both launching
// the identical elementwise-square kernel.
//
// The __global__ kernel square below is reproduced, unmodified, from:
//   NVIDIA/cuda-samples, cpp/0_Introduction/simpleOccupancy/simpleOccupancy.cu
//   Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//   SPDX-License-Identifier: BSD-3-Clause
//
// Deterministic input (reproduced by tests/verify.py):
//   array[i] = i % 1000,  arrayCount = 1 << 20 = 1048576.
// Every index is squared by exactly one thread regardless of launch config, so
// the "manual" and "automatic" launches produce identical output.
//
// Output: argv[1] (default output/cuda_output.txt), one uint32 per line, the
// array after the occupancy-tuned ("automatic") launch's square.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>

#define CHECK(call)                                                          \
  do {                                                                       \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), \
              __FILE__, __LINE__);                                           \
      return 2;                                                              \
    }                                                                        \
  } while (0)

// --- begin: verbatim from NVIDIA/cuda-samples simpleOccupancy.cu ---

__global__ void square(uint32_t *array, int arrayCount)
{
    extern __shared__ int dynamicSmem[];
    int                   idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < arrayCount) {
        array[idx] *= array[idx];
    }
}

// --- end: verbatim from NVIDIA/cuda-samples ---

// Deterministic input generator (new host code; mirrored in tests/verify.py).
static inline uint32_t gen_array(int i) { return (uint32_t)(i % 1000); }

int main(int argc, char **argv) {
  const int arrayCount = 1 << 20;         // 1,048,576
  const int manualBlockSize = 32;         // upstream's own "too small" manualBlockSize
  const char *out = (argc > 1) ? argv[1] : "output/cuda_output.txt";
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
  int autoBlockSize = 0;
  int autoMinGridSize = 0;
  CHECK(cudaOccupancyMaxPotentialBlockSize(&autoMinGridSize, &autoBlockSize,
                                            (void *)square, 0, arrayCount));
  int autoGridSize = (arrayCount + autoBlockSize - 1) / autoBlockSize;
  square<<<autoGridSize, autoBlockSize>>>(dArrayAuto, arrayCount);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  uint32_t *hArrayAuto = (uint32_t *)malloc(bytes);
  CHECK(cudaMemcpy(hArrayAuto, dArrayAuto, bytes, cudaMemcpyDeviceToHost));

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (int i = 0; i < arrayCount; ++i) fprintf(f, "%u\n", hArrayAuto[i]);
  fclose(f);

  cudaFree(dArrayManual);
  cudaFree(dArrayAuto);
  free(hArray);
  free(hArrayAuto);
  return 0;
}
