// shflScanWarpPrefixSum: register-shuffle ("warp shuffle") inclusive prefix sum
// (scan), launched at two block sizes to isolate pure intra-warp shuffle
// communication from the shared-memory-assisted cross-warp broadcast the same
// kernel performs when a block spans more than one warp.
//
// The __global__ kernel shfl_scan_test below is reproduced, unmodified, from:
//   NVIDIA/cuda-samples, cpp/2_Concepts_and_Techniques/shfl_scan/shfl_scan.cu
//   Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//   SPDX-License-Identifier: BSD-3-Clause
//
// Launched with partial_sums = NULL (no cross-block carry) at blockDim.x = 32
// (1 warp; stage 2 is a no-op) and blockDim.x = 256 (8 warps; shared-memory
// cross-warp broadcast fully exercised). Both compute an inclusive scan over
// disjoint segments; only the intra-block communication pattern differs.
//
// Deterministic input (reproduced by tests/verify.py):
//   in[i] = (i % 9) + 1, n = 262144.
//
// Output: argv[1] (default output/cuda_output.txt), the 262144-element scan
// result from the 256-wide (multi-warp) configuration, one int per line.

#include <cstdio>
#include <cstdlib>
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

// --- begin: verbatim from NVIDIA/cuda-samples shfl_scan.cu ---

__global__ void shfl_scan_test(int *data, int width, int *partial_sums = NULL)
{
    extern __shared__ int sums[];
    int                   id      = ((blockIdx.x * blockDim.x) + threadIdx.x);
    int                   lane_id = id % warpSize;
    // determine a warp_id within a block
    int warp_id = threadIdx.x / warpSize;

    // Below is the basic structure of using a shfl instruction
    // for a scan.
    // Record "value" as a variable - we accumulate it along the way
    int value = data[id];

    // Now accumulate in log steps up the chain
    // compute sums, with another thread's value who is
    // distance delta away (i).  Note
    // those threads where the thread 'i' away would have
    // been out of bounds of the warp are unaffected.  This
    // creates the scan sum.

#pragma unroll
    for (int i = 1; i <= width; i *= 2) {
        unsigned int mask = 0xffffffff;
        int          n    = __shfl_up_sync(mask, value, i, width);

        if (lane_id >= i)
            value += n;
    }

    // value now holds the scan value for the individual thread
    // next sum the largest values for each warp

    // write the sum of the warp to smem
    if (threadIdx.x % warpSize == warpSize - 1) {
        sums[warp_id] = value;
    }

    __syncthreads();

    //
    // scan sum the warp sums
    // the same shfl scan operation, but performed on warp sums
    //
    if (warp_id == 0 && lane_id < (blockDim.x / warpSize)) {
        int warp_sum = sums[lane_id];

        int mask = (1 << (blockDim.x / warpSize)) - 1;
        for (int i = 1; i <= (blockDim.x / warpSize); i *= 2) {
            int n = __shfl_up_sync(mask, warp_sum, i, (blockDim.x / warpSize));

            if (lane_id >= i)
                warp_sum += n;
        }

        sums[lane_id] = warp_sum;
    }

    __syncthreads();

    // perform a uniform add across warps in the block
    // read neighbouring warp's sum and add it to threads value
    int blockSum = 0;

    if (warp_id > 0) {
        blockSum = sums[warp_id - 1];
    }

    value += blockSum;

    // Now write out our result
    data[id] = value;

    // last thread has sum, write write out the block's sum
    if (partial_sums != NULL && threadIdx.x == blockDim.x - 1) {
        partial_sums[blockIdx.x] = value;
    }
}

// --- end: verbatim from NVIDIA/cuda-samples ---

// Deterministic input generator (new host code; mirrored in tests/verify.py).
static inline int gen_in(long i) { return (int)(i % 9) + 1; }

int main(int argc, char **argv) {
  const long n = 262144L;  // 2^18, divisible by both 32 and 256
  const char *out = (argc > 1) ? argv[1] : "output/cuda_output.txt";
  const size_t bytes = (size_t)n * sizeof(int);

  int *h_in = (int *)malloc(bytes);
  int *h_warp256 = (int *)malloc(bytes);
  for (long i = 0; i < n; ++i) h_in[i] = gen_in(i);

  int *d_warp32, *d_warp256;
  CHECK(cudaMalloc(&d_warp32, bytes));
  CHECK(cudaMalloc(&d_warp256, bytes));
  CHECK(cudaMemcpy(d_warp32, h_in, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_warp256, h_in, bytes, cudaMemcpyHostToDevice));

  // Single-warp-only configuration: 1 warp/block (stage 2 a functional no-op).
  const int blockSize32 = 32;
  const int gridSize32 = (int)(n / blockSize32);
  const int shmem32 = (blockSize32 / 32) * (int)sizeof(int);
  shfl_scan_test<<<gridSize32, blockSize32, shmem32>>>(d_warp32, 32, NULL);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  // Multi-warp hybrid configuration: 8 warps/block, cross-warp broadcast.
  const int blockSize256 = 256;
  const int gridSize256 = (int)(n / blockSize256);
  const int shmem256 = (blockSize256 / 32) * (int)sizeof(int);
  shfl_scan_test<<<gridSize256, blockSize256, shmem256>>>(d_warp256, 32, NULL);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  CHECK(cudaMemcpy(h_warp256, d_warp256, bytes, cudaMemcpyDeviceToHost));

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (long i = 0; i < n; ++i) fprintf(f, "%d\n", h_warp256[i]);
  fclose(f);

  cudaFree(d_warp32);
  cudaFree(d_warp256);
  free(h_in);
  free(h_warp256);
  return 0;
}
