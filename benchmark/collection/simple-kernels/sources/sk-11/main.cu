// shflScanWarpPrefixSum: register-shuffle ("warp shuffle") inclusive
// prefix sum (scan), launched at two different block sizes to isolate
// pure intra-warp shuffle communication from the shared-memory-assisted
// cross-warp broadcast that the same kernel performs when a block spans
// more than one warp.
//
// The __global__ kernel below (shfl_scan_test) is reproduced, unmodified,
// from:
//   NVIDIA/cuda-samples, cpp/2_Concepts_and_Techniques/shfl_scan/shfl_scan.cu
//   Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//   SPDX-License-Identifier: BSD-3-Clause
//
// NOTE on kernel naming: the upstream file (verified by fetching it in
// full, along with shfl_integral_image.cuh, from the URL above) contains
// exactly one scan kernel, `shfl_scan_test`, plus a small carry-add
// kernel `uniform_add` used only to propagate sums *between* blocks
// across a second kernel launch. There is no separate `shfl2_scan_test`
// kernel anywhere in the file (nor in any tagged release back to CUDA
// 5.5, nor in shfl_integral_image.cuh, all checked directly). Rather
// than invent a kernel that doesn't exist upstream, this case uses the
// one real kernel that does exist, launched at two different block
// sizes -- which is exactly the "single-warp" vs. "multi-warp hybrid"
// contrast the kernel's own body implements generically:
//
// "shfl_scan_test" performs an inclusive scan of `data[]` in two stages:
//   1. An intra-warp scan via `__shfl_up_sync` in a `log2(width)`-step
//      loop -- pure register-to-register communication, no shared memory.
//   2. If the block spans more than one warp (`blockDim.x / warpSize >
//      1`), each warp's total is written to shared memory (`sums[]`),
//      warp 0 performs the same shuffle-scan *on those warp totals*, and
//      every thread reads its warp's prefix from shared memory and adds
//      it uniformly (`blockSum`) -- a shared-memory-assisted cross-warp
//      broadcast on top of the warp-shuffle primitive.
//
// This case launches the identical kernel (`partial_sums = NULL`, so
// each block's segment is scanned independently with no cross-block
// carry) at:
//   - blockDim.x = 32  (1 warp/block): stage 2's `blockDim.x/warpSize`
//     is 1, so `warp_id` is always 0 and `blockSum` is always 0 for
//     every thread -- stage 2 is a functional no-op and the result is a
//     pure single-warp, shared-memory-free shuffle scan of 32 elements.
//   - blockDim.x = 256 (8 warps/block): stage 2 is fully exercised,
//     broadcasting each warp's running total to the next warp via
//     shared memory, producing a 256-element inclusive scan per block.
//
// Both configurations compute the exact same operation (inclusive sum)
// over disjoint, deterministic input segments; only the intra-block
// warp-shuffle-communication pattern (and whether shared memory is
// touched at all) differs -- isolating how the register-shuffle
// mechanism scales from a single warp to a full multi-warp block.
//
// Deterministic input (replicated by reference.h):
//   in[i] = (i % 9) + 1, for i in [0, n), n = 262144
//
// Output: argv[1] (default output/cuda_output.txt), the 262,144-element
// scan result from the 256-wide (multi-warp) configuration, one int per
// line.

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

// --- begin: verbatim from NVIDIA/cuda-samples shfl_scan.cu ---

// Scan using shfl - takes log2(n) steps
// This function demonstrates basic use of the shuffle intrinsic, __shfl_up,
// to perform a scan operation across a block.
// First, it performs a scan (prefix sum in this case) inside a warp
// Then to continue the scan operation across the block,
// each warp's sum is placed into shared memory.  A single warp
// then performs a shuffle scan on that shared memory.  The results
// are then uniformly added to each warp's threads.
// This pyramid type approach is continued by placing each block's
// final sum in global memory and prefix summing that via another kernel call,
// then uniformly adding across the input data via the uniform_add<<<>>> kernel.

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

int main(int argc, char **argv) {
  const long n = 262144L;  // 2^18, divisible by both 32 and 256
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  const size_t bytes = (size_t)n * sizeof(int);

  int *h_in = (int *)malloc(bytes);
  int *h_warp32 = (int *)malloc(bytes);
  int *h_warp256 = (int *)malloc(bytes);
  for (long i = 0; i < n; ++i) {
    h_in[i] = gen_in(i);
  }

  int *d_warp32, *d_warp256;
  CHECK(cudaMalloc(&d_warp32, bytes));
  CHECK(cudaMalloc(&d_warp256, bytes));
  CHECK(cudaMemcpy(d_warp32, h_in, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_warp256, h_in, bytes, cudaMemcpyHostToDevice));

  // Single-warp-only configuration: 1 warp/block, shared-mem cross-warp
  // stage is a functional no-op (warp_id is always 0, blockSum always 0).
  const int blockSize32 = 32;
  const int gridSize32 = (int)(n / blockSize32);
  const int shmem32 = (blockSize32 / 32) * (int)sizeof(int);
  shfl_scan_test<<<gridSize32, blockSize32, shmem32>>>(d_warp32, 32, NULL);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  // Multi-warp hybrid configuration: 8 warps/block, exercises the
  // shared-memory-assisted cross-warp partial-sum broadcast.
  const int blockSize256 = 256;
  const int gridSize256 = (int)(n / blockSize256);
  const int shmem256 = (blockSize256 / 32) * (int)sizeof(int);
  shfl_scan_test<<<gridSize256, blockSize256, shmem256>>>(d_warp256, 32, NULL);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  CHECK(cudaMemcpy(h_warp32, d_warp32, bytes, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(h_warp256, d_warp256, bytes, cudaMemcpyDeviceToHost));

  // CPU reference check: segmented inclusive scan, segment length ==
  // block size used for each configuration.
  int *ref32 = (int *)malloc(bytes);
  int *ref256 = (int *)malloc(bytes);
  reference_segmented_scan(h_in, ref32, n, blockSize32);
  reference_segmented_scan(h_in, ref256, n, blockSize256);

  long max_abs_err_32 = 0, max_abs_err_256 = 0;
  for (long i = 0; i < n; ++i) {
    long d32 = (long)h_warp32[i] - (long)ref32[i];
    long d256 = (long)h_warp256[i] - (long)ref256[i];
    if (d32 < 0) d32 = -d32;
    if (d256 < 0) d256 = -d256;
    if (d32 > max_abs_err_32) max_abs_err_32 = d32;
    if (d256 > max_abs_err_256) max_abs_err_256 = d256;
  }

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (long i = 0; i < n; ++i) fprintf(f, "%d\n", h_warp256[i]);
  fclose(f);

  printf("shflScanWarpPrefixSum done: n=%ld, "
         "max_abs_error(warp32 vs ref)=%ld, max_abs_error(warp256 vs ref)=%ld -> %s\n",
         n, max_abs_err_32, max_abs_err_256, out);
  printf("%s\n", (max_abs_err_32 == 0 && max_abs_err_256 == 0) ? "PASS" : "FAIL");

  cudaFree(d_warp32);
  cudaFree(d_warp256);
  free(h_in);
  free(h_warp32);
  free(h_warp256);
  free(ref32);
  free(ref256);
  return 0;
}
