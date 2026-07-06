// threadFenceSinglePassReduction: a single-kernel-launch sum reduction using
// __threadfence() + a global atomic "ticket" counter (the last block to finish
// detects that and combines all partials itself) vs. the classic
// two-kernel-launch tree reduction for the same sum.
//
// The kernels/device functions below are reproduced, unmodified, from:
//   NVIDIA/cuda-samples,
//   cpp/2_Concepts_and_Techniques/threadFenceReduction/threadFenceReduction_kernel.cuh
//   Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//   SPDX-License-Identifier: BSD-3-Clause
//
// Deterministic input (reproduced by tests/verify.py): input[i]=((i%29)-14)*0.25,
// N = 131072 = 2*256*256. Every value is a multiple of 0.25 with bounded partial
// sums, so every float32 addition is exact regardless of grouping/order and both
// reduction trees yield the identical exact total.
//
// Output: argv[1] (default output/cuda_output.txt), two lines: the single-pass
// kernel's final sum, then the two-kernel-launch reduction's final sum.

#include <cooperative_groups.h>
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

namespace cg = cooperative_groups;

// --- begin: verbatim from NVIDIA/cuda-samples threadFenceReduction_kernel.cuh ---

template <unsigned int blockSize>
__device__ void reduceBlock(volatile float *sdata, float mySum, const unsigned int tid, cg::thread_block cta)
{
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);
    sdata[tid]                       = mySum;
    cg::sync(tile32);

    const int VEC = 32;
    const int vid = tid & (VEC - 1);

    float beta = mySum;
    float temp;

    for (int i = VEC / 2; i > 0; i >>= 1) {
        if (vid < i) {
            temp = sdata[tid + i];
            beta += temp;
            sdata[tid] = beta;
        }
        cg::sync(tile32);
    }
    cg::sync(cta);

    if (cta.thread_rank() == 0) {
        beta = 0;
        for (int i = 0; i < blockDim.x; i += VEC) {
            beta += sdata[i];
        }
        sdata[0] = beta;
    }
    cg::sync(cta);
}

template <unsigned int blockSize, bool nIsPow2>
__device__ void reduceBlocks(const float *g_idata, float *g_odata, unsigned int n, cg::thread_block cta)
{
    extern __shared__ float sdata[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid      = threadIdx.x;
    unsigned int i        = blockIdx.x * (blockSize * 2) + threadIdx.x;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    float        mySum    = 0;

    while (i < n) {
        mySum += g_idata[i];

        if (nIsPow2 || i + blockSize < n)
            mySum += g_idata[i + blockSize];

        i += gridSize;
    }

    // do reduction in shared mem
    reduceBlock<blockSize>(sdata, mySum, tid, cta);

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize, bool nIsPow2>
__global__ void reduceMultiPass(const float *g_idata, float *g_odata, unsigned int n)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    reduceBlocks<blockSize, nIsPow2>(g_idata, g_odata, n, cta);
}

// Global variable used by reduceSinglePass to count how many blocks have finished
__device__ unsigned int retirementCount = 0;

cudaError_t setRetirementCount(int retCnt)
{
    return cudaMemcpyToSymbol(retirementCount, &retCnt, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
}

template <unsigned int blockSize, bool nIsPow2>
__global__ void reduceSinglePass(const float *g_idata, float *g_odata, unsigned int n)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    //
    // PHASE 1: Process all inputs assigned to this block
    //

    reduceBlocks<blockSize, nIsPow2>(g_idata, g_odata, n, cta);

    //
    // PHASE 2: Last block finished will process all partial sums
    //

    if (gridDim.x > 1) {
        const unsigned int      tid = threadIdx.x;
        __shared__ bool         amLast;
        extern float __shared__ smem[];

        // wait until all outstanding memory instructions in this thread are finished
        __threadfence();

        // Thread 0 takes a ticket
        if (tid == 0) {
            unsigned int ticket = atomicInc(&retirementCount, gridDim.x);
            // If the ticket ID is equal to the number of blocks, we are the last block!
            amLast = (ticket == gridDim.x - 1);
        }

        cg::sync(cta);

        // The last block sums the results of all other blocks
        if (amLast) {
            int   i     = tid;
            float mySum = 0;

            while (i < gridDim.x) {
                mySum += g_odata[i];
                i += blockSize;
            }

            reduceBlock<blockSize>(smem, mySum, tid, cta);

            if (tid == 0) {
                g_odata[0] = smem[0];

                // reset retirement count so that next run succeeds
                retirementCount = 0;
            }
        }
    }
}

// --- end: verbatim from NVIDIA/cuda-samples ---

// Deterministic input generator (new host code; mirrored in tests/verify.py).
static inline float gen_input(long i) { return (float)((i % 29) - 14) * 0.25f; }

int main(int argc, char **argv) {
  const int N  = 131072;  // = 2 * 256 * 256, a power of 2
  const int T  = 256;     // threads per block, level 1 / Phase 1 (both variants)
  const int B  = 256;     // blocks launched for level 1 / Phase 1 (both variants)
  const int T2 = 128;     // threads for the two-launch variant's level-2 kernel
  const char *out = (argc > 1) ? argv[1] : "output/cuda_output.txt";
  const size_t bytes_in = (size_t)N * sizeof(float);
  const size_t bytes_partial = (size_t)B * sizeof(float);

  float *h_input = (float *)malloc(bytes_in);
  for (int i = 0; i < N; ++i) h_input[i] = gen_input(i);

  float *d_input, *d_partial_sp, *d_partial_mp;
  CHECK(cudaMalloc(&d_input, bytes_in));
  CHECK(cudaMalloc(&d_partial_sp, bytes_partial));
  CHECK(cudaMalloc(&d_partial_mp, bytes_partial));
  CHECK(cudaMemcpy(d_input, h_input, bytes_in, cudaMemcpyHostToDevice));

  // --- Variant 1: single-pass, __threadfence() + atomic-ticket last-block-finishes ---
  CHECK(setRetirementCount(0));
  size_t smem_sp = (size_t)T * sizeof(float);
  reduceSinglePass<256, true><<<B, T, smem_sp>>>(d_input, d_partial_sp, (unsigned int)N);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  float h_single_pass_result = 0.0f;
  CHECK(cudaMemcpy(&h_single_pass_result, d_partial_sp, sizeof(float), cudaMemcpyDeviceToHost));

  // --- Variant 2: classic two-kernel-launch tree reduction ---
  size_t smem_l1 = (size_t)T * sizeof(float);
  reduceMultiPass<256, true><<<B, T, smem_l1>>>(d_input, d_partial_mp, (unsigned int)N);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  size_t smem_l2 = (size_t)T2 * sizeof(float);
  reduceMultiPass<128, true><<<1, T2, smem_l2>>>(d_partial_mp, d_partial_mp, (unsigned int)B);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  float h_two_launch_result = 0.0f;
  CHECK(cudaMemcpy(&h_two_launch_result, d_partial_mp, sizeof(float), cudaMemcpyDeviceToHost));

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  fprintf(f, "%.9g\n", h_single_pass_result);
  fprintf(f, "%.9g\n", h_two_launch_result);
  fclose(f);

  cudaFree(d_input);
  cudaFree(d_partial_sp);
  cudaFree(d_partial_mp);
  free(h_input);
  return 0;
}
