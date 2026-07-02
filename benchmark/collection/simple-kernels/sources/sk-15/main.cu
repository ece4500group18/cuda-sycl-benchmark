// threadFenceSinglePassReduction: a single-kernel-launch sum reduction that
// uses __threadfence() + a global atomic "ticket" counter so the very last
// thread block to finish can detect that fact and immediately finish the
// job itself, vs. the classic two-kernel-launch tree reduction for the
// same sum (one launch per level of the block-count tree).
//
// The kernels/device functions below are reproduced, unmodified, from:
//   NVIDIA/cuda-samples, cpp/2_Concepts_and_Techniques/threadFenceReduction/threadFenceReduction_kernel.cuh
//   Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//   SPDX-License-Identifier: BSD-3-Clause
//
// "reduceSinglePass<blockSize, nIsPow2>": Phase 1 -- every block reduces
//   its slice of the input to one partial sum via reduceBlocks()/
//   reduceBlock() (shared-memory warp-tree + linear combine), exactly as
//   reduceMultiPass() does below. Phase 2 -- __threadfence() (waits until
//   this block's g_odata write is visible to *all* other blocks, not just
//   within this block), then thread 0 does `atomicInc(&retirementCount,
//   gridDim.x)`: whichever block's ticket equals `gridDim.x - 1` is
//   provably the *last* block to finish (every other block must already
//   have taken a smaller ticket), so that one block alone sums all the
//   partial results and writes the final answer -- all within the same
//   kernel launch.
//
// "reduceMultiPass<blockSize, nIsPow2>", called twice by this file's host
//   code (once over the original input producing B partial sums, once
//   more over those B partial sums producing 1 final sum): the "classic"
//   multi-kernel-launch tree reduction the sample's own comment
//   contrasts reduceSinglePass against -- inter-block communication is
//   achieved purely by the CUDA runtime's guarantee that a kernel launch
//   does not begin until the previous one has completed, instead of a
//   fence + atomic ticket.
//
// Neither kernel depends on cooperative_groups' grid-wide sync
// (`cg::this_grid()` / `grid.sync()`, which requires
// cudaLaunchCooperativeKernel): the `cg::thread_block`/
// `cg::thread_block_tile<32>` handles used below are ordinary block- and
// warp-level sync wrappers (`__syncthreads()` / `__syncwarp()`
// equivalents), so both kernels are launched with plain `<<<...>>>`.
//
// Deterministic input (replicated by reference.h): input[i] = ((i % 29) -
// 14) * 0.25, N = 131072 (= 2 * 256 * 256, a power of 2). Both variants
// launch 256 blocks of 256 threads for level 1 / Phase 1; the two-launch
// variant's second kernel launch uses a single block of 128 threads (256 /
// 2) to combine the 256 partial sums; the single-pass variant's Phase 2
// combines those same 256 partial sums with the *same* 256-thread block
// that did Phase 1 (no new launch). Because float addition is not
// associative, these two differently-shaped combine trees are compared
// against two correspondingly differently-shaped (but individually exact)
// CPU references -- see reference.h.
//
// Output: argv[1] (default output/cuda_output.txt), two lines: the
// single-pass kernel's final sum, then the two-kernel-launch reduction's
// final sum.

#include <cooperative_groups.h>
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

namespace cg = cooperative_groups;

// --- begin: verbatim from NVIDIA/cuda-samples threadFenceReduction_kernel.cuh ---

/*
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - uses n/2 threads
    - only works for power-of-2 arrays

    This version adds multiple elements per thread sequentially.  This reduces
   the overall cost of the algorithm while keeping the work complexity O(n) and
   the step complexity O(log n). (Brent's Theorem optimization)

    See the CUDA SDK "reduction" sample for more information.
*/

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

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n) {
        mySum += g_idata[i];

        // ensure we don't read out of bounds -- this is optimized away for powerOf2
        // sized arrays
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

// Global variable used by reduceSinglePass to count how many blocks have
// finished
__device__ unsigned int retirementCount = 0;

cudaError_t setRetirementCount(int retCnt)
{
    return cudaMemcpyToSymbol(retirementCount, &retCnt, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
}

// This reduction kernel reduces an arbitrary size array in a single kernel
// invocation It does so by keeping track of how many blocks have finished.
// After each thread block completes the reduction of its own block of data, it
// "takes a ticket" by atomically incrementing a global counter.  If the ticket
// value is equal to the number of thread blocks, then the block holding the
// ticket knows that it is the last block to finish.  This last block is
// responsible for summing the results of all the other blocks.
//
// In order for this to work, we must be sure that before a block takes a
// ticket, all of its memory transactions have completed.  This is what
// __threadfence() does -- it blocks until the results of all outstanding memory
// transactions within the calling thread are visible to all other threads.
//
// For more details on the reduction algorithm (notably the multi-pass
// approach), see the "reduction" sample in the CUDA SDK.
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

        // wait until all outstanding memory instructions in this thread are
        // finished
        __threadfence();

        // Thread 0 takes a ticket
        if (tid == 0) {
            unsigned int ticket = atomicInc(&retirementCount, gridDim.x);
            // If the ticket ID is equal to the number of blocks, we are the last
            // block!
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

int main(int argc, char **argv) {
  const int N  = 131072;  // = 2 * 256 * 256, a power of 2
  const int T  = 256;     // threads per block, level 1 / Phase 1 (both variants)
  const int B  = 256;     // blocks launched for level 1 / Phase 1 (both variants)
  const int T2 = 128;     // threads for the two-launch variant's level-2 kernel
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
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

  // CPU reference check: each variant against a CPU replica of its own
  // exact combine tree (see reference.h for why a single naive linear-order
  // sum cannot serve as the reference for both).
  float ref_single_pass = reference_single_pass(h_input, N, T, B);
  float ref_two_launch  = reference_two_launch(h_input, N, T, B, T2);

  double abs_err_single = (double)h_single_pass_result - (double)ref_single_pass;
  if (abs_err_single < 0) abs_err_single = -abs_err_single;
  double abs_err_two = (double)h_two_launch_result - (double)ref_two_launch;
  if (abs_err_two < 0) abs_err_two = -abs_err_two;

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  fprintf(f, "%.9g\n", h_single_pass_result);
  fprintf(f, "%.9g\n", h_two_launch_result);
  fclose(f);

  printf("threadFenceSinglePassReduction done: N=%d, single_pass=%.9g (abs_err=%.3e), "
         "two_launch=%.9g (abs_err=%.3e) -> %s\n",
         N, h_single_pass_result, abs_err_single, h_two_launch_result, abs_err_two, out);
  printf("%s\n", (abs_err_single == 0.0 && abs_err_two == 0.0) ? "PASS" : "FAIL");

  cudaFree(d_input);
  cudaFree(d_partial_sp);
  cudaFree(d_partial_mp);
  free(h_input);
  return 0;
}
