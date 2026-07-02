// sharedScanVsShuffleScan: single-block, 256-element unsigned-int exclusive
// prefix sum (scan), shared-memory-buffered Hillis-Steele scan vs. a
// register/warp-shuffle scan.
//
// The device helpers below (scan1Inclusive, scan1Exclusive, scan4Inclusive,
// scan4Exclusive) and the __global__ kernel scanExclusiveShared are
// reproduced, unmodified in body, from:
//   NVIDIA/cuda-samples, cpp/2_Concepts_and_Techniques/scan/scan.cu
//   https://github.com/NVIDIA/cuda-samples/blob/master/cpp/2_Concepts_and_Techniques/scan/scan.cu
//   Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//   SPDX-License-Identifier: BSD-3-Clause
//
// scanExclusiveShared computes an exclusive prefix sum by first doing a
// tiny "level-0" scan of 4 elements held in a thread's own uint4 register,
// then a "level-1" Hillis-Steele inclusive scan (scan1Inclusive) of the
// resulting per-thread partial sums that is entirely staged through a
// padded __shared__ uint buffer (`s_Data`): every intermediate partial sum,
// at every one of log2(size) doubling steps, is written to and read back
// from shared memory (with two `cg::sync(cta)` barriers per step to make
// the read-then-write hazard-free).
//
// "scanExclusiveShuffle" (new code written for this repository, not part
// of the original file) computes the *exact same* 256-element exclusive
// prefix sum a different way: each of 256 threads keeps its value in a
// register and combines it with other lanes' registers directly via
// `__shfl_up_sync` (an intra-warp, register-to-register Hillis-Steele
// scan -- no shared memory, no barriers, needed to move data between the
// 32 lanes of one warp). Shared memory is touched only for the tiny
// 8-element step of combining the 8 warps' totals into per-warp offsets
// (unavoidable once a scan spans more than one warp) -- a few hundred
// bytes of shared traffic total, vs. `scanExclusiveShared`'s full
// 256-element shared-memory buffer read/written at every doubling step.
//
// Both kernels compute the identical exclusive prefix sum over the same
// 256 unsigned ints in a single thread block; only *how the partial sums
// move between threads* (shared-memory-staged vs. warp-shuffle register
// traffic) differs -- the intra-block "memory movement" contrast this
// case isolates.
//
// Deterministic input (replicated by reference.h):
//   in[i] = (i % 7) + 1, i in [0, 256)
// Unsigned-integer addition is exact and order-independent (commutative,
// associative, and the running sum never approaches 2^32), so a plain
// sequential CPU prefix sum is an exact (max_abs_error == 0) reference
// for both GPU scans regardless of the reduction-tree shape each uses.
//
// Output: argv[1] (default output/cuda_output.txt), 256 lines, the
// exclusive scan computed by scanExclusiveShuffle.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "reference.h"

namespace cg = cooperative_groups;

typedef unsigned int uint;

#define CHECK(call)                                                          \
  do {                                                                       \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), \
              __FILE__, __LINE__);                                          \
      return 2;                                                             \
    }                                                                        \
  } while (0)

// THREADBLOCK_SIZE is the number of *threads* launched for
// scanExclusiveShared; each thread handles one uint4 (4 scalar elements),
// so THREADBLOCK_SIZE=64 processes 4*64 = 256 scalar elements -- the
// upstream file itself defines THREADBLOCK_SIZE=256 (1024 elements per
// block); the value is lowered here purely as a launch-configuration
// choice to land on exactly the 256-element single-block problem size
// used throughout this case. The kernel bodies below are otherwise
// character-for-character unmodified.
#define THREADBLOCK_SIZE 64
#define N_ELEMENTS 256  // = 4 * THREADBLOCK_SIZE

// --- begin: verbatim (bodies unmodified) from NVIDIA/cuda-samples scan.cu ---

// Naive inclusive scan: O(N * log2(N)) operations
// Allocate 2 * 'size' local memory, initialize the first half
// with 'size' zeros avoiding if(pos >= offset) condition evaluation
// and saving instructions
inline __device__ uint scan1Inclusive(uint idata, volatile uint *s_Data, uint size, cg::thread_block cta)
{
    uint pos    = 2 * threadIdx.x - (threadIdx.x & (size - 1));
    s_Data[pos] = 0;
    pos += size;
    s_Data[pos] = idata;

    for (uint offset = 1; offset < size; offset <<= 1) {
        cg::sync(cta);
        uint t = s_Data[pos] + s_Data[pos - offset];
        cg::sync(cta);
        s_Data[pos] = t;
    }

    return s_Data[pos];
}

inline __device__ uint scan1Exclusive(uint idata, volatile uint *s_Data, uint size, cg::thread_block cta)
{
    return scan1Inclusive(idata, s_Data, size, cta) - idata;
}

inline __device__ uint4 scan4Inclusive(uint4 idata4, volatile uint *s_Data, uint size, cg::thread_block cta)
{
    // Level-0 inclusive scan
    idata4.y += idata4.x;
    idata4.z += idata4.y;
    idata4.w += idata4.z;

    // Level-1 exclusive scan
    uint oval = scan1Exclusive(idata4.w, s_Data, size / 4, cta);

    idata4.x += oval;
    idata4.y += oval;
    idata4.z += oval;
    idata4.w += oval;

    return idata4;
}

// Exclusive vector scan: the array to be scanned is stored
// in local thread memory scope as uint4
inline __device__ uint4 scan4Exclusive(uint4 idata4, volatile uint *s_Data, uint size, cg::thread_block cta)
{
    uint4 odata4 = scan4Inclusive(idata4, s_Data, size, cta);
    odata4.x -= idata4.x;
    odata4.y -= idata4.y;
    odata4.z -= idata4.z;
    odata4.w -= idata4.w;
    return odata4;
}

__global__ void scanExclusiveShared(uint4 *d_Dst, uint4 *d_Src, uint size)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    __shared__ uint  s_Data[2 * THREADBLOCK_SIZE];

    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data
    uint4 idata4 = d_Src[pos];

    // Calculate exclusive scan
    uint4 odata4 = scan4Exclusive(idata4, s_Data, size, cta);

    // Write back
    d_Dst[pos] = odata4;
}

// --- end: verbatim from NVIDIA/cuda-samples ---

// --- begin: new code for this repository (not part of the original file) ---
//
// scanExclusiveShuffle: exclusive prefix sum of N_ELEMENTS (256) uints,
// single block of N_ELEMENTS threads (8 warps). Each thread's value stays
// in a register; inter-lane movement within a warp is done with
// __shfl_up_sync (no shared memory, no barrier needed within a warp).
// Shared memory is used only to combine the 8 warps' totals into 8
// per-warp offsets, the minimum amount of shared traffic a >1-warp scan
// can't avoid.
#define WARP_SIZE 32
#define NUM_WARPS (N_ELEMENTS / WARP_SIZE)

__global__ void scanExclusiveShuffle(uint *d_Dst, const uint *d_Src, uint n)
{
    const uint tid      = threadIdx.x;
    const uint lane     = tid & (WARP_SIZE - 1);
    const uint warpId   = tid / WARP_SIZE;
    const uint original = d_Src[tid];
    uint       val      = original;

    // Intra-warp inclusive scan, register-to-register via shuffle.
    for (uint offset = 1; offset < WARP_SIZE; offset <<= 1) {
        uint n_val = __shfl_up_sync(0xffffffffu, val, offset);
        if (lane >= offset) val += n_val;
    }

    __shared__ uint s_WarpTotal[NUM_WARPS];
    __shared__ uint s_WarpOffset[NUM_WARPS];

    if (lane == WARP_SIZE - 1) s_WarpTotal[warpId] = val;
    __syncthreads();

    // Tiny sequential exclusive scan of the 8 warp totals (single thread;
    // 8 additions is not worth parallelizing and does not change the
    // result, since unsigned addition is associative/commutative).
    if (tid == 0) {
        uint running = 0;
        for (uint w = 0; w < NUM_WARPS; ++w) {
            s_WarpOffset[w] = running;
            running += s_WarpTotal[w];
        }
    }
    __syncthreads();

    uint inclusive = s_WarpOffset[warpId] + val;
    d_Dst[tid]     = inclusive - original;  // convert inclusive -> exclusive
}

// --- end: new code for this repository ---

int main(int argc, char **argv) {
  const uint n = N_ELEMENTS;  // 256
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  const size_t bytes = (size_t)n * sizeof(uint);

  uint *hIn           = (uint *)malloc(bytes);
  uint *hOut_shared   = (uint *)malloc(bytes);
  uint *hOut_shuffle  = (uint *)malloc(bytes);
  uint *href          = (uint *)malloc(bytes);

  for (uint i = 0; i < n; ++i) hIn[i] = gen_in(i);

  uint *dIn_shared, *dOut_shared, *dIn_shuffle, *dOut_shuffle;
  CHECK(cudaMalloc(&dIn_shared, bytes));
  CHECK(cudaMalloc(&dOut_shared, bytes));
  CHECK(cudaMalloc(&dIn_shuffle, bytes));
  CHECK(cudaMalloc(&dOut_shuffle, bytes));

  CHECK(cudaMemcpy(dIn_shared, hIn, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dIn_shuffle, hIn, bytes, cudaMemcpyHostToDevice));

  // scanExclusiveShared: single block, THREADBLOCK_SIZE=64 threads, each
  // handling a uint4 (4 scalar elements) -> 256 scalar elements total.
  scanExclusiveShared<<<1, THREADBLOCK_SIZE>>>((uint4 *)dOut_shared, (uint4 *)dIn_shared, n);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  // scanExclusiveShuffle: single block, 256 threads, one scalar per thread.
  scanExclusiveShuffle<<<1, N_ELEMENTS>>>(dOut_shuffle, dIn_shuffle, n);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  CHECK(cudaMemcpy(hOut_shared, dOut_shared, bytes, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(hOut_shuffle, dOut_shuffle, bytes, cudaMemcpyDeviceToHost));

  // CPU reference check.
  reference_exclusive_scan(hIn, href, n);

  long long max_abs_err_shared = 0, max_abs_err_shuffle = 0;
  for (uint i = 0; i < n; ++i) {
    long long d1 = (long long)hOut_shared[i] - (long long)href[i];
    long long d2 = (long long)hOut_shuffle[i] - (long long)href[i];
    if (d1 < 0) d1 = -d1;
    if (d2 < 0) d2 = -d2;
    if (d1 > max_abs_err_shared) max_abs_err_shared = d1;
    if (d2 > max_abs_err_shuffle) max_abs_err_shuffle = d2;
  }

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (uint i = 0; i < n; ++i) fprintf(f, "%u\n", hOut_shuffle[i]);
  fclose(f);

  printf("sharedScanVsShuffleScan done: n=%u, "
         "max_abs_error(shared vs ref)=%lld, max_abs_error(shuffle vs ref)=%lld -> %s\n",
         n, max_abs_err_shared, max_abs_err_shuffle, out);
  printf("%s\n", (max_abs_err_shared == 0 && max_abs_err_shuffle == 0) ? "PASS" : "FAIL");

  cudaFree(dIn_shared);
  cudaFree(dOut_shared);
  cudaFree(dIn_shuffle);
  cudaFree(dOut_shuffle);
  free(hIn);
  free(hOut_shared);
  free(hOut_shuffle);
  free(href);
  return 0;
}
