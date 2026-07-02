// warpShuffleReduction: shared-memory-only sum reduction vs. a
// warp-shuffle-accelerated sum reduction.
//
// The kernels and helper templates below are reproduced, unmodified
// (aside from instantiating the templates for float / blockSize=256),
// from:
//   NVIDIA/cuda-samples, Samples/2_Concepts_and_Techniques/reduction
//   (reduction_kernel.cu), also redistributed as
//   CUDAMicroBench Shuffle/cuda_shuffle/reduction_kernel.cu
//   https://github.com/passlab/CUDAMicroBench
//   Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//   SPDX-License-Identifier: BSD-3-Clause
//
// "reduce2" (sequential addressing, n threads): each thread loads one
//   element into shared memory, then a tree of additions with
//   decreasing stride (`s = blockDim.x/2 ... 1`) reduces it, guarded by
//   __syncthreads(). No bank conflicts, but every reduction step (down
//   to the last 2 elements) goes through __syncthreads().
//
// "reduce4" (first-add-during-load + warp shuffle, n/2 threads): each
//   thread loads *two* elements and adds them while reading from global
//   memory, then reduces down to 64 elements (2 warps) via shared
//   memory with __syncthreads(), and finally reduces the last warp
//   using `__shfl_down_sync` (warp shuffle) with **no** __syncthreads()
//   at all -- shuffle instructions exchange register values directly
//   between lanes of the same warp.
//
// Both kernels compute the same per-block sum; only *how* values move
// between threads during the reduction (shared memory + syncthreads vs.
// warp-shuffle register exchange) differs. This is the canonical
// "simple but not trivial" kernel: the math is just `+=`, but the
// shuffle-based reduction is a widely-used, non-obvious technique.
//
// Deterministic input (replicated by reference.h):
//   x[i] = ((i % 17) - 8) * 0.25f,  i in [0, n)
// n = 1048576 (2^20), threads = 256
//
// Output: argv[1] (default output/cuda_output.txt), one float per
// line, containing the per-block partial sums from reduce4
// (n/(2*256) = 2048 values).

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "reference.h"

namespace cg = cooperative_groups;

#define CHECK(call)                                                          \
  do {                                                                       \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), \
              __FILE__, __LINE__);                                           \
      return 2;                                                              \
    }                                                                        \
  } while (0)

// --- begin: verbatim from NVIDIA cuda-samples reduction_kernel.cu ---

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template <class T>
struct SharedMemory {
  __device__ inline operator T *() {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }

  __device__ inline operator const T *() const {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }
};

// specialize for double to avoid unaligned memory
// access compile errors
template <>
struct SharedMemory<double> {
  __device__ inline operator double *() {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }

  __device__ inline operator const double *() const {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }
};

/*
    This version uses sequential addressing -- no divergence or bank conflicts.
*/
template <class T>
__global__ void reduce2(T *g_idata, T *g_odata, unsigned int n) {
  cg::thread_block cta = cg::this_thread_block();
  T *sdata = SharedMemory<T>();

  // load shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (i < n) ? g_idata[i] : 0;

  cg::sync(cta);

  // do reduction in shared mem
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }

    cg::sync(cta);
  }

  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <class T>
__inline__ __device__ T warpReduceSum(T val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2)
    val += __shfl_down_sync((unsigned int)-1, val, offset);
  return val;
}

/*
    This version uses the warp shuffle operation if available to reduce
    warp synchronization. When shuffle is not available the final warp's
    worth of work is unrolled to reduce looping overhead.

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <class T, unsigned int blockSize>
__global__ void reduce4(T *g_idata, T *g_odata, unsigned int n) {
  cg::thread_block cta = cg::this_thread_block();
  T *sdata = SharedMemory<T>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

  T mySum = (i < n) ? g_idata[i] : 0;

  if (i + blockSize < n) mySum += g_idata[i + blockSize];

  sdata[tid] = mySum;
  cg::sync(cta);

  // do reduction in shared mem
  for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      sdata[tid] = mySum = mySum + sdata[tid + s];
    }

    cg::sync(cta);
  }

  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  if (cta.thread_rank() < 32) {
    // Fetch final intermediate sum from 2nd warp
    if (blockSize >= 64) mySum += sdata[tid + 32];
    // Reduce final warp using shuffle
    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
      mySum += tile32.shfl_down(mySum, offset);
    }
  }

  // write result for this block to global mem
  if (cta.thread_rank() == 0) g_odata[blockIdx.x] = mySum;
}

// --- end: verbatim from NVIDIA cuda-samples ---

int main(int argc, char **argv) {
  const unsigned int n = 1 << 20;  // 1048576
  const unsigned int threads = 256;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";

  const unsigned int blocks2 = n / threads;          // reduce2: 1 elem/thread
  const unsigned int blocks4 = n / (2 * threads);    // reduce4: 2 elem/thread

  const size_t in_bytes = (size_t)n * sizeof(float);
  const size_t out2_bytes = (size_t)blocks2 * sizeof(float);
  const size_t out4_bytes = (size_t)blocks4 * sizeof(float);

  float *hx = (float *)malloc(in_bytes);
  float *h_out2 = (float *)malloc(out2_bytes);
  float *h_out4 = (float *)malloc(out4_bytes);

  for (unsigned int i = 0; i < n; ++i) hx[i] = gen_x(i);

  float *dx, *dout2, *dout4;
  CHECK(cudaMalloc(&dx, in_bytes));
  CHECK(cudaMalloc(&dout2, out2_bytes));
  CHECK(cudaMalloc(&dout4, out4_bytes));

  CHECK(cudaMemcpy(dx, hx, in_bytes, cudaMemcpyHostToDevice));

  // reduce2: n threads total, 256 threads/block, smem = 256*sizeof(float)
  reduce2<float><<<blocks2, threads, threads * sizeof(float)>>>(dx, dout2, n);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  // reduce4: n/2 threads total, 256 threads/block, smem = 256*sizeof(float)
  reduce4<float, 256><<<blocks4, threads, threads * sizeof(float)>>>(dx, dout4, n);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  CHECK(cudaMemcpy(h_out2, dout2, out2_bytes, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(h_out4, dout4, out4_bytes, cudaMemcpyDeviceToHost));

  // CPU reference check.
  float *href2 = (float *)malloc(out2_bytes);
  float *href4 = (float *)malloc(out4_bytes);
  reference_block_sum(hx, href2, n, threads);
  reference_block_sum(hx, href4, n, 2 * threads);

  float max_abs_err2 = 0.0f, max_abs_err4 = 0.0f;
  for (unsigned int b = 0; b < blocks2; ++b) {
    float d = h_out2[b] - href2[b];
    if (d < 0) d = -d;
    if (d > max_abs_err2) max_abs_err2 = d;
  }
  for (unsigned int b = 0; b < blocks4; ++b) {
    float d = h_out4[b] - href4[b];
    if (d < 0) d = -d;
    if (d > max_abs_err4) max_abs_err4 = d;
  }

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (unsigned int b = 0; b < blocks4; ++b) fprintf(f, "%.9g\n", h_out4[b]);
  fclose(f);

  printf("warpShuffleReduction done: n=%u, "
         "max_abs_error(reduce2 vs ref)=%.3e, max_abs_error(reduce4 vs ref)=%.3e -> %s\n",
         n, max_abs_err2, max_abs_err4, out);
  printf("%s\n", (max_abs_err2 < 1e-2f && max_abs_err4 < 1e-2f) ? "PASS" : "FAIL");

  cudaFree(dx);
  cudaFree(dout2);
  cudaFree(dout4);
  free(hx);
  free(h_out2);
  free(h_out4);
  free(href2);
  free(href4);
  return 0;
}
