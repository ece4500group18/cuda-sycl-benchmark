// bankConflictReduction: shared-memory sum reduction with vs. without
// bank conflicts.
//
// The two __global__ kernels below are reproduced, unmodified (aside
// from the warm-up kernel being omitted), from:
//   CUDAMicroBench (BankRedux/sum_cudakernel.cu)
//   https://github.com/passlab/CUDAMicroBench
//   Copyright (c) 2021, University of North Carolina at Charlotte
//   and Lawrence Livermore National Security, LLC.
//   SPDX-License-Identifier: BSD-3-Clause
//
// "sum_cudakernel" (sequential addressing): the active-thread stride
//   `i` halves each step (256,128,64,...) and `cache[cacheIndex]` is
//   accessed by consecutive thread indices -> consecutive shared-memory
//   banks, no bank conflicts.
//
// "sum_cudakernel_bc" (interleaved addressing): `index = 2*i*cacheIndex`
//   grows by a power-of-two stride each step. For i>=2 multiple threads
//   map to shared-memory addresses that differ by a multiple of 32
//   words, i.e. the same bank -> shared-memory bank conflicts.
//
// Both kernels compute the *same* per-block sum; only the shared-memory
// access pattern (and therefore bank-conflict behaviour) differs. This
// is a memory-layout case: identical data layout in shared memory,
// different index pattern used to reduce it.
//
// Deterministic input (replicated by reference.h):
//   x[i] = ((i % 17) - 8) * 0.25f,  i in [0, n)
// n = 1024000, ThreadsPerBlock = 256
//
// Output: argv[1] (default output/cuda_output.txt), one float per line,
// containing result[] from the sequential-addressing kernel
// (sum_cudakernel) -- one partial sum per block.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "reference.h"

#define CHECK(call)                                                          \
  do {                                                                       \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), \
              __FILE__, __LINE__);                                           \
      return 2;                                                              \
    }                                                                        \
  } while (0)

#define REAL float
#define ThreadsPerBlock 256

// --- begin: verbatim from CUDAMicroBench BankRedux/sum_cudakernel.cu ---

__global__ void sum_cudakernel(const REAL *x, REAL *result) {
  __shared__ REAL cache[ThreadsPerBlock];
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int cacheIndex = threadIdx.x;
  cache[cacheIndex] = x[tid];
  __syncthreads();
  for (int i = blockDim.x / 2; i > 0; i /= 2) {
    if (cacheIndex < i) {
      cache[cacheIndex] += cache[cacheIndex + i];
    }
    __syncthreads();
  }
  if (cacheIndex == 0)
  result[blockIdx.x] = cache[cacheIndex];
}

__global__ void sum_cudakernel_bc(const REAL *x, REAL *result) {
  __shared__ REAL cache[ThreadsPerBlock];
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int cacheIndex = threadIdx.x;
  cache[cacheIndex] = x[tid];
  __syncthreads();
  for (int i = 1; i < blockDim.x; i *= 2) {
    int index = 2 * i * cacheIndex;
    if (index < blockDim.x) {
      cache[index] += cache[index + i];
    }
    __syncthreads();
  }
  if (cacheIndex == 0)
  result[blockIdx.x] = cache[cacheIndex];
}

// --- end: verbatim from CUDAMicroBench ---

int main(int argc, char **argv) {
  const int n = 1024000;  // VEC_LEN
  const int threads = ThreadsPerBlock;
  const int blocks = (n + threads - 1) / threads;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";

  const size_t in_bytes = (size_t)n * sizeof(REAL);
  const size_t out_bytes = (size_t)blocks * sizeof(REAL);

  REAL *hx = (REAL *)malloc(in_bytes);
  REAL *h_result = (REAL *)malloc(out_bytes);
  REAL *h_result_bc = (REAL *)malloc(out_bytes);

  for (int i = 0; i < n; ++i) hx[i] = gen_x(i);

  REAL *dx, *d_result, *d_result_bc;
  CHECK(cudaMalloc(&dx, in_bytes));
  CHECK(cudaMalloc(&d_result, out_bytes));
  CHECK(cudaMalloc(&d_result_bc, out_bytes));

  CHECK(cudaMemcpy(dx, hx, in_bytes, cudaMemcpyHostToDevice));

  sum_cudakernel<<<blocks, threads>>>(dx, d_result);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  sum_cudakernel_bc<<<blocks, threads>>>(dx, d_result_bc);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  CHECK(cudaMemcpy(h_result, d_result, out_bytes, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(h_result_bc, d_result_bc, out_bytes, cudaMemcpyDeviceToHost));

  // CPU reference check.
  REAL *href = (REAL *)malloc(out_bytes);
  reference_block_sum(hx, href, n, threads);

  float max_abs_err = 0.0f, max_abs_err_bc = 0.0f;
  for (int b = 0; b < blocks; ++b) {
    float d1 = h_result[b] - href[b];
    float d2 = h_result_bc[b] - href[b];
    if (d1 < 0) d1 = -d1;
    if (d2 < 0) d2 = -d2;
    if (d1 > max_abs_err) max_abs_err = d1;
    if (d2 > max_abs_err_bc) max_abs_err_bc = d2;
  }

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (int b = 0; b < blocks; ++b) fprintf(f, "%.9g\n", h_result[b]);
  fclose(f);

  printf("bankConflictReduction done: n=%d, blocks=%d, "
         "max_abs_error(seq vs ref)=%.3e, max_abs_error(bc vs ref)=%.3e -> %s\n",
         n, blocks, max_abs_err, max_abs_err_bc, out);
  printf("%s\n", (max_abs_err < 1e-3f && max_abs_err_bc < 1e-3f) ? "PASS" : "FAIL");

  cudaFree(dx);
  cudaFree(d_result);
  cudaFree(d_result_bc);
  free(hx);
  free(h_result);
  free(h_result_bc);
  free(href);
  return 0;
}
