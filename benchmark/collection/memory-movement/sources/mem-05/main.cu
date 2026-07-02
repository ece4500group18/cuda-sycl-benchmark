// constMemMatrixDims: elementwise 2D matrix-add reading its matrix
// dimensions from ordinary kernel arguments vs. from __constant__ memory.
//
// The two __global__ kernels below are reproduced, unmodified, from:
//   CUDAMicroBench (ReadOnlyMem_2D_Texture/matadd_2D_cudakernel.cu)
//   https://github.com/passlab/CUDAMicroBench
//   Copyright (c) 2021, University of North Carolina at Charlotte
//   and Lawrence Livermore National Security, LLC.
//   SPDX-License-Identifier: BSD-3-Clause
//
// Both kernels compute the exact same M x N row-major elementwise sum
// d_Result[i] = d_matrixA[i] + d_matrixB[i], using the exact same thread
// -> element mapping (`tidx` = row, `tidy` = column, linear index
// `tidx * N + tidy`), and differ in exactly one thing -- where the bound
// `N` (and the row bound `M`) used in the in-bounds check and the index
// computation comes from:
//
// "add": `d_M`/`d_N` are ordinary `int` kernel parameters, passed by value
//   in the launch's argument buffer -- each thread reads them out of
//   `__constant__`-backed *parameter* space, but every kernel launch must
//   re-supply them.
//
// "add_const": reads `cons_M`/`cons_N` from a `__constant__` global
//   variable, populated once via `cudaMemcpyToSymbol` before the launch --
//   every thread block, and every kernel launch that follows, can reuse
//   the already-resident, broadcast-cached values without the host having
//   to pass them again.
//
// This isolates the constant-memory broadcast/read-only-cache technique
// applied to small, uniformly-read *metadata* (a matrix's dimensions),
// not to the bulk data itself (contrast with cases that put an entire
// data array in `__constant__` memory). Since `M` and `N` here always
// hold the same values by construction, both kernels touch the same
// elements in the same order and produce a byte-identical result.
//
// The original file's `add_warmingup` (a throwaway duplicate of `add`
// used only to warm up the GPU before timing) and the texture-memory
// kernels `add_texture`/`add_texture_constant` (legacy texture-reference
// API, out of scope for this case) are omitted.
//
// Deterministic inputs (replicated by reference.h):
//   A[i] = ((i % 17) - 8) * 0.5
//   B[i] = ((i % 13) - 6) * 0.25
// M = N = 512 (multiple of BLOCK_SIZE=16)
//
// Output: argv[1] (default output/cuda_output.txt), one float per line,
// row-major M*N result computed by add_const (the __constant__-memory
// variant).

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

#define BLOCK_SIZE 16

// --- begin: verbatim from CUDAMicroBench ReadOnlyMem_2D_Texture/matadd_2D_cudakernel.cu ---

//constant memory
__constant__ int cons_M;
__constant__ int cons_N;

__global__ void add(float * d_matrixA, float * d_matrixB, float *d_Result, int d_M, int d_N)
{
    const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    const int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    if(tidx<d_M && tidy<d_N) {
        d_Result[tidx * d_N + tidy] = d_matrixA[tidx * d_N + tidy] + d_matrixB[tidx * d_N + tidy];
    }
}

__global__ void add_const(float * d_matrixA, float * d_matrixB, float *d_Result)
{
    const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    const int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    if(tidx<cons_M && tidy<cons_N) {
        d_Result[tidx * cons_N + tidy] = d_matrixA[tidx * cons_N + tidy] + d_matrixB[tidx * cons_N + tidy];
    }
}

// --- end: verbatim from CUDAMicroBench ---

int main(int argc, char **argv) {
  const int M = 512, N = 512;  // both multiples of BLOCK_SIZE
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  const size_t bytes = (size_t)M * N * sizeof(float);

  float *hA = (float *)malloc(bytes);
  float *hB = (float *)malloc(bytes);
  float *hResult_add = (float *)malloc(bytes);
  float *hResult_const = (float *)malloc(bytes);

  for (int i = 0; i < M * N; ++i) {
    hA[i] = gen_a(i);
    hB[i] = gen_b(i);
  }

  float *dA, *dB, *dResult_add, *dResult_const;
  CHECK(cudaMalloc(&dA, bytes));
  CHECK(cudaMalloc(&dB, bytes));
  CHECK(cudaMalloc(&dResult_add, bytes));
  CHECK(cudaMalloc(&dResult_const, bytes));

  CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

  // Populate the __constant__ dimensions once, before either launch.
  CHECK(cudaMemcpyToSymbol(cons_M, &M, sizeof(int)));
  CHECK(cudaMemcpyToSymbol(cons_N, &N, sizeof(int)));

  dim3 threadsperblock(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 blocks(M / BLOCK_SIZE, N / BLOCK_SIZE, 1);

  add<<<blocks, threadsperblock>>>(dA, dB, dResult_add, M, N);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  add_const<<<blocks, threadsperblock>>>(dA, dB, dResult_const);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  CHECK(cudaMemcpy(hResult_add, dResult_add, bytes, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(hResult_const, dResult_const, bytes, cudaMemcpyDeviceToHost));

  // CPU reference check.
  float *href = (float *)malloc(bytes);
  reference_add(hA, hB, href, M, N);

  double max_abs_err_add = 0.0, max_abs_err_const = 0.0;
  for (int i = 0; i < M * N; ++i) {
    double d1 = (double)hResult_add[i] - (double)href[i];
    double d2 = (double)hResult_const[i] - (double)href[i];
    if (d1 < 0) d1 = -d1;
    if (d2 < 0) d2 = -d2;
    if (d1 > max_abs_err_add) max_abs_err_add = d1;
    if (d2 > max_abs_err_const) max_abs_err_const = d2;
  }

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (int i = 0; i < M * N; ++i) fprintf(f, "%.9g\n", hResult_const[i]);
  fclose(f);

  printf("constMemMatrixDims done: M=%d, N=%d, "
         "max_abs_error(add vs ref)=%.3e, max_abs_error(add_const vs ref)=%.3e -> %s\n",
         M, N, max_abs_err_add, max_abs_err_const, out);
  printf("%s\n", (max_abs_err_add == 0.0 && max_abs_err_const == 0.0) ? "PASS" : "FAIL");

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dResult_add);
  cudaFree(dResult_const);
  free(hA);
  free(hB);
  free(hResult_add);
  free(hResult_const);
  free(href);
  return 0;
}
