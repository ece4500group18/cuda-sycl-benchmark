// asyncCopySingleStage: tiled matrix multiplication, classic synchronous
// shared-memory load vs. single-stage `cuda::pipeline`/`__pipeline_memcpy_async`
// asynchronous shared-memory load.
//
// The two __global__ kernel templates below are reproduced, unmodified
// (instantiated here for BLOCK_SIZE=16, float), from:
//   NVIDIA/cuda-samples, cpp/3_CUDA_Features/globalToShmemAsyncCopy/globalToShmemAsyncCopy.cu
//   (also redistributed unchanged as CUDAMicroBench GSOverlap/globalToShmemAsyncCopy.cu,
//    https://github.com/passlab/CUDAMicroBench)
//   Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
//   SPDX-License-Identifier: BSD-3-Clause
//
// Both kernels compute C = A * B for an n x n row-major matrix using a
// BLOCK_SIZE x BLOCK_SIZE tile per thread block, and both keep the *same*
// per-thread accumulation loop (identical math, identical order):
//
// "MatrixMulNaive": each thread block loads its current A-tile/B-tile into
//   __shared__ As/Bs with a plain assignment
//   (`As[threadIdx.y][threadIdx.x] = A[...]`) -- a synchronous, per-thread
//   global-memory read followed by a shared-memory store -- then
//   __syncthreads() before consuming the tile.
//
// "MatrixMulAsyncCopySingleStage": each thread block loads the same tile
//   using `__pipeline_memcpy_async` (global memory -> shared memory,
//   bypassing registers), followed by `__pipeline_commit()` +
//   `__pipeline_wait_prior(0)` instead of a plain assignment, then the same
//   __syncthreads() before consuming the tile. `USE_CPP_API` is left at its
//   upstream default of 0, so this file compiles to the plain
//   `__pipeline_*` intrinsics path (not the `nvcuda::experimental::pipeline`
//   C++ wrapper path), exactly as the original file does out of the box.
//
// Both kernels read/write the same global arrays in the same row-major
// layout and accumulate `Csub` in the exact same per-tile, per-k order --
// this isolates *how* a global-to-shared tile copy is issued (synchronous
// per-thread load-then-store vs. an asynchronous copy-engine instruction
// that a thread can issue without occupying it until the data arrives) as
// the sole difference, the canonical "memory movement" contrast for the
// async-copy hardware path introduced with compute capability 8.0.
//
// Deterministic inputs (replicated by reference.h):
//   A[i] = ((i % 17) - 8) * 0.25f
//   B[i] = ((i % 23) - 11) * 0.5f
// n = 256 (multiple of BLOCK_SIZE=16)
//
// Output: argv[1] (default output/cuda_output.txt), one float per line,
// row-major C = A*B computed by MatrixMulAsyncCopySingleStage.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
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

// --- begin: verbatim from NVIDIA/cuda-samples globalToShmemAsyncCopy.cu ---

#define USE_CPP_API 0

template <int BLOCK_SIZE> __global__ void MatrixMulNaive(float *C, float *A,
                                                        float *B, int wA,
                                                        int wB) {
    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * blockIdx.y;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * blockIdx.x;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
            a <= aEnd;
            a += aStep, b += bStep) {

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[threadIdx.y][threadIdx.x] = A[a + wA * threadIdx.y + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[b + wB * threadIdx.y + threadIdx.x];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * blockIdx.y + BLOCK_SIZE * blockIdx.x;
    C[c + wB * threadIdx.y + threadIdx.x] = Csub;
}

template <int BLOCK_SIZE> __global__ void MatrixMulAsyncCopySingleStage(float *C, const float *A,
                                                        const float *B, int wA,
                                                        int wB) {

    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * blockIdx.y;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * blockIdx.x;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Single-stage pipeline version
    float Csub = 0.0;

#if USE_CPP_API
    nvcuda_namespace::pipeline pipe;
#endif
    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        // Load the matrices from device memory to shared memory; each thread loads
        // one element of each matrix
        {
            const float *A_float = reinterpret_cast<const float*>(A + a + wA * threadIdx.y + threadIdx.x);
            const float *B_float = reinterpret_cast<const float*>(B + b + wB * threadIdx.y + threadIdx.x);

#if USE_CPP_API

            nvcuda_namespace::memcpy_async(As[threadIdx.y][threadIdx.x], *A_float, pipe);
            nvcuda_namespace::memcpy_async(Bs[threadIdx.y][threadIdx.x], *B_float, pipe);

            pipe.commit_and_wait();
#else
            __pipeline_memcpy_async(&As[threadIdx.y][threadIdx.x], A_float, sizeof(float));
            __pipeline_memcpy_async(&Bs[threadIdx.y][threadIdx.x], B_float, sizeof(float));

            __pipeline_commit();
            __pipeline_wait_prior(0);
#endif
        }

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes four element
    int c = wB * BLOCK_SIZE * blockIdx.y + BLOCK_SIZE * blockIdx.x;
    C[c + wB * threadIdx.y + threadIdx.x] = Csub;
}

// --- end: verbatim from NVIDIA/cuda-samples ---

int main(int argc, char **argv) {
  const int n = 256;  // multiple of BLOCK_SIZE
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  const size_t bytes = (size_t)n * n * sizeof(float);

  float *hA = (float *)malloc(bytes);
  float *hB = (float *)malloc(bytes);
  float *hC_naive = (float *)malloc(bytes);
  float *hC_async = (float *)malloc(bytes);

  for (int i = 0; i < n * n; ++i) {
    hA[i] = gen_a(i);
    hB[i] = gen_b(i);
  }

  float *dA, *dB, *dC_naive, *dC_async;
  CHECK(cudaMalloc(&dA, bytes));
  CHECK(cudaMalloc(&dB, bytes));
  CHECK(cudaMalloc(&dC_naive, bytes));
  CHECK(cudaMalloc(&dC_async, bytes));

  CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(n / BLOCK_SIZE, n / BLOCK_SIZE);

  MatrixMulNaive<BLOCK_SIZE><<<dimGrid, dimBlock>>>(dC_naive, dA, dB, n, n);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  MatrixMulAsyncCopySingleStage<BLOCK_SIZE><<<dimGrid, dimBlock>>>(dC_async, dA, dB, n, n);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  CHECK(cudaMemcpy(hC_naive, dC_naive, bytes, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(hC_async, dC_async, bytes, cudaMemcpyDeviceToHost));

  // CPU reference check.
  float *href = (float *)malloc(bytes);
  reference_matmul(hA, hB, href, n);

  double max_abs_err_naive = 0.0, max_abs_err_async = 0.0;
  for (int i = 0; i < n * n; ++i) {
    double d1 = (double)hC_naive[i] - (double)href[i];
    double d2 = (double)hC_async[i] - (double)href[i];
    if (d1 < 0) d1 = -d1;
    if (d2 < 0) d2 = -d2;
    if (d1 > max_abs_err_naive) max_abs_err_naive = d1;
    if (d2 > max_abs_err_async) max_abs_err_async = d2;
  }

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (int i = 0; i < n * n; ++i) fprintf(f, "%.9g\n", hC_async[i]);
  fclose(f);

  printf("asyncCopySingleStage done: n=%d, "
         "max_abs_error(naive vs ref)=%.3e, max_abs_error(async vs ref)=%.3e -> %s\n",
         n, max_abs_err_naive, max_abs_err_async, out);
  printf("%s\n", (max_abs_err_naive < 5e-2 && max_abs_err_async < 5e-2) ? "PASS" : "FAIL");

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC_naive);
  cudaFree(dC_async);
  free(hA);
  free(hB);
  free(hC_naive);
  free(hC_async);
  free(href);
  return 0;
}
