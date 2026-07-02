// tiledMatmulShmem: tiled matrix multiplication, global-memory-only vs.
// shared-memory-tiled.
//
// The two __global__ kernels below are reproduced, unmodified, from:
//   CUDAMicroBench (Shmem/mm_kernel.cu)
//   https://github.com/passlab/CUDAMicroBench
//   Copyright (c) 2021, University of North Carolina at Charlotte
//   and Lawrence Livermore National Security, LLC.
//   SPDX-License-Identifier: BSD-3-Clause
//
// Both kernels compute C = A * B for an n x n row-major matrix using a
// BLOCK_SIZE x BLOCK_SIZE tile per thread block:
//
// "global_block": each thread accumulates Csub by repeatedly reading
//   BLOCK_SIZE elements of A and B directly from *global* memory inside
//   the inner k-loop -- the same A/B tile elements are re-read from
//   global memory by every thread in the block.
//
// "shared_block": each thread block first copies its current A-tile and
//   B-tile into __shared__ arrays As/Bs (one element per thread), calls
//   __syncthreads(), then every thread computes its partial dot product
//   by reading the tile from *shared* memory -- each global-memory
//   element is read once per block instead of once per thread.
//
// Both kernels compute the exact same C = A*B; only where the A/B tile
// data lives during the inner product (global vs. shared memory)
// differs -- a canonical memory-movement / memory-layout case.
//
// Deterministic inputs (replicated by reference.h):
//   A[i] = ((i % 17) - 8) * 0.25
//   B[i] = ((i % 23) - 11) * 0.5
// n = 128 (multiple of BLOCK_SIZE=16)
//
// Output: argv[1] (default output/cuda_output.txt), one double per
// line, row-major C = A*B computed by the shared-memory kernel
// (shared_block).

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

#define REAL double
#define BLOCK_SIZE 16

// --- begin: verbatim from CUDAMicroBench Shmem/mm_kernel.cu ---

__global__
void global_block(REAL* A, REAL* B, REAL* C, int n) {
    int wA = n;
    int wB = n;

    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    REAL Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
      for (int k = 0; k < BLOCK_SIZE; ++k) {
          Csub += A[a + wA * ty + k] * B[b + wB * k + tx];
      }
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

__global__
void shared_block(REAL* A, REAL* B, REAL* C, int n) {
    int wA = n;
    int wB = n;

    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    REAL Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ REAL As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ REAL Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

// --- end: verbatim from CUDAMicroBench ---

int main(int argc, char **argv) {
  const int n = 128;  // multiple of BLOCK_SIZE
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  const size_t bytes = (size_t)n * n * sizeof(REAL);

  REAL *hA = (REAL *)malloc(bytes);
  REAL *hB = (REAL *)malloc(bytes);
  REAL *hC_global = (REAL *)malloc(bytes);
  REAL *hC_shared = (REAL *)malloc(bytes);

  for (int i = 0; i < n * n; ++i) {
    hA[i] = gen_a(i);
    hB[i] = gen_b(i);
  }

  REAL *dA, *dB, *dC_global, *dC_shared;
  CHECK(cudaMalloc(&dA, bytes));
  CHECK(cudaMalloc(&dB, bytes));
  CHECK(cudaMalloc(&dC_global, bytes));
  CHECK(cudaMalloc(&dC_shared, bytes));

  CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(n / BLOCK_SIZE, n / BLOCK_SIZE);

  global_block<<<dimGrid, dimBlock>>>(dA, dB, dC_global, n);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  shared_block<<<dimGrid, dimBlock>>>(dA, dB, dC_shared, n);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  CHECK(cudaMemcpy(hC_global, dC_global, bytes, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(hC_shared, dC_shared, bytes, cudaMemcpyDeviceToHost));

  // CPU reference check.
  REAL *href = (REAL *)malloc(bytes);
  reference_matmul(hA, hB, href, n);

  double max_abs_err_global = 0.0, max_abs_err_shared = 0.0;
  for (int i = 0; i < n * n; ++i) {
    double d1 = hC_global[i] - href[i];
    double d2 = hC_shared[i] - href[i];
    if (d1 < 0) d1 = -d1;
    if (d2 < 0) d2 = -d2;
    if (d1 > max_abs_err_global) max_abs_err_global = d1;
    if (d2 > max_abs_err_shared) max_abs_err_shared = d2;
  }

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (int i = 0; i < n * n; ++i) fprintf(f, "%.17g\n", hC_shared[i]);
  fclose(f);

  printf("tiledMatmulShmem done: n=%d, "
         "max_abs_error(global vs ref)=%.3e, max_abs_error(shared vs ref)=%.3e -> %s\n",
         n, max_abs_err_global, max_abs_err_shared, out);
  printf("%s\n", (max_abs_err_global < 1e-9 && max_abs_err_shared < 1e-9) ? "PASS" : "FAIL");

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC_global);
  cudaFree(dC_shared);
  free(hA);
  free(hB);
  free(hC_global);
  free(hC_shared);
  free(href);
  return 0;
}
