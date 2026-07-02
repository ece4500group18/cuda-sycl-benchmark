// csrScanOrderSpMM: sparse-times-sparse matrix product A*B, computed by
// two kernels that scan/intersect the same CSR-shaped operand data in
// different orders.
//
// The two __global__ kernels below are reproduced, unmodified, from:
//   CUDAMicroBench (CoMem_SpMM/SpMM_cudakernel.cu)
//   https://github.com/passlab/CUDAMicroBench
//   Copyright (c) 2021, University of North Carolina at Charlotte
//   and Lawrence Livermore National Security, LLC.
//   SPDX-License-Identifier: BSD-3-Clause
//
// Both kernels compute the identical sparse matrix product
//   result[row][col] = sum_k A[row][k] * B[k][col]
// for every (row, col) in an NUM_ROWS x NUM_ROWS result, from exactly
// the same two logical matrices A and B -- but they scan the operand
// data in fundamentally different orders:
//
// "spmm_csr_csr_kernel": B is passed in row-major CSR order (`ptrB`
//   indexed by B's row, `indicesB` giving B's column). For each nonzero
//   A[row][k] and each output column `col`, the kernel does a *full
//   linear scan* over all of `indicesB`/`dataB` (all nnzB entries of
//   B), keeping only the entries with `indicesB[j] == col` that also
//   fall in `ptrB[k]..ptrB[k+1]`. Every output column rescans the
//   *entire* B array -- an O(num_rows * nnzB) access pattern per row of
//   A, most of it wasted work re-touching entries that don't match.
//
// "spmm_csc_csr_kernel": the same logical B, but passed in column-major
//   CSC order (`ptrB` indexed by B's column, `indicesB` giving B's row
//   within that column). For each output column `col`, the kernel
//   bounds the scan to `ptrB[col]..ptrB[col+1]` -- B's nonzeros in that
//   column only -- and merges that short range against A's row range
//   (`ptrA[row]..ptrA[row+1]`) by testing `indicesA[i] == indicesB[j]`.
//   This is a textbook bounded intersection of two short, sorted index
//   lists: `nnzA_row + nnzB_col` comparisons per (row, col) pair instead
//   of a full nnzB rescan.
//
// This case is NOT primarily a "CSR-format matrix vs CSC-format matrix"
// benchmark: both kernels compute exactly the same A*B product. What it
// isolates is *memory access order* -- an unbounded linear rescan of an
// entire operand array per output column, vs. a bounded nested-range
// intersection of two short index lists -- for otherwise identical dot
// products. Because every nonzero value is a small integer (see
// reference.h), integer addition is exactly commutative/associative, so
// both scan orders must produce byte-identical results despite summing
// the same partial products in different orders.
//
// Deterministic inputs (replicated by reference.h): A and B are both
// 32 x 32 circulant sparse integer matrices with exactly 4 nonzeros per
// row (and, by construction, exactly 4 nonzeros per column too), built
// from fixed column-offset and value formulas -- no rand()/drand48().
//
// Output: argv[1] (default output/output.txt), NUM_ROWS*NUM_ROWS
// values (one per line, row-major), the spmm_csc_csr_kernel result.

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

#define REAL float

// --- begin: verbatim from CUDAMicroBench CoMem_SpMM/SpMM_cudakernel.cu ---

__global__ void spmm_csr_csr_kernel(const int num_rows, const int *ptrA, const int * indicesA, const REAL *dataA, const int *ptrB, const int * indicesB, const REAL *dataB,  REAL* result, int nnzA, int nnzB)
{
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < num_rows){

        int row_start = ptrA[row];
        int row_end = ptrA[row+1];

        for(int k =0; k<num_rows; k++){ //iterate over B column
          float dot = 0;
          for ( int i = row_start; i < row_end; i++) {
            //int colNum = k;  //The col of the element
            for(int j = 0; j < nnzB; j++) { //nnz should be number of non-zero element of B
              if (indicesB[j] == k && j >= ptrB[indicesA[i]] && j < ptrB[indicesA[i]+1]) {
                dot += dataA[i] * dataB[j];
              }
            }
          }
        result[row*num_rows+k] = dot;
        }
    }
}

__global__ void spmm_csc_csr_kernel(const int num_rows, const int *ptrA, const int * indicesA, const REAL *dataA, const int *ptrB, const int * indicesB, const REAL *dataB,  REAL* result, int nnzA, int nnzB)
{
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < num_rows){
        int row_start = ptrA[row];
        int row_end = ptrA[row+1];
        for(int column = 0; column < num_rows; column++){
            int column_start = ptrB[column];
            int column_end = ptrB[column+1];
            float dot = 0;
            for ( int i = row_start; i < row_end; i++) {
                for(int j = column_start; j < column_end; j++) {
                    if(indicesA[i] == indicesB[j]){
                          dot += dataA[i] * dataB[j];
                    }
                }
            }
         result[row*num_rows+column] = dot;
        }
    }
}

// --- end: verbatim from CUDAMicroBench ---

int main(int argc, char **argv) {
  const int n = NUM_ROWS;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  const size_t result_bytes = (size_t)n * (size_t)n * sizeof(REAL);

  // Host-side operand arrays.
  int *hPtrA = (int *)malloc((n + 1) * sizeof(int));
  int *hIndicesA = (int *)malloc(NNZ_A * sizeof(int));
  REAL *hDataA = (REAL *)malloc(NNZ_A * sizeof(REAL));

  int *hPtrB_csr = (int *)malloc((n + 1) * sizeof(int));
  int *hIndicesB_csr = (int *)malloc(NNZ_B * sizeof(int));
  REAL *hDataB_csr = (REAL *)malloc(NNZ_B * sizeof(REAL));

  int *hPtrB_csc = (int *)malloc((n + 1) * sizeof(int));
  int *hIndicesB_csc = (int *)malloc(NNZ_B * sizeof(int));
  REAL *hDataB_csc = (REAL *)malloc(NNZ_B * sizeof(REAL));

  build_csr_a<REAL>(hPtrA, hIndicesA, hDataA);
  build_csr_b<REAL>(hPtrB_csr, hIndicesB_csr, hDataB_csr);
  build_csc_b<REAL>(hPtrB_csc, hIndicesB_csc, hDataB_csc);

  REAL *hResultCsr = (REAL *)malloc(result_bytes);
  REAL *hResultCsc = (REAL *)malloc(result_bytes);

  // Device buffers. A is shared by both kernels; B is uploaded twice,
  // once in each storage order the corresponding kernel expects.
  int *dPtrA, *dIndicesA, *dPtrB_csr, *dIndicesB_csr, *dPtrB_csc, *dIndicesB_csc;
  REAL *dDataA, *dDataB_csr, *dDataB_csc, *dResultCsr, *dResultCsc;

  CHECK(cudaMalloc(&dPtrA, (n + 1) * sizeof(int)));
  CHECK(cudaMalloc(&dIndicesA, NNZ_A * sizeof(int)));
  CHECK(cudaMalloc(&dDataA, NNZ_A * sizeof(REAL)));

  CHECK(cudaMalloc(&dPtrB_csr, (n + 1) * sizeof(int)));
  CHECK(cudaMalloc(&dIndicesB_csr, NNZ_B * sizeof(int)));
  CHECK(cudaMalloc(&dDataB_csr, NNZ_B * sizeof(REAL)));

  CHECK(cudaMalloc(&dPtrB_csc, (n + 1) * sizeof(int)));
  CHECK(cudaMalloc(&dIndicesB_csc, NNZ_B * sizeof(int)));
  CHECK(cudaMalloc(&dDataB_csc, NNZ_B * sizeof(REAL)));

  CHECK(cudaMalloc(&dResultCsr, result_bytes));
  CHECK(cudaMalloc(&dResultCsc, result_bytes));

  CHECK(cudaMemcpy(dPtrA, hPtrA, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dIndicesA, hIndicesA, NNZ_A * sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dDataA, hDataA, NNZ_A * sizeof(REAL), cudaMemcpyHostToDevice));

  CHECK(cudaMemcpy(dPtrB_csr, hPtrB_csr, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dIndicesB_csr, hIndicesB_csr, NNZ_B * sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dDataB_csr, hDataB_csr, NNZ_B * sizeof(REAL), cudaMemcpyHostToDevice));

  CHECK(cudaMemcpy(dPtrB_csc, hPtrB_csc, (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dIndicesB_csc, hIndicesB_csc, NNZ_B * sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dDataB_csc, hDataB_csc, NNZ_B * sizeof(REAL), cudaMemcpyHostToDevice));

  spmm_csr_csr_kernel<<<1, 32>>>(n, dPtrA, dIndicesA, dDataA, dPtrB_csr, dIndicesB_csr, dDataB_csr, dResultCsr, NNZ_A, NNZ_B);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  spmm_csc_csr_kernel<<<1, 32>>>(n, dPtrA, dIndicesA, dDataA, dPtrB_csc, dIndicesB_csc, dDataB_csc, dResultCsc, NNZ_A, NNZ_B);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  CHECK(cudaMemcpy(hResultCsr, dResultCsr, result_bytes, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(hResultCsc, dResultCsc, result_bytes, cudaMemcpyDeviceToHost));

  // CPU reference check.
  REAL *hRef = (REAL *)malloc(result_bytes);
  reference_spmm<REAL>(hRef);

  double max_abs_err_csr = 0.0, max_abs_err_csc = 0.0;
  for (int idx = 0; idx < n * n; ++idx) {
    double d1 = (double)hResultCsr[idx] - (double)hRef[idx];
    double d2 = (double)hResultCsc[idx] - (double)hRef[idx];
    if (d1 < 0) d1 = -d1;
    if (d2 < 0) d2 = -d2;
    if (d1 > max_abs_err_csr) max_abs_err_csr = d1;
    if (d2 > max_abs_err_csc) max_abs_err_csc = d2;
  }

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (int idx = 0; idx < n * n; ++idx) fprintf(f, "%.17g\n", (double)hResultCsc[idx]);
  fclose(f);

  printf("csrScanOrderSpMM done: num_rows=%d, nnzA=%d, nnzB=%d, "
         "max_abs_error(csr_csr vs ref)=%.3e, max_abs_error(csc_csr vs ref)=%.3e -> %s\n",
         n, NNZ_A, NNZ_B, max_abs_err_csr, max_abs_err_csc, out);
  printf("%s\n", (max_abs_err_csr == 0.0 && max_abs_err_csc == 0.0) ? "PASS" : "FAIL");

  cudaFree(dPtrA); cudaFree(dIndicesA); cudaFree(dDataA);
  cudaFree(dPtrB_csr); cudaFree(dIndicesB_csr); cudaFree(dDataB_csr);
  cudaFree(dPtrB_csc); cudaFree(dIndicesB_csc); cudaFree(dDataB_csc);
  cudaFree(dResultCsr); cudaFree(dResultCsc);

  free(hPtrA); free(hIndicesA); free(hDataA);
  free(hPtrB_csr); free(hIndicesB_csr); free(hDataB_csr);
  free(hPtrB_csc); free(hIndicesB_csc); free(hDataB_csc);
  free(hResultCsr); free(hResultCsc); free(hRef);
  return 0;
}
