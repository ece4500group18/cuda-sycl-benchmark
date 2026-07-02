// sparseDenseSpmvLayout: sparse matrix-vector product, y = M * x, computed
// via a compressed CSR gather vs. the identical math swept over the full
// dense row-major matrix with a zero-skip guard.
//
// The two __global__ kernels below are reproduced, unmodified, from:
//   CUDAMicroBench (MiniTransfer_SpMV/SpMV_cudakernel.cu)
//   https://github.com/passlab/CUDAMicroBench
//   Copyright (c) 2021, University of North Carolina at Charlotte
//   and Lawrence Livermore National Security, LLC.
//   SPDX-License-Identifier: BSD-3-Clause
//
// Both kernels compute one thread per row of y = M * x for the same
// num_rows x num_rows matrix M and the same vector x:
//
// "spmv_csr": row `row`'s dot product is `sum_{n=row_start}^{row_end-1}
//   data[n] * x[indices[n]]`, i.e. it *gathers* only the row's nonzero
//   values and their column indices from three compact arrays (`ptr`,
//   `indices`, `data`) sized proportional to the number of nonzeros --
//   the classic compressed-sparse-row storage layout.
//
// "spmv_dense_check_and_compute": row `i`'s dot product sweeps the
//   *entire* row of the full num_rows x num_rows dense matrix,
//   `temp += matrix[i*num_rows+j] * vector[j]` for every `j` in
//   `[0, num_rows)`, skipping the multiply-add only when the matrix
//   entry read back is exactly `0.0`.
//
// This directory's inputs are built so both kernels visit the row's
// nonzeros in the exact same ascending-column order and every value
// involved is a small integer, so the two kernels are guaranteed to
// produce byte-identical output (see reference.h for why).
//
// Deterministic inputs (built on the host from reference.h's formulas):
//   num_rows = 256, 5 nonzeros/row at columns (row*7 + k*13) % 256
//   (k = 0..4, always 5 distinct columns per row); nonzero value
//   val(row,k) in {-4,-3,-2,-1,1,2,3,4} (never 0); x[i] = (i % 11) - 5.
// Launch: 1 block of 256 threads, 1 thread per row (num_rows = 256).
//
// Output: argv[1] (default output/cuda_output.txt), one float per line,
// y = M*x computed by spmv_csr.

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

// REAL is #define'd to `float` in the upstream SpMV.h; reproduced here so
// this file stays a single self-contained translation unit.
#define REAL float

// --- begin: verbatim from CUDAMicroBench MiniTransfer_SpMV/SpMV_cudakernel.cu ---

//2 csr-format of the matrix, copy csr formatted-matrix via discrete memory
__global__ void spmv_csr(const int num_rows, const int *ptr, const int * indices, const REAL *data, const REAL * x, REAL *y)
{
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < num_rows){
        REAL dot = 0;
        int row_start = ptr[row];
        int row_end = ptr[row+1];

        for(int n = row_start; n < row_end; n++){
           dot += data[n] * x[indices[n]];
        }
        y[row] = dot;
    }
}

__global__ void spmv_dense_check_and_compute(REAL* matrix, REAL* vector, REAL *y, int num_rows)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_rows) {
        y[i] = 0;
        REAL temp = 0.0;
        for (int j = 0; j < num_rows; j++) {
            if (matrix[i * num_rows + j] != 0.0)
                temp += matrix[i * num_rows + j] * vector[j];
        }
        y[i] = temp;
    }
}

// --- end: verbatim from CUDAMicroBench ---

int main(int argc, char **argv) {
  const int num_rows = NUM_ROWS;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";

  // Host input vector.
  REAL *hx = (REAL *)malloc(num_rows * sizeof(REAL));
  double *hx_d = (double *)malloc(num_rows * sizeof(double));
  for (int i = 0; i < num_rows; ++i) {
    hx_d[i] = gen_x(i);
    hx[i] = (REAL)hx_d[i];
  }

  // Host CSR arrays: ptr[num_rows+1], indices[NNZ], data[NNZ], built from
  // the same (column, value) pairs used for the dense matrix below, sorted
  // ascending by column per row (see reference.h).
  int *h_ptr = (int *)malloc((num_rows + 1) * sizeof(int));
  int *h_indices = (int *)malloc(NNZ * sizeof(int));
  REAL *h_data = (REAL *)malloc(NNZ * sizeof(REAL));

  // Host dense matrix, row-major, num_rows x num_rows, zero-initialized
  // except at the same nonzero (row, column) positions as the CSR arrays.
  REAL *h_matrix = (REAL *)calloc((size_t)num_rows * num_rows, sizeof(REAL));

  for (int row = 0; row < num_rows; ++row) {
    Nz nz[NNZ_PER_ROW];
    row_nonzeros_sorted(row, nz);
    h_ptr[row] = row * NNZ_PER_ROW;
    for (int k = 0; k < NNZ_PER_ROW; ++k) {
      int idx = row * NNZ_PER_ROW + k;
      h_indices[idx] = nz[k].col;
      h_data[idx] = (REAL)nz[k].val;
      h_matrix[(size_t)row * num_rows + nz[k].col] = (REAL)nz[k].val;
    }
  }
  h_ptr[num_rows] = NNZ;

  // Device buffers for the CSR path.
  int *d_ptr, *d_indices;
  REAL *d_data, *d_x_csr, *d_y_csr;
  CHECK(cudaMalloc(&d_ptr, (num_rows + 1) * sizeof(int)));
  CHECK(cudaMalloc(&d_indices, NNZ * sizeof(int)));
  CHECK(cudaMalloc(&d_data, NNZ * sizeof(REAL)));
  CHECK(cudaMalloc(&d_x_csr, num_rows * sizeof(REAL)));
  CHECK(cudaMalloc(&d_y_csr, num_rows * sizeof(REAL)));

  CHECK(cudaMemcpy(d_ptr, h_ptr, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_indices, h_indices, NNZ * sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_data, h_data, NNZ * sizeof(REAL), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_x_csr, hx, num_rows * sizeof(REAL), cudaMemcpyHostToDevice));

  // Device buffers for the dense path.
  REAL *d_matrix, *d_x_dense, *d_y_dense;
  CHECK(cudaMalloc(&d_matrix, (size_t)num_rows * num_rows * sizeof(REAL)));
  CHECK(cudaMalloc(&d_x_dense, num_rows * sizeof(REAL)));
  CHECK(cudaMalloc(&d_y_dense, num_rows * sizeof(REAL)));

  CHECK(cudaMemcpy(d_matrix, h_matrix, (size_t)num_rows * num_rows * sizeof(REAL), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_x_dense, hx, num_rows * sizeof(REAL), cudaMemcpyHostToDevice));

  spmv_csr<<<1, 256>>>(num_rows, d_ptr, d_indices, d_data, d_x_csr, d_y_csr);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  spmv_dense_check_and_compute<<<1, 256>>>(d_matrix, d_x_dense, d_y_dense, num_rows);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  REAL *hy_csr = (REAL *)malloc(num_rows * sizeof(REAL));
  REAL *hy_dense = (REAL *)malloc(num_rows * sizeof(REAL));
  CHECK(cudaMemcpy(hy_csr, d_y_csr, num_rows * sizeof(REAL), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(hy_dense, d_y_dense, num_rows * sizeof(REAL), cudaMemcpyDeviceToHost));

  // CPU reference check.
  double *href = (double *)malloc(num_rows * sizeof(double));
  reference_spmv(hx_d, href);

  double max_abs_err_csr = 0.0, max_abs_err_dense = 0.0;
  for (int i = 0; i < num_rows; ++i) {
    double d1 = (double)hy_csr[i] - href[i];
    double d2 = (double)hy_dense[i] - href[i];
    if (d1 < 0) d1 = -d1;
    if (d2 < 0) d2 = -d2;
    if (d1 > max_abs_err_csr) max_abs_err_csr = d1;
    if (d2 > max_abs_err_dense) max_abs_err_dense = d2;
  }

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (int i = 0; i < num_rows; ++i) fprintf(f, "%.9g\n", hy_csr[i]);
  fclose(f);

  printf("sparseDenseSpmvLayout done: num_rows=%d, nnz=%d, "
         "max_abs_error(csr vs ref)=%.3e, max_abs_error(dense vs ref)=%.3e -> %s\n",
         num_rows, NNZ, max_abs_err_csr, max_abs_err_dense, out);
  printf("%s\n", (max_abs_err_csr == 0.0 && max_abs_err_dense == 0.0) ? "PASS" : "FAIL");

  cudaFree(d_ptr);
  cudaFree(d_indices);
  cudaFree(d_data);
  cudaFree(d_x_csr);
  cudaFree(d_y_csr);
  cudaFree(d_matrix);
  cudaFree(d_x_dense);
  cudaFree(d_y_dense);
  free(hx);
  free(hx_d);
  free(h_ptr);
  free(h_indices);
  free(h_data);
  free(h_matrix);
  free(hy_csr);
  free(hy_dense);
  free(href);
  return 0;
}
