// memAlign: aligned vs. misaligned global-memory access for DAXPY
// (y[i] += a*x[i]).
//
// The two __global__ kernels below are reproduced, unmodified, from:
//   CUDAMicroBench (MemAlign/axpy_cudakernel.cu)
//   https://github.com/passlab/CUDAMicroBench
//   Copyright (c) 2021, University of North Carolina at Charlotte
//   and Lawrence Livermore National Security, LLC.
//   SPDX-License-Identifier: BSD-3-Clause
//
// "aligned":    thread i writes y[i], i in [1, n)        -> 128B-aligned
//               128-byte transactions per warp (for double, 32 lanes)
// "misaligned": thread i writes y[i+1], i in [0, n-1)     -> every warp's
//               transaction is shifted by one element (8 bytes), forcing
//               an extra memory transaction per warp on the same range.
//
// Both kernels compute the exact same mathematical result
// (y[i] += a*x[i] for i in [1,n)); only the *memory access pattern*
// differs. This isolates the cost of misaligned global memory access.
//
// Deterministic inputs (replicated by reference.h):
//   x[i] = ((i % 17) - 8) * 0.25
//   y[i] = ((i % 23) - 11) * 0.5
// n = 1048576, a = 123.456
//
// Output: argv[1] (default output/cuda_output.txt), one double per line,
// containing the final y array after running the ALIGNED kernel (the
// kernel whose result is meant to be compared against the SYCL port).

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

// --- begin: verbatim from CUDAMicroBench MemAlign/axpy_cudakernel.cu ---

__global__
void axpy_cudakernel_1perThread(double* x, double* y, int n, double a)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i > 0 && i < n) y[i] += a*x[i];
}

__global__
void axpy_cudakernel_1perThread_misaligned(double* x, double* y, int n, double a)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x + 1;
    if (i < n) y[i] += a*x[i];
}

// --- end: verbatim from CUDAMicroBench ---

int main(int argc, char **argv) {
  const long n = 1048576;       // VEC_LEN, same order of magnitude as upstream
  const double a = 123.456;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  const size_t bytes = (size_t)n * sizeof(double);

  double *hx = (double *)malloc(bytes);
  double *hy_aligned = (double *)malloc(bytes);
  double *hy_misaligned = (double *)malloc(bytes);

  for (long i = 0; i < n; ++i) {
    hx[i] = gen_x(i);
    double y0 = gen_y(i);
    hy_aligned[i] = y0;
    hy_misaligned[i] = y0;
  }

  double *dx, *dy_aligned, *dy_misaligned;
  CHECK(cudaMalloc(&dx, bytes));
  CHECK(cudaMalloc(&dy_aligned, bytes));
  CHECK(cudaMalloc(&dy_misaligned, bytes));

  CHECK(cudaMemcpy(dx, hx, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dy_aligned, hy_aligned, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dy_misaligned, hy_misaligned, bytes, cudaMemcpyHostToDevice));

  const int threads = 256;
  const int blocks = (int)((n + threads - 1) / threads);

  // Aligned version (this is the result written to argv[1]).
  axpy_cudakernel_1perThread<<<blocks, threads>>>(dx, dy_aligned, (int)n, a);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  // Misaligned version (computed for completeness / future timing
  // comparisons; not written to argv[1]).
  axpy_cudakernel_1perThread_misaligned<<<blocks, threads>>>(dx, dy_misaligned, (int)n, a);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  CHECK(cudaMemcpy(hy_aligned, dy_aligned, bytes, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(hy_misaligned, dy_misaligned, bytes, cudaMemcpyDeviceToHost));

  // CPU reference check (informational; does not gate the file output).
  double *href = (double *)malloc(bytes);
  for (long i = 0; i < n; ++i) href[i] = gen_y(i);
  reference_axpy(hx, href, n, a);

  double max_abs_err = 0.0;
  for (long i = 0; i < n; ++i) {
    double d = hy_aligned[i] - href[i];
    if (d < 0) d = -d;
    if (d > max_abs_err) max_abs_err = d;
  }

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (long i = 0; i < n; ++i) fprintf(f, "%.17g\n", hy_aligned[i]);
  fclose(f);

  printf("memAlign done: n=%ld, max_abs_error(aligned vs ref)=%.3e -> %s\n",
         n, max_abs_err, out);
  printf("%s\n", (max_abs_err == 0.0) ? "PASS" : "FAIL");

  cudaFree(dx);
  cudaFree(dy_aligned);
  cudaFree(dy_misaligned);
  free(hx);
  free(hy_aligned);
  free(hy_misaligned);
  free(href);
  return 0;
}
