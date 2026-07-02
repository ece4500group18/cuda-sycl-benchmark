// unifiedMemoryAccess: strided gather AXPY-like read (y[j] = a*x[j*stride]),
// discrete device memory (explicit cudaMalloc + cudaMemcpy) vs. CUDA
// Unified/managed memory (cudaMallocManaged, demand-paged migration).
//
// The __global__ kernel below is reproduced, unmodified, from:
//   CUDAMicroBench (UniMem/LowAccessDensityTest_cuda.cu)
//   https://github.com/passlab/CUDAMicroBench
//   Copyright (c) 2021, University of North Carolina at Charlotte
//   and Lawrence Livermore National Security, LLC.
//   SPDX-License-Identifier: BSD-3-Clause
//
// The *same* kernel is launched twice against two differently-provisioned
// input buffers:
//
// - "discrete": `x` lives in a plain `cudaMalloc`'d device buffer,
//   populated by one explicit, up-front `cudaMemcpyHostToDevice` of the
//   entire array before the kernel runs.
// - "managed": `x` lives in a `cudaMallocManaged` buffer. The host writes
//   into it directly (a plain `memcpy`, no CUDA API call) and the kernel
//   reads it directly; any page not yet resident on the device is faulted
//   in and migrated on first touch by the CUDA runtime instead of being
//   copied up front by the host.
//
// Because the kernel and the input values are identical in both cases,
// the two runs must produce byte-identical output; what differs is *how
// the input array's pages get from host to device memory* -- an explicit,
// synchronous bulk copy vs. on-demand page migration -- the canonical
// "memory movement" contrast at the host/device boundary.
//
// Deterministic input (replicated by reference.h):
//   x[i] = ((i % 17) - 8) * 0.25f, i in [0, n)
// n = 1048576 (2^20), stride = 16 (n is a multiple of stride).
//
// Output: argv[1] (default output/cuda_output.txt), one float per line,
// y[] = a*x[j*stride] for j in [0, n/stride), computed via the managed
// buffer.

#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// --- begin: verbatim from CUDAMicroBench UniMem/LowAccessDensityTest_cuda.cu ---

__global__
void
LowAccessDensityTest_cudakernel(REAL* x, REAL* y, int n, REAL a, int stride)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < (n/stride)) y[i] = a*x[i*stride];
}

// --- end: verbatim from CUDAMicroBench ---

int main(int argc, char **argv) {
  const long n = 1048576L;   // 2^20
  const int stride = 16;
  const long n_out = n / stride;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  const REAL a = 123.456f;

  const size_t in_bytes = (size_t)n * sizeof(REAL);
  const size_t out_bytes = (size_t)n_out * sizeof(REAL);

  REAL *hx = (REAL *)malloc(in_bytes);
  for (long i = 0; i < n; ++i) hx[i] = gen_x(i);

  // --- discrete memory path: explicit cudaMalloc + cudaMemcpy H2D ---
  REAL *dx_discrete, *dy_discrete;
  CHECK(cudaMalloc(&dx_discrete, in_bytes));
  CHECK(cudaMalloc(&dy_discrete, out_bytes));
  CHECK(cudaMemcpy(dx_discrete, hx, in_bytes, cudaMemcpyHostToDevice));

  LowAccessDensityTest_cudakernel<<<(int)((n + 255) / 256), 256>>>(
      dx_discrete, dy_discrete, (int)n, a, stride);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  REAL *hy_discrete = (REAL *)malloc(out_bytes);
  CHECK(cudaMemcpy(hy_discrete, dy_discrete, out_bytes, cudaMemcpyDeviceToHost));

  // --- managed memory path: cudaMallocManaged, host memcpy, demand paging ---
  REAL *x_managed;
  CHECK(cudaMallocManaged(&x_managed, in_bytes));
  memcpy(x_managed, hx, in_bytes);  // host write, no CUDA API call

  REAL *dy_managed;
  CHECK(cudaMalloc(&dy_managed, out_bytes));

  LowAccessDensityTest_cudakernel<<<(int)((n + 255) / 256), 256>>>(
      x_managed, dy_managed, (int)n, a, stride);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  REAL *hy_managed = (REAL *)malloc(out_bytes);
  CHECK(cudaMemcpy(hy_managed, dy_managed, out_bytes, cudaMemcpyDeviceToHost));

  // CPU reference check.
  REAL *href = (REAL *)malloc(out_bytes);
  reference_strided_axpy(hx, href, n, a, stride);

  double max_abs_err_discrete = 0.0, max_abs_err_managed = 0.0;
  for (long j = 0; j < n_out; ++j) {
    double d1 = (double)hy_discrete[j] - (double)href[j];
    double d2 = (double)hy_managed[j] - (double)href[j];
    if (d1 < 0) d1 = -d1;
    if (d2 < 0) d2 = -d2;
    if (d1 > max_abs_err_discrete) max_abs_err_discrete = d1;
    if (d2 > max_abs_err_managed) max_abs_err_managed = d2;
  }

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (long j = 0; j < n_out; ++j) fprintf(f, "%.9g\n", hy_managed[j]);
  fclose(f);

  printf("unifiedMemoryAccess done: n=%ld, stride=%d, "
         "max_abs_error(discrete vs ref)=%.3e, max_abs_error(managed vs ref)=%.3e -> %s\n",
         n, stride, max_abs_err_discrete, max_abs_err_managed, out);
  printf("%s\n", (max_abs_err_discrete == 0.0 && max_abs_err_managed == 0.0) ? "PASS" : "FAIL");

  cudaFree(dx_discrete);
  cudaFree(dy_discrete);
  cudaFree(x_managed);
  cudaFree(dy_managed);
  free(hx);
  free(hy_discrete);
  free(hy_managed);
  free(href);
  return 0;
}
