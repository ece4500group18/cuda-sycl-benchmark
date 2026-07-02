// warpDivergence: branchy vs. branch-free per-element kernel computing
// the same parity-dependent linear combination.
//
// The two __global__ kernels below are reproduced, unmodified, from:
//   CUDAMicroBench (WarpDivRedux/warpDivergenceTest_cudakernel.cu)
//   https://github.com/passlab/CUDAMicroBench
//   Copyright (c) 2021, University of North Carolina at Charlotte
//   and Lawrence Livermore National Security, LLC.
//   SPDX-License-Identifier: BSD-3-Clause
//
// Both kernels compute, for each index i:
//   if (i % 2 == 0): z[i] = 2*x[i] + 3*y[i]
//   else:            z[i] = 3*x[i] + 2*y[i]
//
// "warpDivergence": uses an `if/else` on `tid % 2` to choose the
//   coefficients `a`,`b`. Within a warp, even- and odd-indexed threads
//   take *different control-flow paths*, so the warp executes both
//   branches serially (warp divergence).
//
// "noWarpDivergence": computes `even = (tid % 2 == 0)` as 0/1 and
//   blends the coefficients arithmetically
//   (`a = even*2 + (1-even)*3`, `b = even*3 + (1-even)*2`) with no
//   data-dependent branch -- every thread in the warp executes the
//   same instructions (branch-free / divergence-free).
//
// Both kernels compute *exactly the same per-element result*; only the
// control-flow structure differs. This is a "simple but not trivial"
// kernel: the math is a one-line linear combination, but choosing how
// to express the parity-dependent coefficient selection (branch vs.
// arithmetic blend) is a classic SIMT optimization technique.
//
// Deterministic inputs (replicated by reference.h):
//   x[i] = ((i % 17) - 8) * 0.25f
//   y[i] = ((i % 23) - 11) * 0.5f
// n = 1024000
//
// Output: argv[1] (default output/cuda_output.txt), one float per
// line, z[] computed by the branch-free kernel (noWarpDivergence).

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

// --- begin: verbatim from CUDAMicroBench WarpDivRedux/warpDivergenceTest_cudakernel.cu ---

__global__ void warpDivergence(float *x, float *y, float *z) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a = 2, b = 3;
    if (tid % 2 != 0) {
        a = 3;
        b = 2;
    }
    z[tid] = a * x[tid] + b * y[tid];
}

__global__ void noWarpDivergence(float *x, float *y, float *z) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int even = tid % 2 == 0;
    float a = even * 2 + (1 - even) * 3;
    float b = even * 3 + (1 - even) * 2;
    z[tid] = a * x[tid] + b * y[tid];
}

// --- end: verbatim from CUDAMicroBench ---

int main(int argc, char **argv) {
  const int n = 1024000;  // VEC_LEN
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  const size_t bytes = (size_t)n * sizeof(float);

  float *hx = (float *)malloc(bytes);
  float *hy = (float *)malloc(bytes);
  float *hz_div = (float *)malloc(bytes);
  float *hz_nodiv = (float *)malloc(bytes);

  for (int i = 0; i < n; ++i) {
    hx[i] = gen_x(i);
    hy[i] = gen_y(i);
  }

  float *dx, *dy, *dz_div, *dz_nodiv;
  CHECK(cudaMalloc(&dx, bytes));
  CHECK(cudaMalloc(&dy, bytes));
  CHECK(cudaMalloc(&dz_div, bytes));
  CHECK(cudaMalloc(&dz_nodiv, bytes));

  CHECK(cudaMemcpy(dx, hx, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dy, hy, bytes, cudaMemcpyHostToDevice));

  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;

  warpDivergence<<<blocks, threads>>>(dx, dy, dz_div);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  noWarpDivergence<<<blocks, threads>>>(dx, dy, dz_nodiv);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  CHECK(cudaMemcpy(hz_div, dz_div, bytes, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(hz_nodiv, dz_nodiv, bytes, cudaMemcpyDeviceToHost));

  // CPU reference check.
  float *href = (float *)malloc(bytes);
  reference_z(hx, hy, href, n);

  float max_abs_err_div = 0.0f, max_abs_err_nodiv = 0.0f;
  for (int i = 0; i < n; ++i) {
    float d1 = hz_div[i] - href[i];
    float d2 = hz_nodiv[i] - href[i];
    if (d1 < 0) d1 = -d1;
    if (d2 < 0) d2 = -d2;
    if (d1 > max_abs_err_div) max_abs_err_div = d1;
    if (d2 > max_abs_err_nodiv) max_abs_err_nodiv = d2;
  }

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (int i = 0; i < n; ++i) fprintf(f, "%.9g\n", hz_nodiv[i]);
  fclose(f);

  printf("warpDivergence done: n=%d, "
         "max_abs_error(branchy vs ref)=%.3e, max_abs_error(branchfree vs ref)=%.3e -> %s\n",
         n, max_abs_err_div, max_abs_err_nodiv, out);
  printf("%s\n", (max_abs_err_div == 0.0f && max_abs_err_nodiv == 0.0f) ? "PASS" : "FAIL");

  cudaFree(dx);
  cudaFree(dy);
  cudaFree(dz_div);
  cudaFree(dz_nodiv);
  free(hx);
  free(hy);
  free(hz_div);
  free(hz_nodiv);
  free(href);
  return 0;
}
