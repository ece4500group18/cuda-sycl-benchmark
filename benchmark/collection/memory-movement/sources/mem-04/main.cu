// coalescedAxpyDistribution: AXPY (y[i] += a*x[i]), cyclic (grid-stride)
// vs. block/chunked thread-to-element distribution.
//
// The two __global__ kernels below are reproduced, unmodified, from:
//   CUDAMicroBench (CoMem_AXPY/axpy_cudakernel.cu)
//   https://github.com/passlab/CUDAMicroBench
//   Copyright (c) 2021, University of North Carolina at Charlotte
//   and Lawrence Livermore National Security, LLC.
//   SPDX-License-Identifier: BSD-3-Clause
//
// Both kernels compute the same y[i] += a*x[i] update over the full index
// range, but map loop iterations to threads differently:
//
// "axpy_cudakernel_cyclic": thread `tid` handles indices
//   tid, tid+total_threads, tid+2*total_threads, ... -- a grid-stride loop.
//   At any given step, the 32 threads of a warp touch 32 *consecutive*
//   indices, i.e. one contiguous, coalesced 128/256-byte memory
//   transaction per warp.
//
// "axpy_cudakernel_block": thread `tid` handles one *contiguous chunk* of
//   `n / total_threads` indices, starting at `tid * block_size`. At any
//   given step, the 32 threads of a warp touch 32 indices separated by
//   `block_size` elements each -- a strided, uncoalesced access pattern
//   spread across up to 32 different memory segments per warp.
//
// Both kernels touch every index in [0, n) exactly once and compute the
// exact same per-element result; only *which global-memory addresses a
// warp accesses together* differs -- the canonical "memory layout /
// memory movement" illustration of why thread-to-data mapping determines
// coalescing.
//
// Deterministic inputs (replicated by reference.h):
//   x[i] = ((i % 17) - 8) * 0.25
//   y[i] = ((i % 23) - 11) * 0.5
// n = 1048576 (= 1024 * 256 * 4, a multiple of the 1024x256 = 262144
// total threads launched below, so `n / total_threads` divides evenly).
//
// Output: argv[1] (default output/cuda_output.txt), one double per line,
// y[] after the cyclic kernel's update.

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

#define REAL double

// --- begin: verbatim from CUDAMicroBench CoMem_AXPY/axpy_cudakernel.cu ---

/* block distribution of loop iteration */
__global__
void axpy_cudakernel_block(REAL* x, REAL* y, int n, REAL a) {
	int thread_num = threadIdx.x + blockIdx.x * blockDim.x;
	int total_threads = gridDim.x * blockDim.x;

	int block_size = n / total_threads; //dividable, TODO handle non-dividiable later

	int start_index = thread_num * block_size;
	int stop_index = start_index + block_size;
	int i;
        for (i=start_index; i<stop_index; i++) {
		if (i < n) y[i] += a*x[i];
	}
}

/* cyclic distribution of loop distribution */
__global__
void axpy_cudakernel_cyclic(REAL* x, REAL* y, int n, REAL a) {
	int thread_num = threadIdx.x + blockIdx.x * blockDim.x;
	int total_threads = gridDim.x * blockDim.x;

	int i;
	for (i=thread_num; i<n; i+=total_threads) {
		if (i < n) y[i] += a*x[i];
	}
}

// --- end: verbatim from CUDAMicroBench ---

int main(int argc, char **argv) {
  const long n = 1048576L;  // multiple of (1024 * 256) total threads
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  const size_t bytes = (size_t)n * sizeof(REAL);
  const REAL a = 123.456;

  REAL *hx = (REAL *)malloc(bytes);
  REAL *hy_block = (REAL *)malloc(bytes);
  REAL *hy_cyclic = (REAL *)malloc(bytes);

  for (long i = 0; i < n; ++i) {
    hx[i] = gen_x(i);
    hy_block[i] = gen_y(i);
    hy_cyclic[i] = gen_y(i);
  }

  REAL *dx, *dy_block, *dy_cyclic;
  CHECK(cudaMalloc(&dx, bytes));
  CHECK(cudaMalloc(&dy_block, bytes));
  CHECK(cudaMalloc(&dy_cyclic, bytes));

  CHECK(cudaMemcpy(dx, hx, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dy_block, hy_block, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dy_cyclic, hy_cyclic, bytes, cudaMemcpyHostToDevice));

  axpy_cudakernel_block<<<1024, 256>>>(dx, dy_block, (int)n, a);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  axpy_cudakernel_cyclic<<<1024, 256>>>(dx, dy_cyclic, (int)n, a);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  CHECK(cudaMemcpy(hy_block, dy_block, bytes, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(hy_cyclic, dy_cyclic, bytes, cudaMemcpyDeviceToHost));

  // CPU reference check.
  REAL *href = (REAL *)malloc(bytes);
  reference_axpy(hx, href, n, a);

  double max_abs_err_block = 0.0, max_abs_err_cyclic = 0.0;
  for (long i = 0; i < n; ++i) {
    double d1 = hy_block[i] - href[i];
    double d2 = hy_cyclic[i] - href[i];
    if (d1 < 0) d1 = -d1;
    if (d2 < 0) d2 = -d2;
    if (d1 > max_abs_err_block) max_abs_err_block = d1;
    if (d2 > max_abs_err_cyclic) max_abs_err_cyclic = d2;
  }

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (long i = 0; i < n; ++i) fprintf(f, "%.17g\n", hy_cyclic[i]);
  fclose(f);

  printf("coalescedAxpyDistribution done: n=%ld, "
         "max_abs_error(block vs ref)=%.3e, max_abs_error(cyclic vs ref)=%.3e -> %s\n",
         n, max_abs_err_block, max_abs_err_cyclic, out);
  printf("%s\n", (max_abs_err_block == 0.0 && max_abs_err_cyclic == 0.0) ? "PASS" : "FAIL");

  cudaFree(dx);
  cudaFree(dy_block);
  cudaFree(dy_cyclic);
  free(hx);
  free(hy_block);
  free(hy_cyclic);
  free(href);
  return 0;
}
