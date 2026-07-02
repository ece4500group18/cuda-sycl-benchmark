// dynamicSharedMinReduction: dynamically-sized extern __shared__
// tree-halving min-reduction per block vs. a naive single-thread linear
// scan over the same block's input.
//
// The "timedReduction" __global__ kernel below is reproduced from:
//   NVIDIA/cuda-samples, cpp/0_Introduction/clock/clock.cu
//   https://github.com/NVIDIA/cuda-samples/blob/master/cpp/0_Introduction/clock/clock.cu
//   Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//   SPDX-License-Identifier: BSD-3-Clause
//
// The only change from the upstream kernel body is the removal of the
// `clock_t *timer` parameter and the two `clock()` calls that recorded
// per-block start/end timestamps (and the `__syncthreads()` that existed
// solely to make the trailing timestamp include all reduction work) --
// that instrumentation is diagnostic-only (it measures wall-clock
// duration, not a value used in the reduction) and irrelevant to a
// deterministic CUDA-vs-SYCL output comparison. The reduction algorithm
// itself -- the `extern __shared__ float shared[]` dynamic shared-memory
// declaration, the two-elements-per-thread load, and the
// `for (d = blockDim.x; d > 0; d /= 2)` tree-halving loop -- is
// unmodified.
//
// "timedReduction": each block dynamically allocates 2*blockDim.x floats
//   of shared memory (sized at launch via the kernel's third `<<<>>>`
//   argument, i.e. `extern __shared__`, not a compile-time constant).
//   Thread `tid` loads two elements, `input[tid]` and
//   `input[tid + blockDim.x]`, into shared memory, then a loop with
//   `d` halving from `blockDim.x` down to 1 repeatedly compares
//   `shared[tid]` against `shared[tid + d]` (guarded by `tid < d` and
//   `__syncthreads()`) and keeps the smaller value -- a classic
//   tree-halving reduction, here computing a minimum instead of the more
//   common sum.
//
// "naiveMinReduction": a single thread per block (thread 0; the kernel
//   is launched with 1 thread per block) walks the same `n` input
//   elements in plain index order with a linear scan, keeping a running
//   minimum. No shared memory, no synchronization, no parallelism within
//   the block -- this is new code written for this repository, added to
//   give the dynamic-shared-memory tree reduction a trivially-correct,
//   easy-to-verify-by-inspection counterpart that isolates exactly the
//   tree/shared-memory control-flow technique being demonstrated.
//
// Both kernels compute the minimum of the exact same N = 2*NUM_THREADS
// input floats (every block reads from the same input buffer starting at
// offset 0, matching the upstream sample, which does not offset `input`
// by `blockIdx.x` either -- so all NUM_BLOCKS blocks are expected to
// produce the identical minimum). `min` over a fixed set of floats does
// not depend on comparison order, so both kernels' per-block outputs are
// expected to match the CPU reference exactly (max_abs_error == 0).
//
// Deterministic input (replicated by reference.h):
//   input[i] = ((i % 37) - 18) * 0.5f,  i in [0, 128)
// NUM_BLOCKS = 8, NUM_THREADS = 64 (each block reduces 2*64 = 128 floats)
//
// Output: argv[1] (default output/cuda_output.txt), NUM_BLOCKS lines,
// one float per line -- the per-block minimum computed by
// "timedReduction".

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

#define NUM_BLOCKS  8
#define NUM_THREADS 64

// --- begin: adapted from NVIDIA/cuda-samples cpp/0_Introduction/clock/clock.cu ---
// (clock_t timer parameter and clock() timestamp calls removed; the
// dynamic-shared-memory tree-halving reduction body is unmodified)

__global__ static void timedReduction(const float *input, float *output) {
    // __shared__ float shared[2 * blockDim.x];
    extern __shared__ float shared[];

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    // Copy input.
    shared[tid]              = input[tid];
    shared[tid + blockDim.x] = input[tid + blockDim.x];

    // Perform reduction to find minimum.
    for (int d = blockDim.x; d > 0; d /= 2) {
        __syncthreads();

        if (tid < d) {
            float f0 = shared[tid];
            float f1 = shared[tid + d];

            if (f1 < f0) {
                shared[tid] = f1;
            }
        }
    }

    // Write result.
    if (tid == 0)
        output[bid] = shared[0];
}

// --- end: adapted from NVIDIA/cuda-samples ---

// New code for this repository: a naive, single-thread-per-block linear
// scan computing the same per-block minimum, with no shared memory and
// no synchronization, for comparison against the dynamic-shared-memory
// tree reduction above.
__global__ void naiveMinReduction(const float *input, float *output, int n) {
    if (threadIdx.x == 0) {
        float m = input[0];
        for (int i = 1; i < n; ++i) {
            float v = input[i];
            if (v < m) m = v;
        }
        output[blockIdx.x] = m;
    }
}

int main(int argc, char **argv) {
  const int n = 2 * NUM_THREADS;  // 128 floats reduced per block
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";

  float input[n];
  for (int i = 0; i < n; ++i) input[i] = gen_input(i);

  float h_tree[NUM_BLOCKS];
  float h_naive[NUM_BLOCKS];

  float *dinput = NULL, *doutput_tree = NULL, *doutput_naive = NULL;
  CHECK(cudaMalloc((void **)&dinput, sizeof(float) * n));
  CHECK(cudaMalloc((void **)&doutput_tree, sizeof(float) * NUM_BLOCKS));
  CHECK(cudaMalloc((void **)&doutput_naive, sizeof(float) * NUM_BLOCKS));

  CHECK(cudaMemcpy(dinput, input, sizeof(float) * n, cudaMemcpyHostToDevice));

  timedReduction<<<NUM_BLOCKS, NUM_THREADS, sizeof(float) * 2 * NUM_THREADS>>>(
      dinput, doutput_tree);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  naiveMinReduction<<<NUM_BLOCKS, 1>>>(dinput, doutput_naive, n);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  CHECK(cudaMemcpy(h_tree, doutput_tree, sizeof(float) * NUM_BLOCKS, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(h_naive, doutput_naive, sizeof(float) * NUM_BLOCKS, cudaMemcpyDeviceToHost));

  // CPU reference check.
  float ref = reference_min(n);

  double max_abs_err_tree = 0.0, max_abs_err_naive = 0.0;
  for (int b = 0; b < NUM_BLOCKS; ++b) {
    double d1 = (double)h_tree[b] - (double)ref;
    double d2 = (double)h_naive[b] - (double)ref;
    if (d1 < 0) d1 = -d1;
    if (d2 < 0) d2 = -d2;
    if (d1 > max_abs_err_tree) max_abs_err_tree = d1;
    if (d2 > max_abs_err_naive) max_abs_err_naive = d2;
  }

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (int b = 0; b < NUM_BLOCKS; ++b) fprintf(f, "%.9g\n", h_tree[b]);
  fclose(f);

  printf("dynamicSharedMinReduction done: NUM_BLOCKS=%d, NUM_THREADS=%d, n=%d, "
         "max_abs_error(tree vs ref)=%.3e, max_abs_error(naive vs ref)=%.3e -> %s\n",
         NUM_BLOCKS, NUM_THREADS, n, max_abs_err_tree, max_abs_err_naive, out);
  printf("%s\n", (max_abs_err_tree == 0.0 && max_abs_err_naive == 0.0) ? "PASS" : "FAIL");

  cudaFree(dinput);
  cudaFree(doutput_tree);
  cudaFree(doutput_naive);
  return 0;
}
