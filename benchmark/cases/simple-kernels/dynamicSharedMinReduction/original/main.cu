// dynamicSharedMinReduction: dynamically-sized extern __shared__ tree-halving
// min-reduction per block, paired with a naive single-thread linear scan over
// the same input.
//
// The timedReduction __global__ kernel below is adapted from:
//   NVIDIA/cuda-samples, cpp/0_Introduction/clock/clock.cu
//   Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//   SPDX-License-Identifier: BSD-3-Clause
// The only change from upstream is removal of the `clock_t *timer` parameter
// and its two clock() timestamp calls (diagnostic-only). The reduction body --
// the extern __shared__ float declaration, two-elements-per-thread load, and
// the `for (d = blockDim.x; d > 0; d /= 2)` tree-halving loop -- is unmodified.
//
// naiveMinReduction is new code: one thread per block linearly scans the same n
// elements for the minimum, as a trivially-correct counterpart.
//
// Deterministic input (reproduced by tests/verify.py):
//   input[i] = ((i % 37) - 18) * 0.5f,  i in [0, 128)
// NUM_BLOCKS = 8, NUM_THREADS = 64 (each block reduces 2*64 = 128 floats; all
// blocks read from offset 0, so every block yields the same minimum).
//
// Output: argv[1] (default output/cuda_output.txt), 8 lines, per-block minimum
// from timedReduction.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(call)                                                          \
  do {                                                                       \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), \
              __FILE__, __LINE__);                                           \
      return 2;                                                              \
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

// New code for this repository: a naive single-thread-per-block linear scan
// computing the same per-block minimum, for comparison against the
// dynamic-shared-memory tree reduction above.
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

// Deterministic input generator (new host code; mirrored in tests/verify.py).
static inline float gen_input(long i) { return (float)((i % 37) - 18) * 0.5f; }

int main(int argc, char **argv) {
  const int n = 2 * NUM_THREADS;  // 128 floats reduced per block
  const char *out = (argc > 1) ? argv[1] : "output/cuda_output.txt";

  float input[n];
  for (int i = 0; i < n; ++i) input[i] = gen_input(i);

  float h_tree[NUM_BLOCKS];

  float *dinput = NULL, *doutput_tree = NULL, *doutput_naive = NULL;
  CHECK(cudaMalloc((void **)&dinput, sizeof(float) * n));
  CHECK(cudaMalloc((void **)&doutput_tree, sizeof(float) * NUM_BLOCKS));
  CHECK(cudaMalloc((void **)&doutput_naive, sizeof(float) * NUM_BLOCKS));

  CHECK(cudaMemcpy(dinput, input, sizeof(float) * n, cudaMemcpyHostToDevice));

  timedReduction<<<NUM_BLOCKS, NUM_THREADS, sizeof(float) * 2 * NUM_THREADS>>>(
      dinput, doutput_tree);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  // run the naive counterpart too (same minimum); result not written out
  naiveMinReduction<<<NUM_BLOCKS, 1>>>(dinput, doutput_naive, n);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  CHECK(cudaMemcpy(h_tree, doutput_tree, sizeof(float) * NUM_BLOCKS, cudaMemcpyDeviceToHost));

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (int b = 0; b < NUM_BLOCKS; ++b) fprintf(f, "%.9g\n", h_tree[b]);
  fclose(f);

  cudaFree(dinput);
  cudaFree(doutput_tree);
  cudaFree(doutput_naive);
  return 0;
}
