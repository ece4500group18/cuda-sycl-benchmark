// fp16PackedScalarProduct: packed half2 SIMD dot product via explicit
// __hfma2/__hadd2 intrinsics vs. via native half2 operator+/operator* overloads,
// both doing the identical grid-stride multiply-accumulate + shared-memory tree
// reduction.
//
// The two __global__ kernels and the two __forceinline__ __device__ reduction
// helpers below are reproduced, unmodified, from:
//   NVIDIA/cuda-samples, cpp/0_Introduction/fp16ScalarProduct/fp16ScalarProduct.cu
//   Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//   SPDX-License-Identifier: BSD-3-Clause
//
// Deterministic inputs (reproduced by tests/verify.py), replacing the sample's
// rand()-seeded generateInput(), chosen so every intermediate fp16 sum is an
// exactly-representable integer (fp16 is exact on [-2048, 2048]):
//   a[i].x = a[i].y = i % 4
//   b[i].x = b[i].y = i % 2
// NUM_OF_BLOCKS = NUM_OF_THREADS = 128, size = 128*128*16 = 262144.
//
// Output: argv[1] (default output/cuda_output.txt), 128 lines, one per-block
// float dot-product partial from scalarProductKernel_intrinsics.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CHECK(call)                                                          \
  do {                                                                       \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), \
              __FILE__, __LINE__);                                           \
      return 2;                                                              \
    }                                                                        \
  } while (0)

#define NUM_OF_BLOCKS  128
#define NUM_OF_THREADS 128

// --- begin: verbatim from NVIDIA/cuda-samples fp16ScalarProduct.cu ---

__forceinline__ __device__ void reduceInShared_intrinsics(half2 *const v)
{
    if (threadIdx.x < 64)
        v[threadIdx.x] = __hadd2(v[threadIdx.x], v[threadIdx.x + 64]);
    __syncthreads();
    if (threadIdx.x < 32)
        v[threadIdx.x] = __hadd2(v[threadIdx.x], v[threadIdx.x + 32]);
    __syncthreads();
    if (threadIdx.x < 16)
        v[threadIdx.x] = __hadd2(v[threadIdx.x], v[threadIdx.x + 16]);
    __syncthreads();
    if (threadIdx.x < 8)
        v[threadIdx.x] = __hadd2(v[threadIdx.x], v[threadIdx.x + 8]);
    __syncthreads();
    if (threadIdx.x < 4)
        v[threadIdx.x] = __hadd2(v[threadIdx.x], v[threadIdx.x + 4]);
    __syncthreads();
    if (threadIdx.x < 2)
        v[threadIdx.x] = __hadd2(v[threadIdx.x], v[threadIdx.x + 2]);
    __syncthreads();
    if (threadIdx.x < 1)
        v[threadIdx.x] = __hadd2(v[threadIdx.x], v[threadIdx.x + 1]);
    __syncthreads();
}

__forceinline__ __device__ void reduceInShared_native(half2 *const v)
{
    if (threadIdx.x < 64)
        v[threadIdx.x] = v[threadIdx.x] + v[threadIdx.x + 64];
    __syncthreads();
    if (threadIdx.x < 32)
        v[threadIdx.x] = v[threadIdx.x] + v[threadIdx.x + 32];
    __syncthreads();
    if (threadIdx.x < 16)
        v[threadIdx.x] = v[threadIdx.x] + v[threadIdx.x + 16];
    __syncthreads();
    if (threadIdx.x < 8)
        v[threadIdx.x] = v[threadIdx.x] + v[threadIdx.x + 8];
    __syncthreads();
    if (threadIdx.x < 4)
        v[threadIdx.x] = v[threadIdx.x] + v[threadIdx.x + 4];
    __syncthreads();
    if (threadIdx.x < 2)
        v[threadIdx.x] = v[threadIdx.x] + v[threadIdx.x + 2];
    __syncthreads();
    if (threadIdx.x < 1)
        v[threadIdx.x] = v[threadIdx.x] + v[threadIdx.x + 1];
    __syncthreads();
}

__global__ void
scalarProductKernel_intrinsics(half2 const *const a, half2 const *const b, float *const results, size_t const size)
{
    const int        stride = gridDim.x * blockDim.x;
    __shared__ half2 shArray[NUM_OF_THREADS];

    shArray[threadIdx.x] = __float2half2_rn(0.f);
    half2 value          = __float2half2_rn(0.f);

    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < size; i += stride) {
        value = __hfma2(a[i], b[i], value);
    }

    shArray[threadIdx.x] = value;
    __syncthreads();
    reduceInShared_intrinsics(shArray);

    if (threadIdx.x == 0) {
        half2 result        = shArray[0];
        float f_result      = __low2float(result) + __high2float(result);
        results[blockIdx.x] = f_result;
    }
}

__global__ void
scalarProductKernel_native(half2 const *const a, half2 const *const b, float *const results, size_t const size)
{
    const int        stride = gridDim.x * blockDim.x;
    __shared__ half2 shArray[NUM_OF_THREADS];

    half2 value(0.f, 0.f);
    shArray[threadIdx.x] = value;

    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < size; i += stride) {
        value = a[i] * b[i] + value;
    }

    shArray[threadIdx.x] = value;
    __syncthreads();
    reduceInShared_native(shArray);

    if (threadIdx.x == 0) {
        half2 result        = shArray[0];
        float f_result      = (float)result.y + (float)result.x;
        results[blockIdx.x] = f_result;
    }
}

// --- end: verbatim from NVIDIA/cuda-samples ---

// Deterministic input generators (new host code; mirrored in tests/verify.py).
static inline float gen_a_lane(size_t i) { return (float)(i % 4); }
static inline float gen_b_lane(size_t i) { return (float)(i % 2); }

int main(int argc, char **argv) {
  const size_t size = (size_t)NUM_OF_BLOCKS * NUM_OF_THREADS * 16;  // 262144
  const char *out = (argc > 1) ? argv[1] : "output/cuda_output.txt";

  half2 *hA = (half2 *)malloc(size * sizeof(half2));
  half2 *hB = (half2 *)malloc(size * sizeof(half2));
  for (size_t i = 0; i < size; ++i) {
    float av = gen_a_lane(i);
    float bv = gen_b_lane(i);
    hA[i].x = av; hA[i].y = av;
    hB[i].x = bv; hB[i].y = bv;
  }

  half2 *dA, *dB;
  float *dResultsNative, *dResultsIntrinsics;
  CHECK(cudaMalloc(&dA, size * sizeof(half2)));
  CHECK(cudaMalloc(&dB, size * sizeof(half2)));
  CHECK(cudaMalloc(&dResultsNative, NUM_OF_BLOCKS * sizeof(float)));
  CHECK(cudaMalloc(&dResultsIntrinsics, NUM_OF_BLOCKS * sizeof(float)));

  CHECK(cudaMemcpy(dA, hA, size * sizeof(half2), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dB, hB, size * sizeof(half2), cudaMemcpyHostToDevice));

  scalarProductKernel_native<<<NUM_OF_BLOCKS, NUM_OF_THREADS>>>(dA, dB, dResultsNative, size);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  scalarProductKernel_intrinsics<<<NUM_OF_BLOCKS, NUM_OF_THREADS>>>(dA, dB, dResultsIntrinsics, size);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  float *resultsIntrinsics = (float *)malloc(NUM_OF_BLOCKS * sizeof(float));
  CHECK(cudaMemcpy(resultsIntrinsics, dResultsIntrinsics, NUM_OF_BLOCKS * sizeof(float), cudaMemcpyDeviceToHost));

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (int b = 0; b < NUM_OF_BLOCKS; ++b) fprintf(f, "%.9g\n", (double)resultsIntrinsics[b]);
  fclose(f);

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dResultsNative);
  cudaFree(dResultsIntrinsics);
  free(hA);
  free(hB);
  free(resultsIntrinsics);
  return 0;
}
