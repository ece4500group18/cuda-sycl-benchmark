// fp16PackedScalarProduct: packed half2 SIMD dot product computed via
// explicit __hfma2/__hadd2 PTX intrinsics vs. via native half2
// operator+/operator* overloads, both doing the identical grid-stride
// multiply-accumulate followed by the identical shared-memory tree
// reduction.
//
// The four functions below (two __global__ kernels and the two
// __forceinline__ __device__ tree-reduction helpers they call) are
// reproduced, unmodified, from:
//   NVIDIA/cuda-samples,
//   cpp/0_Introduction/fp16ScalarProduct/fp16ScalarProduct.cu
//   https://github.com/NVIDIA/cuda-samples/blob/master/cpp/0_Introduction/fp16ScalarProduct/fp16ScalarProduct.cu
//   Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//   SPDX-License-Identifier: BSD-3-Clause
//
// scalarProductKernel_native / reduceInShared_native use half2's native
// operator+ and operator* overloads; scalarProductKernel_intrinsics /
// reduceInShared_intrinsics use the equivalent explicit __hfma2/__hadd2
// intrinsics. Both kernels walk `size` half2 elements per block with an
// identical grid-stride loop and then run the identical 64/32/16/8/4/2/1
// shared-memory tree reduction -- the point of the comparison is which
// PTX instructions the compiler selects for "the same math written two
// ways" (half2's operator+/operator* are themselves thin wrappers
// around __hadd2/__hmul2 in <cuda_fp16.hpp>), not a numeric difference.
//
// Deterministic inputs (replicated by reference.h), replacing the
// original sample's rand()-seeded generateInput(), chosen so every
// intermediate fp16 sum in the per-thread accumulation and the block's
// tree reduction is an exactly-representable integer (fp16 represents
// every integer in [-2048, 2048] exactly) -- making the comparison
// below a genuine exact bit match, tighter than the original sample's
// own 1e-5 float tolerance:
//   a[i].x = a[i].y = (i % 4)
//   b[i].x = b[i].y = (i % 2)
// NUM_OF_BLOCKS = 128, NUM_OF_THREADS = 128 (as in the original
// sample), size = NUM_OF_BLOCKS * NUM_OF_THREADS * 16 = 262144.
//
// Why this stays exact: the grid stride (NUM_OF_BLOCKS * NUM_OF_THREADS
// = 16384) is divisible by both 4 and 2, so a given thread's `i % 4`
// and `i % 2` are constant across all 16 of its grid-stride iterations,
// and a given thread's starting index `i = t + 128*b` has `i % 4 == t %
// 4` and `i % 2 == t % 2` independent of the block index `b`. Summing
// the resulting per-thread partials (0, 16, 0, or 48, one per lane, for
// t % 4 == 0/1/2/3 respectively, 32 threads of each residue per
// 128-thread block) over a block's 128 threads gives exactly 2048 in
// each half2 lane, for every block -- right at, but not over, fp16's
// exact-integer boundary, and requires no assumption that additions
// happen in a particular order (every intermediate partial sum is a
// non-negative integer <= 2048, so every fp16 add along the way is
// exact regardless of order).
//
// Output: argv[1] (default output/cuda_output.txt), 128 lines, one
// per-block float dot-product partial from scalarProductKernel_intrinsics.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
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

int main(int argc, char **argv) {
  const size_t size = (size_t)NUM_OF_BLOCKS * NUM_OF_THREADS * 16;  // 262144
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";

  half2 *hA = (half2 *)malloc(size * sizeof(half2));
  half2 *hB = (half2 *)malloc(size * sizeof(half2));
  for (size_t i = 0; i < size; ++i) {
    float av = (float)gen_a_lane(i);
    float bv = (float)gen_b_lane(i);
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

  float *resultsNative = (float *)malloc(NUM_OF_BLOCKS * sizeof(float));
  float *resultsIntrinsics = (float *)malloc(NUM_OF_BLOCKS * sizeof(float));
  CHECK(cudaMemcpy(resultsNative, dResultsNative, NUM_OF_BLOCKS * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(resultsIntrinsics, dResultsIntrinsics, NUM_OF_BLOCKS * sizeof(float), cudaMemcpyDeviceToHost));

  // CPU reference: per-block partials, same block/thread decomposition
  // and grid-stride order as the GPU kernels (see reference.h).
  double *refBlocks = (double *)malloc(NUM_OF_BLOCKS * sizeof(double));
  reference_block_partials(refBlocks, size, NUM_OF_BLOCKS, NUM_OF_THREADS);

  double max_abs_err_native = 0.0, max_abs_err_intrinsics = 0.0;
  for (int b = 0; b < NUM_OF_BLOCKS; ++b) {
    double d1 = (double)resultsNative[b] - refBlocks[b];
    double d2 = (double)resultsIntrinsics[b] - refBlocks[b];
    if (d1 < 0) d1 = -d1;
    if (d2 < 0) d2 = -d2;
    if (d1 > max_abs_err_native) max_abs_err_native = d1;
    if (d2 > max_abs_err_intrinsics) max_abs_err_intrinsics = d2;
  }

  // Aggregate scalar, summed in the same sequential block order as the
  // original sample's own `result += results[i]` loop, for both kernels
  // and the reference, so the aggregate comparison needs no tolerance
  // beyond what the per-block comparison above already established.
  double sum_native = 0.0, sum_intrinsics = 0.0, sum_ref = 0.0;
  for (int b = 0; b < NUM_OF_BLOCKS; ++b) {
    sum_native += resultsNative[b];
    sum_intrinsics += resultsIntrinsics[b];
    sum_ref += refBlocks[b];
  }
  double agg_err_native = sum_native - sum_ref;
  double agg_err_intrinsics = sum_intrinsics - sum_ref;
  if (agg_err_native < 0) agg_err_native = -agg_err_native;
  if (agg_err_intrinsics < 0) agg_err_intrinsics = -agg_err_intrinsics;

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (int b = 0; b < NUM_OF_BLOCKS; ++b) fprintf(f, "%.9g\n", (double)resultsIntrinsics[b]);
  fclose(f);

  int pass = (max_abs_err_native == 0.0 && max_abs_err_intrinsics == 0.0 &&
              agg_err_native == 0.0 && agg_err_intrinsics == 0.0);

  printf("fp16PackedScalarProduct done: size=%zu, blocks=%d, threads=%d, "
         "max_abs_error(native vs ref)=%.3e, max_abs_error(intrinsics vs ref)=%.3e, "
         "sum_native=%.6f, sum_intrinsics=%.6f, sum_ref=%.6f -> %s\n",
         size, NUM_OF_BLOCKS, NUM_OF_THREADS, max_abs_err_native, max_abs_err_intrinsics,
         sum_native, sum_intrinsics, sum_ref, out);
  printf("%s\n", pass ? "PASS" : "FAIL");

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dResultsNative);
  cudaFree(dResultsIntrinsics);
  free(hA);
  free(hB);
  free(resultsNative);
  free(resultsIntrinsics);
  free(refBlocks);
  return 0;
}
