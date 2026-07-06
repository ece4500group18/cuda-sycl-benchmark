// bitonicVsOddEvenMergeSort: two in-shared-memory sorting *networks* -- a
// bitonic sort (bitonicSortShared) and a Batcher odd-even merge sort
// (oddEvenMergeSortShared) -- run over the same (key, value) array, both
// producing the identical sorted permutation via a different comparator-network
// topology.
//
// The Comparator device function and both __global__ kernels below are
// reproduced, unmodified, from:
//   NVIDIA/cuda-samples, cpp/2_Concepts_and_Techniques/sortingNetworks/
//   Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//   SPDX-License-Identifier: BSD-3-Clause
//   (bitonicSortShared from bitonicSort.cu; oddEvenMergeSortShared from
//   oddEvenMergeSort.cu; Comparator from sortingNetworks_common.cuh)
//
// Deterministic inputs (reproduced by tests/verify.py):
//   key[i] = (i * 40503u) % 65536u   (uint32 arithmetic; 40503 invertible mod
//                                     65536, so all 1024 keys are distinct)
//   val[i] = i
//   arrayLength = 1024 (== SHARED_SIZE_LIMIT: a single block sorts it), dir = 1.
//
// Output: argv[1] (default output/cuda_output.txt), one "key val" pair per line,
// the bitonic-sort kernel's sorted result. Checked by tests/verify.py against a
// CPU sort-by-key (keys distinct -> exact, unambiguous oracle).

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

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

// typedef and constant adapted from sortingNetworks_common.h/.cuh so the
// kernels below compile standalone.
typedef unsigned int uint;
#define SHARED_SIZE_LIMIT 1024U

// --- begin: verbatim from NVIDIA/cuda-samples sortingNetworks_common.cuh ---

__device__ inline void Comparator(uint &keyA, uint &valA, uint &keyB, uint &valB, uint dir)
{
    uint t;

    if ((keyA > keyB) == dir) {
        t    = keyA;
        keyA = keyB;
        keyB = t;
        t    = valA;
        valA = valB;
        valB = t;
    }
}

// --- end: verbatim from sortingNetworks_common.cuh ---

// --- begin: verbatim from NVIDIA/cuda-samples bitonicSort.cu ---

__global__ void
bitonicSortShared(uint *d_DstKey, uint *d_DstVal, uint *d_SrcKey, uint *d_SrcVal, uint arrayLength, uint dir)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    // Shared memory storage for one or more short vectors
    __shared__ uint s_key[SHARED_SIZE_LIMIT];
    __shared__ uint s_val[SHARED_SIZE_LIMIT];

    // Offset to the beginning of subbatch and load data
    d_SrcKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_SrcVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    s_key[threadIdx.x + 0]                       = d_SrcKey[0];
    s_val[threadIdx.x + 0]                       = d_SrcVal[0];
    s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcKey[(SHARED_SIZE_LIMIT / 2)];
    s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcVal[(SHARED_SIZE_LIMIT / 2)];

    for (uint size = 2; size < arrayLength; size <<= 1) {
        // Bitonic merge
        uint ddd = dir ^ ((threadIdx.x & (size / 2)) != 0);

        for (uint stride = size / 2; stride > 0; stride >>= 1) {
            cg::sync(cta);
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            Comparator(s_key[pos + 0], s_val[pos + 0], s_key[pos + stride], s_val[pos + stride], ddd);
        }
    }

    // ddd == dir for the last bitonic merge step
    {
        for (uint stride = arrayLength / 2; stride > 0; stride >>= 1) {
            cg::sync(cta);
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            Comparator(s_key[pos + 0], s_val[pos + 0], s_key[pos + stride], s_val[pos + stride], dir);
        }
    }

    cg::sync(cta);
    d_DstKey[0]                       = s_key[threadIdx.x + 0];
    d_DstVal[0]                       = s_val[threadIdx.x + 0];
    d_DstKey[(SHARED_SIZE_LIMIT / 2)] = s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
    d_DstVal[(SHARED_SIZE_LIMIT / 2)] = s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}

// --- end: verbatim from bitonicSort.cu ---

// --- begin: verbatim from NVIDIA/cuda-samples oddEvenMergeSort.cu ---

__global__ void
oddEvenMergeSortShared(uint *d_DstKey, uint *d_DstVal, uint *d_SrcKey, uint *d_SrcVal, uint arrayLength, uint dir)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    // Shared memory storage for one or more small vectors
    __shared__ uint s_key[SHARED_SIZE_LIMIT];
    __shared__ uint s_val[SHARED_SIZE_LIMIT];

    // Offset to the beginning of subbatch and load data
    d_SrcKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_SrcVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    s_key[threadIdx.x + 0]                       = d_SrcKey[0];
    s_val[threadIdx.x + 0]                       = d_SrcVal[0];
    s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcKey[(SHARED_SIZE_LIMIT / 2)];
    s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcVal[(SHARED_SIZE_LIMIT / 2)];

    for (uint size = 2; size <= arrayLength; size <<= 1) {
        uint stride = size / 2;
        uint offset = threadIdx.x & (stride - 1);

        {
            cg::sync(cta);
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            Comparator(s_key[pos + 0], s_val[pos + 0], s_key[pos + stride], s_val[pos + stride], dir);
            stride >>= 1;
        }

        for (; stride > 0; stride >>= 1) {
            cg::sync(cta);
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));

            if (offset >= stride)
                Comparator(s_key[pos - stride], s_val[pos - stride], s_key[pos + 0], s_val[pos + 0], dir);
        }
    }

    cg::sync(cta);
    d_DstKey[0]                       = s_key[threadIdx.x + 0];
    d_DstVal[0]                       = s_val[threadIdx.x + 0];
    d_DstKey[(SHARED_SIZE_LIMIT / 2)] = s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
    d_DstVal[(SHARED_SIZE_LIMIT / 2)] = s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}

// --- end: verbatim from oddEvenMergeSort.cu ---

// Deterministic key/value generators (new host code; mirrored in tests/verify.py).
static inline uint gen_key(int i) { return ((uint)i * 40503u) % 65536u; }
static inline uint gen_val(int i) { return (uint)i; }

int main(int argc, char **argv) {
  const uint n = SHARED_SIZE_LIMIT;  // arrayLength, one shared-memory block's worth
  const uint dir = 1;                // ascending
  const char *out = (argc > 1) ? argv[1] : "output/cuda_output.txt";
  const size_t bytes = (size_t)n * sizeof(uint);

  uint *h_key = (uint *)malloc(bytes);
  uint *h_val = (uint *)malloc(bytes);
  for (uint i = 0; i < n; ++i) {
    h_key[i] = gen_key((int)i);
    h_val[i] = gen_val((int)i);
  }

  uint *d_srcKey, *d_srcVal;
  CHECK(cudaMalloc(&d_srcKey, bytes));
  CHECK(cudaMalloc(&d_srcVal, bytes));
  CHECK(cudaMemcpy(d_srcKey, h_key, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_srcVal, h_val, bytes, cudaMemcpyHostToDevice));

  const uint blockCount = n / SHARED_SIZE_LIMIT;   // 1
  const uint threadCount = SHARED_SIZE_LIMIT / 2;  // 512

  // bitonic sort path
  uint *d_bitonicKey, *d_bitonicVal;
  CHECK(cudaMalloc(&d_bitonicKey, bytes));
  CHECK(cudaMalloc(&d_bitonicVal, bytes));
  bitonicSortShared<<<blockCount, threadCount>>>(d_bitonicKey, d_bitonicVal, d_srcKey, d_srcVal, n, dir);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  // odd-even merge sort path (run for the same-network comparison; both must
  // produce the identical permutation)
  uint *d_oemsKey, *d_oemsVal;
  CHECK(cudaMalloc(&d_oemsKey, bytes));
  CHECK(cudaMalloc(&d_oemsVal, bytes));
  oddEvenMergeSortShared<<<blockCount, threadCount>>>(d_oemsKey, d_oemsVal, d_srcKey, d_srcVal, n, dir);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  uint *h_bitonic_key = (uint *)malloc(bytes);
  uint *h_bitonic_val = (uint *)malloc(bytes);
  CHECK(cudaMemcpy(h_bitonic_key, d_bitonicKey, bytes, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(h_bitonic_val, d_bitonicVal, bytes, cudaMemcpyDeviceToHost));

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (uint i = 0; i < n; ++i) fprintf(f, "%u %u\n", h_bitonic_key[i], h_bitonic_val[i]);
  fclose(f);

  cudaFree(d_srcKey);
  cudaFree(d_srcVal);
  cudaFree(d_bitonicKey);
  cudaFree(d_bitonicVal);
  cudaFree(d_oemsKey);
  cudaFree(d_oemsVal);
  free(h_key);
  free(h_val);
  free(h_bitonic_key);
  free(h_bitonic_val);
  return 0;
}
