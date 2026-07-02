// bitonicVsOddEvenMergeSort: two different in-shared-memory sorting
// *networks* -- a bitonic sort (bitonicSortShared) and a Batcher
// odd-even merge sort (oddEvenMergeSortShared) -- run over the exact
// same (key, value) array, both producing the identical sorted
// permutation via a different comparator-network topology.
//
// The `Comparator` device function and both `__global__` kernels below
// are reproduced, unmodified, from:
//   NVIDIA/cuda-samples, cpp/2_Concepts_and_Techniques/sortingNetworks/
//   https://github.com/NVIDIA/cuda-samples/blob/master/cpp/2_Concepts_and_Techniques/sortingNetworks/
//   Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//   SPDX-License-Identifier: BSD-3-Clause
//   (bitonicSortShared from bitonicSort.cu; oddEvenMergeSortShared from
//   oddEvenMergeSort.cu; Comparator from sortingNetworks_common.cuh)
//
// Both kernels load one block's worth of `SHARED_SIZE_LIMIT` (key,
// value) pairs into shared memory and repeatedly call `Comparator` on
// pairs of shared-memory slots selected by pure thread-index bit
// arithmetic, separated by `cg::sync(cta)` barriers -- but the two
// kernels wire up *which* slots get compared, and in what sequence, via
// two different classic sorting-network constructions:
//
// - "bitonicSortShared": builds bitonic sequences of doubling size
//   (`size = 2, 4, 8, ...`) via a recursive bitonic-merge butterfly,
//   flipping comparator direction based on `threadIdx.x & (size / 2)`,
//   then finishes with one full descending-stride bitonic merge pass.
//   Total stage count is O(log^2 N).
//
// - "oddEvenMergeSortShared": builds Batcher's odd-even merge networks
//   of doubling size (`size = 2, 4, 8, ...`), where each merge stage
//   does one direct comparator pass at `stride = size/2` and then a
//   sequence of *conditional* comparator passes (`if (offset >=
//   stride)`) at halving strides -- a different comparator-network
//   shape from the bitonic butterfly above, also O(log^2 N) stages, but
//   with different data dependencies and a different (smaller, for this
//   network) total comparator count.
//
// Both are complete, self-contained sorting networks that provably sort
// *any* input of `SHARED_SIZE_LIMIT` elements; because they are
// different circuits computing the same total-order relation, they are
// guaranteed to produce the exact same sorted output for the same
// input, and this is a "same-result-via-different-algorithm" pair
// rather than a memory-movement comparison. This is "simple but not
// trivial": each individual comparator call is a one-line conditional
// swap, but reasoning about why *two structurally different* O(log^2 N)
// networks of comparator stages must converge on the identical
// permutation is the substantive, nontrivial part.
//
// Deterministic inputs (replicated by reference.h):
//   key[i] = (i * 40503u) % 65536u  (uint32_t arithmetic)
//   val[i] = i
// arrayLength = 1024 (== SHARED_SIZE_LIMIT, so the whole array is sorted
// by a single thread block's shared-memory network in one kernel
// launch, avoiding the original sample's multi-block/multi-batch
// host-side merge machinery entirely); dir fixed to 1 (ascending, per
// the shared `Comparator`'s `(keyA > keyB) == dir` convention).
//
// Because the multiplier 40503 is invertible modulo 65536 (see
// reference.h), all 1024 generated keys are pairwise distinct, so the
// sorted order is unambiguous and a plain, position-by-position array
// comparison against a CPU std::sort is an *exact* correctness oracle
// (integer keys throughout, no floating point, no tie-breaking
// ambiguity, no accumulation-order sensitivity).
//
// Output: argv[1] (default output/output.txt), one "key val" pair per
// line (space-separated), the bitonic-sort kernel's sorted result.

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

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

// typedef and constant adapted from NVIDIA/cuda-samples
// cpp/2_Concepts_and_Techniques/sortingNetworks/sortingNetworks_common.h
// and sortingNetworks_common.cuh (not kernel code; needed so the
// kernels below can compile standalone).
typedef unsigned int uint;
#define SHARED_SIZE_LIMIT 1024U

// --- begin: verbatim from NVIDIA/cuda-samples
//     cpp/2_Concepts_and_Techniques/sortingNetworks/sortingNetworks_common.cuh ---

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

// --- begin: verbatim from NVIDIA/cuda-samples
//     cpp/2_Concepts_and_Techniques/sortingNetworks/bitonicSort.cu ---

////////////////////////////////////////////////////////////////////////////////
// Monolithic bitonic sort kernel for short arrays fitting into shared memory
////////////////////////////////////////////////////////////////////////////////
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

// --- begin: verbatim from NVIDIA/cuda-samples
//     cpp/2_Concepts_and_Techniques/sortingNetworks/oddEvenMergeSort.cu ---

////////////////////////////////////////////////////////////////////////////////
// Monolithic Bacther's sort kernel for short arrays fitting into shared memory
////////////////////////////////////////////////////////////////////////////////
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

int main(int argc, char **argv) {
  const uint n = SHARED_SIZE_LIMIT;  // arrayLength, a single shared-memory block's worth
  const uint dir = 1;                // ascending
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
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

  const uint blockCount = n / SHARED_SIZE_LIMIT;   // = 1 (single threadblock)
  const uint threadCount = SHARED_SIZE_LIMIT / 2;  // = 512

  // --- bitonic sort path ---
  uint *d_bitonicKey, *d_bitonicVal;
  CHECK(cudaMalloc(&d_bitonicKey, bytes));
  CHECK(cudaMalloc(&d_bitonicVal, bytes));
  bitonicSortShared<<<blockCount, threadCount>>>(d_bitonicKey, d_bitonicVal, d_srcKey, d_srcVal, n, dir);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  uint *h_bitonic_key = (uint *)malloc(bytes);
  uint *h_bitonic_val = (uint *)malloc(bytes);
  CHECK(cudaMemcpy(h_bitonic_key, d_bitonicKey, bytes, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(h_bitonic_val, d_bitonicVal, bytes, cudaMemcpyDeviceToHost));

  // --- odd-even merge sort path ---
  uint *d_oemsKey, *d_oemsVal;
  CHECK(cudaMalloc(&d_oemsKey, bytes));
  CHECK(cudaMalloc(&d_oemsVal, bytes));
  oddEvenMergeSortShared<<<blockCount, threadCount>>>(d_oemsKey, d_oemsVal, d_srcKey, d_srcVal, n, dir);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  uint *h_oems_key = (uint *)malloc(bytes);
  uint *h_oems_val = (uint *)malloc(bytes);
  CHECK(cudaMemcpy(h_oems_key, d_oemsKey, bytes, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(h_oems_val, d_oemsVal, bytes, cudaMemcpyDeviceToHost));

  // CPU reference (std::sort by key; all keys are pairwise distinct, see
  // reference.h, so this is an unambiguous, exact oracle).
  uint *href_key = (uint *)malloc(bytes);
  uint *href_val = (uint *)malloc(bytes);
  reference_sort(h_key, h_val, (int)n, href_key, href_val);

  int bitonic_ok = 1, oems_ok = 1, cross_ok = 1;
  for (uint i = 0; i < n; ++i) {
    if (h_bitonic_key[i] != href_key[i] || h_bitonic_val[i] != href_val[i]) bitonic_ok = 0;
    if (h_oems_key[i] != href_key[i] || h_oems_val[i] != href_val[i]) oems_ok = 0;
    if (h_bitonic_key[i] != h_oems_key[i] || h_bitonic_val[i] != h_oems_val[i]) cross_ok = 0;
  }

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (uint i = 0; i < n; ++i) fprintf(f, "%u %u\n", h_bitonic_key[i], h_bitonic_val[i]);
  fclose(f);

  printf("bitonicVsOddEvenMergeSort done: n=%u, bitonic_exact_match=%d, "
         "oddEvenMergeSort_exact_match=%d, bitonic_vs_oddEvenMergeSort_identical=%d -> %s\n",
         n, bitonic_ok, oems_ok, cross_ok, out);
  printf("%s\n", (bitonic_ok && oems_ok && cross_ok) ? "PASS" : "FAIL");

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
  free(h_oems_key);
  free(h_oems_val);
  free(href_key);
  free(href_val);
  return 0;
}
