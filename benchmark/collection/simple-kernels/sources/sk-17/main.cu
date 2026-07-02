// warpAggregatedAtomicCompaction: warp-aggregated atomic increment
// (cg::coalesced_threads() + shfl-broadcast, issuing one atomicAdd per
// *active warp*) vs. a naive one-atomicAdd-per-thread baseline, both used
// to perform stream compaction (copy every positive element of src[] into
// a contiguous dst[] array, tracking the compacted count in *nres).
//
// atomicAggInc (__device__ helper) and filter_arr (__global__ kernel)
// below are reproduced, unmodified, from:
//   NVIDIA/cuda-samples,
//   cpp/3_CUDA_Features/warpAggregatedAtomicsCG/warpAggregatedAtomicsCG.cu
//   https://github.com/NVIDIA/cuda-samples/blob/master/cpp/3_CUDA_Features/warpAggregatedAtomicsCG/warpAggregatedAtomicsCG.cu
//   Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//   SPDX-License-Identifier: BSD-3-Clause
//
// filter_arr_naive is new code written for this repository: the identical
// stream-compaction loop, but every thread that keeps an element calls
// atomicAdd(nres, 1) directly, with zero coordination between the threads
// of a warp. This is "simple but not trivial": the compaction predicate
// itself (`src[i] > 0`) is trivial, but reasoning about how many atomic
// read-modify-write instructions actually retire on the shared counter --
// and recognizing that a warp-level leader-election-plus-broadcast can
// replace up to 32 per-thread atomics with a single one, without changing
// *what* gets computed -- is the substantive part:
//   - filter_arr: cg::coalesced_threads() finds the subset of the warp's
//     32 lanes that are still active at this program point; lane 0 of
//     that group ("the leader") performs a single atomicAdd(counter,
//     active.size()) for the whole group, and active.shfl(res, 0)
//     broadcasts the resulting base offset back to every other active
//     lane, which then adds its own active.thread_rank() to land on a
//     unique slot -- at most one atomicAdd per warp per loop iteration,
//     regardless of how many of the warp's lanes kept an element.
//   - filter_arr_naive: every lane that keeps an element calls
//     atomicAdd(nres, 1) on its own -- up to 32x more atomic traffic on
//     the same counter when a whole warp's worth of lanes are active.
//
// Both kernels visit every index in [0, n) exactly once and, for every i
// with src[i] > 0, write src[i] to some slot of dst[]; the only
// difference is the mechanism used to obtain a unique slot index. Because
// block/warp scheduling order on the GPU is unspecified, the *position*
// a given kept value ends up at in dst[] is not reproducible run-to-run
// or between the two kernels -- only the resulting *set* (multiset) of
// compacted values, and its size, is guaranteed. The correctness oracle
// therefore sorts both GPU outputs (and a CPU-filtered-and-sorted
// reference) and compares them as multisets -- i.e. exact integer
// equality after sorting -- rather than assuming a positional/index-exact
// match; this is a documented set-equality oracle, not an incidental
// exact match.
//
// Deterministic input (replicated by reference.h):
//   src[i] = (i % 13) - 6, for i in [0, n), n = 1 << 20 = 1048576
// (13-periodic, values in [-6, 6]; 6 of every 13 values are > 0, so the
// expected compacted count, 6 * (n / 13) [+ the partial final period], is
// known ahead of time from reference.h alone.)
//
// Output: argv[1] (default output/cuda_output.txt): first line = the
// compacted count produced by filter_arr (*nres), followed by that many
// lines, one int per line, the compacted dst[] array from filter_arr,
// sorted ascending.

#include <algorithm>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "reference.h"

namespace cg = cooperative_groups;

#define CHECK(call)                                                          \
  do {                                                                       \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), \
              __FILE__, __LINE__);                                          \
      return 2;                                                             \
    }                                                                        \
  } while (0)

// --- begin: verbatim from NVIDIA/cuda-samples warpAggregatedAtomicsCG.cu ---

// warp-aggregated atomic increment
__device__ int atomicAggInc(int *counter)
{
    cg::coalesced_group active = cg::coalesced_threads();

    // leader does the update
    int res = 0;
    if (active.thread_rank() == 0) {
        res = atomicAdd(counter, active.size());
    }

    // broadcast result
    res = active.shfl(res, 0);

    // each thread computes its own value
    return res + active.thread_rank();
}

__global__ void filter_arr(int *dst, int *nres, const int *src, int n)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = id; i < n; i += gridDim.x * blockDim.x) {
        if (src[i] > 0)
            dst[atomicAggInc(nres)] = src[i];
    }
}

// --- end: verbatim from NVIDIA/cuda-samples ---

// New code for this repository: a naive baseline for the same stream
// compaction, with no warp-level coordination -- every thread that keeps
// an element issues its own atomicAdd on the shared counter.
__global__ void filter_arr_naive(int *dst, int *nres, const int *src, int n)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = id; i < n; i += gridDim.x * blockDim.x) {
        if (src[i] > 0)
            dst[atomicAdd(nres, 1)] = src[i];
    }
}

int main(int argc, char **argv) {
  const int n = 1 << 20;  // 1048576, a multiple of the 512-thread block size
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  const size_t bytes = (size_t)n * sizeof(int);

  int *h_src = (int *)malloc(bytes);
  for (int i = 0; i < n; ++i) h_src[i] = gen_src(i);

  int *d_src, *d_dst_agg, *d_dst_naive, *d_nres_agg, *d_nres_naive;
  CHECK(cudaMalloc(&d_src, bytes));
  CHECK(cudaMalloc(&d_dst_agg, bytes));
  CHECK(cudaMalloc(&d_dst_naive, bytes));
  CHECK(cudaMalloc(&d_nres_agg, sizeof(int)));
  CHECK(cudaMalloc(&d_nres_naive, sizeof(int)));

  CHECK(cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemset(d_nres_agg, 0, sizeof(int)));
  CHECK(cudaMemset(d_nres_naive, 0, sizeof(int)));

  const int threadsPerBlock = 512;
  const int blocks = n / threadsPerBlock;  // 2048, exact (n is a multiple)

  filter_arr<<<blocks, threadsPerBlock>>>(d_dst_agg, d_nres_agg, d_src, n);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  filter_arr_naive<<<blocks, threadsPerBlock>>>(d_dst_naive, d_nres_naive, d_src, n);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  int nres_agg = 0, nres_naive = 0;
  CHECK(cudaMemcpy(&nres_agg, d_nres_agg, sizeof(int), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(&nres_naive, d_nres_naive, sizeof(int), cudaMemcpyDeviceToHost));

  int *h_dst_agg = (int *)malloc(bytes);
  int *h_dst_naive = (int *)malloc(bytes);
  CHECK(cudaMemcpy(h_dst_agg, d_dst_agg, (size_t)nres_agg * sizeof(int), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(h_dst_naive, d_dst_naive, (size_t)nres_naive * sizeof(int), cudaMemcpyDeviceToHost));

  std::sort(h_dst_agg, h_dst_agg + nres_agg);
  std::sort(h_dst_naive, h_dst_naive + nres_naive);

  // CPU reference check: filter + sort (see reference.h for why sorting
  // is the correct oracle here, rather than a positional comparison).
  int *href_sorted = (int *)malloc(bytes);
  int ref_count = reference_filter_sorted(h_src, n, href_sorted);

  int agg_ok = (nres_agg == ref_count);
  for (int i = 0; agg_ok && i < ref_count; ++i)
    if (h_dst_agg[i] != href_sorted[i]) agg_ok = 0;

  int naive_ok = (nres_naive == ref_count);
  for (int i = 0; naive_ok && i < ref_count; ++i)
    if (h_dst_naive[i] != href_sorted[i]) naive_ok = 0;

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  fprintf(f, "%d\n", nres_agg);
  for (int i = 0; i < nres_agg; ++i) fprintf(f, "%d\n", h_dst_agg[i]);
  fclose(f);

  printf("warpAggregatedAtomicCompaction done: n=%d, ref_count=%d, "
         "nres_agg=%d (set_match=%d), nres_naive=%d (set_match=%d) -> %s\n",
         n, ref_count, nres_agg, agg_ok, nres_naive, naive_ok, out);
  printf("%s\n", (agg_ok && naive_ok) ? "PASS" : "FAIL");

  cudaFree(d_src);
  cudaFree(d_dst_agg);
  cudaFree(d_dst_naive);
  cudaFree(d_nres_agg);
  cudaFree(d_nres_naive);
  free(h_src);
  free(h_dst_agg);
  free(h_dst_naive);
  free(href_sorted);
  return 0;
}
