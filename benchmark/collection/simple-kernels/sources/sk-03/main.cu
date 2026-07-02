// binaryPartitionOddEvenReduce: cooperative_groups::binary_partition() to
// split a warp into odd/even sub-groups + cg::reduce() per sub-group before
// one atomicAdd per sub-group, vs. a naive per-thread atomicAdd of odd/even
// counts and sums.
//
// The __global__ kernel below (oddEvenCountAndSumCG) is reproduced,
// unmodified, from:
//   NVIDIA/cuda-samples, cpp/3_CUDA_Features/binaryPartitionCG/binaryPartitionCG.cu
//   Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//   SPDX-License-Identifier: BSD-3-Clause
//
// For every element it is responsible for, a thread checks whether its
// value is odd or even, then calls `cg::binary_partition(tile32, elem & 1)`
// to split its 32-thread warp tile into two divergent sub-groups (the
// "odd" threads and the "even" threads of that warp, whatever their sizes
// happen to be). Each sub-group then does a single `cg::reduce(subTile,
// elem, cg::plus<int>())` -- an intra-group tree/shuffle reduction -- and
// only the sub-group's rank-0 thread issues two atomics: one to bump the
// shared odd counter (`numOfOdds`) by the sub-group's size, and one to add
// the sub-group's already-reduced partial sum into the appropriate global
// slot. So a 32-thread warp that diverges on odd/even performs at most two
// atomicAdd calls total (one per sub-group), no matter how the odd/even
// split falls, instead of one atomicAdd per thread per accumulator.
//
// "oddEvenCountAndSumNaive" (new code written for this comparison, not
// present upstream) computes the exact same three quantities -- the total
// count of odd elements, the sum of odd elements, and the sum of even
// elements -- but the straightforward way: every thread does its own
// `atomicAdd` of 1 (if odd) and its own `atomicAdd` of `elem` into the
// appropriate slot, with no partitioning or local reduction at all. Every
// thread that touches an odd/even element issues its own atomics on the
// same handful of global addresses, so under warp divergence this is
// exactly the pattern `binary_partition`+`reduce` is designed to avoid.
//
// Both kernels visit every index in [0, size) exactly once (grid-stride
// loop) and combine per-element odd/even counts and sums with plain integer
// addition, which is commutative and associative regardless of grouping or
// order -- so both must produce byte-identical final totals to each other
// and to a straightforward CPU odd/even count-and-sum loop; only *how many
// atomic RMW operations reach global memory, and in what groupings*
// differs. This isolates divergent-warp-partitioning-then-intra-group-
// reduce as the technique for cutting atomic traffic under branch
// divergence.
//
// Deterministic input (replicated by reference.h, replacing the upstream
// sample's own `rand() % 50`):
//   gen(i) = (i * 7 + 3) % 50
// size = 1 << 16 = 65536.
//
// Output: argv[1] (default output/cuda_output.txt), 3 lines: number of odd
// elements, sum of odd elements, sum of even elements (as computed by
// oddEvenCountAndSumCG).

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
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

// --- begin: verbatim from NVIDIA/cuda-samples binaryPartitionCG.cu ---

/**
 * CUDA kernel device code
 *
 * Creates cooperative groups and performs odd/even counting & summation.
 */
__global__ void oddEvenCountAndSumCG(int *inputArr, int *numOfOdds, int *sumOfOddAndEvens, unsigned int size)
{
    cg::thread_block          cta    = cg::this_thread_block();
    cg::grid_group            grid   = cg::this_grid();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    for (int i = grid.thread_rank(); i < size; i += grid.size()) {
        int  elem    = inputArr[i];
        auto subTile = cg::binary_partition(tile32, elem & 1);
        if (elem & 1) // Odd numbers group
        {
            int oddGroupSum = cg::reduce(subTile, elem, cg::plus<int>());

            if (subTile.thread_rank() == 0) {
                // Add number of odds present in this group of Odds.
                atomicAdd(numOfOdds, subTile.size());

                // Add local reduction of odds present in this group of Odds.
                atomicAdd(&sumOfOddAndEvens[0], oddGroupSum);
            }
        }
        else // Even numbers group
        {
            int evenGroupSum = cg::reduce(subTile, elem, cg::plus<int>());

            if (subTile.thread_rank() == 0) {
                // Add local reduction of even present in this group of evens.
                atomicAdd(&sumOfOddAndEvens[1], evenGroupSum);
            }
        }
        // reconverge warp so for next loop iteration we ensure convergence of
        // above diverged threads to perform coalesced loads of inputArr.
        cg::sync(tile32);
    }
}

// --- end: verbatim from NVIDIA/cuda-samples ---

// --- begin: new code for this repository -- naive per-thread atomic
// counterpart to oddEvenCountAndSumCG above, computing the exact same
// three quantities with one atomicAdd per thread per touched accumulator
// (no binary_partition, no intra-group reduce). ---

__global__ void oddEvenCountAndSumNaive(int *inputArr, int *numOfOdds, int *sumOfOddAndEvens, unsigned int size)
{
    unsigned int tid          = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int total_threads = gridDim.x * blockDim.x;

    for (unsigned int i = tid; i < size; i += total_threads) {
        int elem = inputArr[i];
        if (elem & 1) { // Odd
            atomicAdd(numOfOdds, 1);
            atomicAdd(&sumOfOddAndEvens[0], elem);
        } else { // Even
            atomicAdd(&sumOfOddAndEvens[1], elem);
        }
    }
}

// --- end: new code for this repository ---

int main(int argc, char **argv) {
  const unsigned int arrSize = 1u << 16;  // 65536
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";

  int *hInputArr = (int *)malloc(sizeof(int) * arrSize);
  for (unsigned int i = 0; i < arrSize; ++i) hInputArr[i] = gen(i);

  int *dInputArr;
  int *dNumOfOddsCG, *dSumOfOddEvenCG;
  int *dNumOfOddsNaive, *dSumOfOddEvenNaive;

  CHECK(cudaMalloc(&dInputArr, sizeof(int) * arrSize));
  CHECK(cudaMalloc(&dNumOfOddsCG, sizeof(int)));
  CHECK(cudaMalloc(&dSumOfOddEvenCG, sizeof(int) * 2));
  CHECK(cudaMalloc(&dNumOfOddsNaive, sizeof(int)));
  CHECK(cudaMalloc(&dSumOfOddEvenNaive, sizeof(int) * 2));

  CHECK(cudaMemcpy(dInputArr, hInputArr, sizeof(int) * arrSize, cudaMemcpyHostToDevice));
  CHECK(cudaMemset(dNumOfOddsCG, 0, sizeof(int)));
  CHECK(cudaMemset(dSumOfOddEvenCG, 0, sizeof(int) * 2));
  CHECK(cudaMemset(dNumOfOddsNaive, 0, sizeof(int)));
  CHECK(cudaMemset(dSumOfOddEvenNaive, 0, sizeof(int) * 2));

  const int threadsPerBlock = 256;  // multiple of 32: tile32 partitions evenly
  const int blocksPerGrid   = 128;  // 32768 threads, grid-stride covers arrSize

  oddEvenCountAndSumCG<<<blocksPerGrid, threadsPerBlock>>>(
      dInputArr, dNumOfOddsCG, dSumOfOddEvenCG, arrSize);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  oddEvenCountAndSumNaive<<<blocksPerGrid, threadsPerBlock>>>(
      dInputArr, dNumOfOddsNaive, dSumOfOddEvenNaive, arrSize);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  int hNumOfOddsCG = 0, hSumOfOddEvenCG[2] = {0, 0};
  int hNumOfOddsNaive = 0, hSumOfOddEvenNaive[2] = {0, 0};

  CHECK(cudaMemcpy(&hNumOfOddsCG, dNumOfOddsCG, sizeof(int), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(hSumOfOddEvenCG, dSumOfOddEvenCG, sizeof(int) * 2, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(&hNumOfOddsNaive, dNumOfOddsNaive, sizeof(int), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(hSumOfOddEvenNaive, dSumOfOddEvenNaive, sizeof(int) * 2, cudaMemcpyDeviceToHost));

  // CPU reference check.
  long long refNumOfOdds = 0, refSumOfOdds = 0, refSumOfEvens = 0;
  reference_odd_even_count_and_sum(arrSize, &refNumOfOdds, &refSumOfOdds, &refSumOfEvens);

  int exact_ok_cg =
      ((long long)hNumOfOddsCG == refNumOfOdds) &&
      ((long long)hSumOfOddEvenCG[0] == refSumOfOdds) &&
      ((long long)hSumOfOddEvenCG[1] == refSumOfEvens);
  int exact_ok_naive =
      ((long long)hNumOfOddsNaive == refNumOfOdds) &&
      ((long long)hSumOfOddEvenNaive[0] == refSumOfOdds) &&
      ((long long)hSumOfOddEvenNaive[1] == refSumOfEvens);

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  fprintf(f, "%d\n", hNumOfOddsCG);
  fprintf(f, "%d\n", hSumOfOddEvenCG[0]);
  fprintf(f, "%d\n", hSumOfOddEvenCG[1]);
  fclose(f);

  printf("binaryPartitionOddEvenReduce done: size=%u, "
         "numOfOdds(cg)=%d numOfOdds(naive)=%d, "
         "sumOfOdds(cg)=%d sumOfOdds(naive)=%d, "
         "sumOfEvens(cg)=%d sumOfEvens(naive)=%d -> %s\n",
         arrSize, hNumOfOddsCG, hNumOfOddsNaive,
         hSumOfOddEvenCG[0], hSumOfOddEvenNaive[0],
         hSumOfOddEvenCG[1], hSumOfOddEvenNaive[1], out);
  printf("%s\n", (exact_ok_cg && exact_ok_naive) ? "PASS" : "FAIL");

  cudaFree(dInputArr);
  cudaFree(dNumOfOddsCG);
  cudaFree(dSumOfOddEvenCG);
  cudaFree(dNumOfOddsNaive);
  cudaFree(dSumOfOddEvenNaive);
  free(hInputArr);
  return 0;
}
