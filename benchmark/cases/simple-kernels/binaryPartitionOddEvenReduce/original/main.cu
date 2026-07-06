// binaryPartitionOddEvenReduce: cooperative_groups::binary_partition() splits a
// warp into odd/even sub-groups + cg::reduce() per sub-group before one
// atomicAdd per sub-group, vs. a naive per-thread atomicAdd counterpart.
//
// The __global__ kernel oddEvenCountAndSumCG below is reproduced, unmodified,
// from:
//   NVIDIA/cuda-samples, cpp/3_CUDA_Features/binaryPartitionCG/binaryPartitionCG.cu
//   Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//   SPDX-License-Identifier: BSD-3-Clause
//
// oddEvenCountAndSumNaive is new code written for this repository: it computes
// the same three quantities with one atomicAdd per thread per accumulator.
//
// Deterministic input (reproduced by tests/verify.py), replacing the upstream
// sample's rand() % 50:
//   gen(i) = (i * 7 + 3) % 50,  size = 1 << 16 = 65536.
//
// Output: argv[1] (default output/cuda_output.txt), 3 lines: number of odd
// elements, sum of odd elements, sum of even elements (from the CG kernel).

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

#define CHECK(call)                                                          \
  do {                                                                       \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), \
              __FILE__, __LINE__);                                           \
      return 2;                                                              \
    }                                                                        \
  } while (0)

// --- begin: verbatim from NVIDIA/cuda-samples binaryPartitionCG.cu ---

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
// counterpart to oddEvenCountAndSumCG. ---

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

// Deterministic input generator (new host code; mirrored in tests/verify.py).
static inline int gen(unsigned int i) { return (int)((i * 7u + 3u) % 50u); }

int main(int argc, char **argv) {
  const unsigned int arrSize = 1u << 16;  // 65536
  const char *out = (argc > 1) ? argv[1] : "output/cuda_output.txt";

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
  const int blocksPerGrid   = 128;  // grid-stride covers arrSize

  oddEvenCountAndSumCG<<<blocksPerGrid, threadsPerBlock>>>(
      dInputArr, dNumOfOddsCG, dSumOfOddEvenCG, arrSize);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  oddEvenCountAndSumNaive<<<blocksPerGrid, threadsPerBlock>>>(
      dInputArr, dNumOfOddsNaive, dSumOfOddEvenNaive, arrSize);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  int hNumOfOddsCG = 0, hSumOfOddEvenCG[2] = {0, 0};

  CHECK(cudaMemcpy(&hNumOfOddsCG, dNumOfOddsCG, sizeof(int), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(hSumOfOddEvenCG, dSumOfOddEvenCG, sizeof(int) * 2, cudaMemcpyDeviceToHost));

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  fprintf(f, "%d\n", hNumOfOddsCG);
  fprintf(f, "%d\n", hSumOfOddEvenCG[0]);
  fprintf(f, "%d\n", hSumOfOddEvenCG[1]);
  fclose(f);

  cudaFree(dInputArr);
  cudaFree(dNumOfOddsCG);
  cudaFree(dSumOfOddEvenCG);
  cudaFree(dNumOfOddsNaive);
  cudaFree(dSumOfOddEvenNaive);
  free(hInputArr);
  return 0;
}
