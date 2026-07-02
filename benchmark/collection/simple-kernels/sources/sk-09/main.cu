// gridSyncCGReduction: single-kernel cooperative-groups grid-wide sum
// reduction (cg::this_grid() + grid.sync(), launched via
// cudaLaunchCooperativeKernel) vs. the classic two-kernel-launch tree
// reduction that combines the exact same per-block partial sums without
// any cooperative launch or grid-wide barrier.
//
// The device function `reduceBlock` and the __global__ kernel
// `reduceSinglePassMultiBlockCG` below are reproduced from:
//   NVIDIA/cuda-samples,
//   cpp/2_Concepts_and_Techniques/reductionMultiBlockCG/reductionMultiBlockCG.cu
//   Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//   SPDX-License-Identifier: BSD-3-Clause
//
// `reduceBlock` is reproduced completely unmodified. `reduceSinglePassMultiBlockCG`
// is reproduced with a single, mechanical type change: the upstream kernel
// operates on `float* g_idata`/`float* g_odata` (with an internal `double`
// shared-memory accumulator); here both the input and output pointers are
// `double*` instead, so the per-block partial sum written to global memory
// no longer gets truncated back down to float precision. This is required
// to make the "exact match" correctness oracle below hold (see
// reference.h) -- no other line of the algorithm is touched.
//
// reduceSinglePassMultiBlockCG works in two phases inside ONE kernel
// launch, separated by a grid-wide barrier:
//   Phase 1 (per block): each thread sums its grid-stride share of the
//     input into shared memory, `reduceBlock` combines the block's threads
//     (warp-shuffle tree, then a sequential leader loop) into one partial
//     sum per block, written to g_odata[blockIdx.x].
//   -- cg::sync(grid): every block in the grid must reach this point before
//      any block proceeds; only possible because the whole grid was
//      launched as one cooperative kernel (cudaLaunchCooperativeKernel)
//      with all blocks guaranteed co-resident. --
//   Phase 2 (grid thread 0 only): sequentially adds g_odata[1..gridDim.x-1]
//     onto g_odata[0].
//
// `reduceBlockPartial` and `reduceCombinePartials` below are NEW kernels
// written for this repository, not present upstream. They split exactly
// the same two phases across two ordinary (non-cooperative) kernel
// launches instead of one cooperative kernel with an internal grid.sync():
// `reduceBlockPartial` is phase 1 verbatim (same grid-stride assignment,
// same call to the unmodified `reduceBlock` helper, same per-block partial
// written to g_odata[blockIdx.x]), and `reduceCombinePartials` is phase 2
// verbatim (the same sequential "add all other blocks onto slot 0" loop),
// just issued as its own kernel after the first one completes -- ordinary
// kernel-launch/stream ordering stands in for the grid-wide barrier.
//
// Both paths touch the same numBlocks*numThreads grid-stride assignment
// and combine partial sums with the identical two-step (warp-tree-then-
// leader-loop, then sequential-across-blocks) arithmetic; only *how the
// two phases are joined* differs -- a single cooperative-launch kernel
// with grid.sync() vs. two plain kernel launches -- which isolates the
// grid-sync cooperative-launch technique as the sole difference between
// the two code paths.
//
// Deterministic input (replicated by reference.h):
//   input[i] = ((i % 23) - 11) * 0.5, n = 1<<18 = 262144 doubles.
// Every value is an exact multiple of 0.5 with |input[i]| <= 5.5, chosen so
// that no addition in either reduction (in any grouping or order) can ever
// incur floating-point rounding error -- see reference.h for the full
// argument. This makes max_abs_error == 0 an exact oracle for both GPU
// paths despite their different internal summation groupings.
//
// Output: argv[1] (default output/cuda_output.txt), one line: the scalar
// sum computed by the cooperative single-pass kernel.

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

// --- begin: verbatim from NVIDIA/cuda-samples reductionMultiBlockCG.cu ---
// (reduceSinglePassMultiBlockCG adapted: float* g_idata/g_odata -> double*,
// see header comment above; reduceBlock is completely unmodified)

__device__ void reduceBlock(double *sdata, const cg::thread_block &cta)
{
    const unsigned int        tid    = cta.thread_rank();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

    sdata[tid] = cg::reduce(tile32, sdata[tid], cg::plus<double>());
    cg::sync(cta);

    double beta = 0.0;
    if (cta.thread_rank() == 0) {
        beta = 0;
        for (int i = 0; i < blockDim.x; i += tile32.size()) {
            beta += sdata[i];
        }
        sdata[0] = beta;
    }
    cg::sync(cta);
}

extern "C" __global__ void reduceSinglePassMultiBlockCG(const double *g_idata, double *g_odata, unsigned int n)
{
    // Handle to thread block group
    cg::thread_block block = cg::this_thread_block();
    cg::grid_group   grid  = cg::this_grid();

    extern double __shared__ sdata[];

    // Stride over grid and add the values to a shared memory buffer
    sdata[block.thread_rank()] = 0;

    for (int i = grid.thread_rank(); i < n; i += grid.size()) {
        sdata[block.thread_rank()] += g_idata[i];
    }

    cg::sync(block);

    // Reduce each block (called once per block)
    reduceBlock(sdata, block);
    // Write out the result to global memory
    if (block.thread_rank() == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
    cg::sync(grid);

    if (grid.thread_rank() == 0) {
        for (int block = 1; block < gridDim.x; block++) {
            g_odata[0] += g_odata[block];
        }
    }
}

// --- end: verbatim from NVIDIA/cuda-samples ---

// --- begin: new code for this repository -- splits the exact same two
// phases of reduceSinglePassMultiBlockCG above across two ordinary kernel
// launches, with no cg::grid_group / grid.sync() / cooperative launch at
// all, to isolate the grid-sync technique as the sole difference. ---

// Phase 1 (same as reduceSinglePassMultiBlockCG's first half): each block
// reduces its grid-stride share of g_idata into one partial sum, using the
// same unmodified reduceBlock() helper, written to g_odata[blockIdx.x].
__global__ void reduceBlockPartial(const double *g_idata, double *g_odata, unsigned int n)
{
    cg::thread_block block = cg::this_thread_block();

    extern double __shared__ sdata[];

    sdata[block.thread_rank()] = 0;

    unsigned int total_threads = gridDim.x * blockDim.x;
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += total_threads) {
        sdata[block.thread_rank()] += g_idata[i];
    }

    cg::sync(block);

    reduceBlock(sdata, block);
    if (block.thread_rank() == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

// Phase 2 (same as reduceSinglePassMultiBlockCG's tail, just as its own
// kernel launched *after* reduceBlockPartial completes, instead of after an
// in-kernel grid.sync()): sequentially add every other block's partial sum
// onto slot 0.
__global__ void reduceCombinePartials(double *g_odata, unsigned int numBlocks)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (unsigned int block = 1; block < numBlocks; ++block) {
            g_odata[0] += g_odata[block];
        }
    }
}

// --- end: new code for this repository ---

int main(int argc, char **argv) {
  const long n = 1L << 18;  // 262144 doubles
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  const size_t bytes = (size_t)n * sizeof(double);

  int dev = 0;
  CHECK(cudaSetDevice(dev));
  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, dev));
  if (!prop.cooperativeLaunch) {
    fprintf(stderr,
            "device %d does not support cooperative kernel launch; "
            "this case requires it. Skipping.\n",
            dev);
    return 0;
  }

  double *hInput = (double *)malloc(bytes);
  for (long i = 0; i < n; ++i) hInput[i] = gen_val(i);

  double *dInput;
  CHECK(cudaMalloc(&dInput, bytes));
  CHECK(cudaMemcpy(dInput, hInput, bytes, cudaMemcpyHostToDevice));

  // Fixed, modest launch configuration, clamped to what the device can
  // actually run as one cooperative grid (all blocks must be co-resident
  // at once for grid.sync() to be legal). The exactness of both reductions
  // below (see reference.h) holds for *any* numBlocks/threadsPerBlock, so
  // this clamp only affects performance, never correctness.
  const int threadsPerBlock = 128;
  int numBlocksPerSm = 0;
  CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocksPerSm, reduceSinglePassMultiBlockCG, threadsPerBlock,
      threadsPerBlock * (int)sizeof(double)));
  int numBlocks = numBlocksPerSm * prop.multiProcessorCount;
  if (numBlocks > 64) numBlocks = 64;  // keep the grid modest
  if (numBlocks < 1) numBlocks = 1;

  const size_t partialBytes = (size_t)numBlocks * sizeof(double);
  const int smemSize = threadsPerBlock * (int)sizeof(double);

  double *dOutCG, *dOutMulti;
  CHECK(cudaMalloc(&dOutCG, partialBytes));
  CHECK(cudaMalloc(&dOutMulti, partialBytes));

  // --- Path 1: single cooperative kernel, grid-wide sync. ---
  unsigned int un = (unsigned int)n;
  void *kernelArgs[] = {(void *)&dInput, (void *)&dOutCG, (void *)&un};
  dim3 dimBlock(threadsPerBlock, 1, 1);
  dim3 dimGrid(numBlocks, 1, 1);
  CHECK(cudaLaunchCooperativeKernel((void *)reduceSinglePassMultiBlockCG, dimGrid, dimBlock,
                                     kernelArgs, smemSize, NULL));
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  // --- Path 2: two ordinary kernel launches, no grid.sync(). ---
  reduceBlockPartial<<<dimGrid, dimBlock, smemSize>>>(dInput, dOutMulti, un);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  reduceCombinePartials<<<1, 1>>>(dOutMulti, (unsigned int)numBlocks);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  double hResultCG = 0.0, hResultMulti = 0.0;
  CHECK(cudaMemcpy(&hResultCG, dOutCG, sizeof(double), cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(&hResultMulti, dOutMulti, sizeof(double), cudaMemcpyDeviceToHost));

  // CPU reference check.
  double href = reference_reduce_sum(n);

  double err_cg = hResultCG - href;
  double err_multi = hResultMulti - href;
  if (err_cg < 0) err_cg = -err_cg;
  if (err_multi < 0) err_multi = -err_multi;

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  fprintf(f, "%.17g\n", hResultCG);
  fclose(f);

  printf("gridSyncCGReduction done: n=%ld, numBlocks=%d, threadsPerBlock=%d, "
         "result(cooperative)=%.17g, result(multi-launch)=%.17g, reference=%.17g, "
         "max_abs_error(cooperative vs ref)=%.3e, max_abs_error(multi-launch vs ref)=%.3e -> %s\n",
         n, numBlocks, threadsPerBlock, hResultCG, hResultMulti, href, err_cg, err_multi, out);
  printf("%s\n", (err_cg == 0.0 && err_multi == 0.0) ? "PASS" : "FAIL");

  cudaFree(dInput);
  cudaFree(dOutCG);
  cudaFree(dOutMulti);
  free(hInput);
  return 0;
}
