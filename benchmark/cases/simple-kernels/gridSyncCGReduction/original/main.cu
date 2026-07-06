// gridSyncCGReduction: single-kernel cooperative-groups grid-wide sum reduction
// (cg::this_grid() + grid.sync(), launched via cudaLaunchCooperativeKernel) vs.
// the classic two-kernel-launch tree reduction combining the same per-block
// partial sums without any cooperative launch or grid-wide barrier.
//
// reduceBlock and reduceSinglePassMultiBlockCG below are reproduced from:
//   NVIDIA/cuda-samples,
//   cpp/2_Concepts_and_Techniques/reductionMultiBlockCG/reductionMultiBlockCG.cu
//   Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//   SPDX-License-Identifier: BSD-3-Clause
// reduceBlock is unmodified; reduceSinglePassMultiBlockCG has one mechanical
// change: float* g_idata/g_odata -> double* (so per-block partials keep double
// precision, required for the exact-match oracle). reduceBlockPartial and
// reduceCombinePartials are NEW kernels splitting the same two phases across
// two ordinary launches, isolating the grid-sync cooperative-launch technique.
//
// Deterministic input (reproduced by tests/verify.py):
//   input[i] = ((i % 23) - 11) * 0.5, n = 1<<18 = 262144 doubles. Every value
// is a multiple of 0.5 with |.|<=5.5, so every partial sum in any order is
// exactly representable -> exact oracle.
//
// Output: argv[1] (default output/cuda_output.txt), one line: the scalar sum
// from the cooperative single-pass kernel.

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

// --- begin: verbatim from NVIDIA/cuda-samples reductionMultiBlockCG.cu ---
// (reduceSinglePassMultiBlockCG adapted: float* g_idata/g_odata -> double*;
// reduceBlock is completely unmodified)

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

// --- begin: new code for this repository -- splits the exact same two phases
// across two ordinary (non-cooperative) kernel launches, no grid.sync(). ---

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
  const char *out = (argc > 1) ? argv[1] : "output/cuda_output.txt";
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
  for (long i = 0; i < n; ++i) hInput[i] = (double)((i % 23) - 11) * 0.5;

  double *dInput;
  CHECK(cudaMalloc(&dInput, bytes));
  CHECK(cudaMemcpy(dInput, hInput, bytes, cudaMemcpyHostToDevice));

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

  // Path 1: single cooperative kernel, grid-wide sync.
  unsigned int un = (unsigned int)n;
  void *kernelArgs[] = {(void *)&dInput, (void *)&dOutCG, (void *)&un};
  dim3 dimBlock(threadsPerBlock, 1, 1);
  dim3 dimGrid(numBlocks, 1, 1);
  CHECK(cudaLaunchCooperativeKernel((void *)reduceSinglePassMultiBlockCG, dimGrid, dimBlock,
                                     kernelArgs, smemSize, NULL));
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  // Path 2: two ordinary kernel launches, no grid.sync().
  reduceBlockPartial<<<dimGrid, dimBlock, smemSize>>>(dInput, dOutMulti, un);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  reduceCombinePartials<<<1, 1>>>(dOutMulti, (unsigned int)numBlocks);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  double hResultCG = 0.0;
  CHECK(cudaMemcpy(&hResultCG, dOutCG, sizeof(double), cudaMemcpyDeviceToHost));

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  fprintf(f, "%.17g\n", hResultCG);
  fclose(f);

  cudaFree(dInput);
  cudaFree(dOutCG);
  cudaFree(dOutMulti);
  free(hInput);
  return 0;
}
