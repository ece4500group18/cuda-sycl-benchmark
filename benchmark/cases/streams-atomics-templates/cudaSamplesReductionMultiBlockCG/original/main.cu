// Single-pass grid-wide sum reduction using cooperative groups: warp-tile
// cg::reduce, block reduction, then a grid.sync() before the final
// cross-block accumulation — launched with cudaLaunchCooperativeKernel.
//
// Extracted from NVIDIA/cuda-samples
// 2_Concepts_and_Techniques/reductionMultiBlockCG (reductionMultiBlockCG.cu).
// Upstream: @ b7c5481c (BSD-3-Clause, NVIDIA).
// reduceBlock, reduceSinglePassMultiBlockCG and the cooperative-launch
// wrapper are upstream code verbatim. The harness feeds deterministic hash
// data, sizes the grid by occupancy like upstream, and dumps the sum.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);exit(2);}}

namespace cg = cooperative_groups;

// ---- upstream device code (verbatim) -------------------------------------------
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

// This reduction kernel reduces an arbitrary size array in a single kernel
// invocation
extern "C" __global__ void reduceSinglePassMultiBlockCG(const float *g_idata, float *g_odata, unsigned int n)
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

// Wrapper function for kernel launch (upstream, verbatim)
void call_reduceSinglePassMultiBlockCG(int size, int threads, int numBlocks, float *d_idata, float *d_odata)
{
    int   smemSize     = threads * sizeof(double);
    void *kernelArgs[] = {
        (void *)&d_idata,
        (void *)&d_odata,
        (void *)&size,
    };

    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(numBlocks, 1, 1);

    cudaLaunchCooperativeKernel((void *)reduceSinglePassMultiBlockCG, dimGrid, dimBlock, kernelArgs, smemSize, NULL);
}
// ---- end upstream code -----------------------------------------------------------

static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

int main(int argc, char **argv) {
  const unsigned int n = 1 << 20;
  const int threads = 256;
  const char *out_path = (argc > 1) ? argv[1] : "output/output.txt";

  cudaDeviceProp prop = {};
  CK(cudaGetDeviceProperties(&prop, 0));
  if (!prop.cooperativeLaunch) {
    fprintf(stderr, "device lacks cooperative launch support\n");
    return 2;
  }

  // Size the grid to what a cooperative launch can hold (upstream approach).
  int numBlocksPerSm = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocksPerSm, reduceSinglePassMultiBlockCG, threads, threads * sizeof(double)));
  int numBlocks = prop.multiProcessorCount * numBlocksPerSm;
  if (numBlocks > (int)((n + threads - 1) / threads)) numBlocks = (n + threads - 1) / threads;

  float *h = (float*)malloc(n * sizeof(float));
  for (unsigned int i = 0; i < n; ++i) h[i] = 2.0f * h01(i, 131) - 1.0f;

  float *d_idata, *d_odata;
  CK(cudaMalloc(&d_idata, n * sizeof(float)));
  CK(cudaMalloc(&d_odata, numBlocks * sizeof(float)));
  CK(cudaMemcpy(d_idata, h, n * sizeof(float), cudaMemcpyHostToDevice));

  call_reduceSinglePassMultiBlockCG(n, threads, numBlocks, d_idata, d_odata);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

  float sum = 0.0f;
  CK(cudaMemcpy(&sum, d_odata, sizeof(float), cudaMemcpyDeviceToHost));

  FILE *f = fopen(out_path, "w");
  if (!f) { fprintf(stderr, "open %s\n", out_path); return 2; }
  fprintf(f, "%.9g\n", sum);
  fclose(f);

  cudaFree(d_idata); cudaFree(d_odata); free(h);
  return 0;
}
