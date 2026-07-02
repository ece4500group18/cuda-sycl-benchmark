// separableConvHaloTiling: separable 2D convolution (row pass then column
// pass), shared-memory halo-tile kernel vs. naive kernel that re-reads halo
// pixels directly from global memory every time.
//
// The two __global__ kernels below (convolutionRowsKernel,
// convolutionColumnsKernel) are reproduced, unmodified, from:
//   NVIDIA/cuda-samples, cpp/2_Concepts_and_Techniques/convolutionSeparable/convolutionSeparable.cu
//   https://github.com/NVIDIA/cuda-samples/blob/master/cpp/2_Concepts_and_Techniques/convolutionSeparable/convolutionSeparable.cu
//   Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//   SPDX-License-Identifier: BSD-3-Clause
//
// Each thread block cooperatively stages a ROWS_BLOCKDIM_Y x ((ROWS_RESULT_
// STEPS + 2*ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X) tile of the source image
// (main data plus left/right -- or, for the column kernel, upper/lower --
// halo) into `__shared__ s_Data` exactly once, with __syncthreads()
// (via cg::sync) as the single synchronization point, and then every
// output pixel produced by that block reads its 9-tap neighborhood back
// out of that shared-memory tile instead of touching global memory again.
// Each shared-memory element loaded is subsequently read by up to
// KERNEL_LENGTH=9 different output computations -- the canonical
// "load the halo into shared memory once, reuse from shared memory many
// times" memory-movement technique for stencil/convolution kernels.
//
// convolutionRowsNaiveKernel / convolutionColumnsNaiveKernel below are new
// code written for this repository: one thread per output pixel, computing
// the exact same 9-tap weighted sum, in the exact same left-to-right
// accumulation order, but reading every input pixel -- including the
// pixels a neighboring thread already read as its own "halo" -- straight
// from global memory, with no shared-memory staging and no reuse. Every
// interior pixel of the image is therefore re-fetched from global memory
// up to 9 times (once per output pixel it contributes to) instead of the
// single shared-memory load the tiled kernel performs. Both kernel pairs
// compute the identical math over the identical data; only *how many times
// each pixel is moved out of global memory, and through which memory
// space* differs -- isolating shared-memory halo-tile reuse as the sole
// memory-movement technique under test.
//
// Deterministic inputs (replicated by reference.h):
//   image[y*imageW+x] = ((x+y) % 13) - 6           (integers in [-6, 6])
//   kernel[i]         = ((i % 5) - 2) * 0.25        (multiples of 0.25 in [-0.5, 0.5])
// imageW = imageH = 256 (a multiple of every block/tile-size divisibility
// constraint required by the tiled kernels' launch configuration, see
// convolutionRowsGPU/convolutionColumnsGPU in the upstream sample).
// KERNEL_RADIUS is fixed at 4 (KERNEL_LENGTH = 9) here -- the upstream
// sample defaults to KERNEL_RADIUS=8, but the kernel bodies themselves are
// radius-agnostic (KERNEL_RADIUS is a compile-time constant substituted at
// build time), and radius 4 keeps ROWS_BLOCKDIM_X*ROWS_HALO_STEPS=16 >= 4
// and COLUMNS_BLOCKDIM_Y*COLUMNS_HALO_STEPS=8 >= 4 (the upstream sample's
// own asserts) comfortably satisfied while keeping the benchmark small.
// Image and kernel values are deliberately dyadic (integers and
// multiples of 0.25) so every partial product/sum in the convolution is
// exactly representable in IEEE-754 float -- see reference.h for why that
// makes an exact (not tolerance-bounded) CPU/GPU comparison valid here.
//
// Output: argv[1] (default output/output.txt), the shared-memory tiled
// kernel's final image (row pass then column pass), one float per line,
// row-major.

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

#define KERNEL_RADIUS 4
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)

__constant__ float c_Kernel[KERNEL_LENGTH];

// --- begin: verbatim from NVIDIA/cuda-samples convolutionSeparable.cu ---

#define ROWS_BLOCKDIM_X   16
#define ROWS_BLOCKDIM_Y   4
#define ROWS_RESULT_STEPS 8
#define ROWS_HALO_STEPS   1

__global__ void convolutionRowsKernel(float *d_Dst, float *d_Src, int imageW, int imageH, int pitch)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    __shared__ float s_Data[ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

    // Offset to the left halo edge
    const int baseX = (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
    const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;

    d_Src += baseY * pitch + baseX;
    d_Dst += baseY * pitch + baseX;

// Load main data
#pragma unroll

    for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++) {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = d_Src[i * ROWS_BLOCKDIM_X];
    }

// Load left halo
#pragma unroll

    for (int i = 0; i < ROWS_HALO_STEPS; i++) {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] =
            (baseX >= -i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
    }

// Load right halo
#pragma unroll

    for (int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++) {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] =
            (imageW - baseX > i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
    }

    // Compute and store results
    cg::sync(cta);
#pragma unroll

    for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++) {
        float sum = 0;

#pragma unroll

        for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {
            sum += c_Kernel[KERNEL_RADIUS - j] * s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j];
        }

        d_Dst[i * ROWS_BLOCKDIM_X] = sum;
    }
}

#define COLUMNS_BLOCKDIM_X   16
#define COLUMNS_BLOCKDIM_Y   8
#define COLUMNS_RESULT_STEPS 8
#define COLUMNS_HALO_STEPS   1

__global__ void convolutionColumnsKernel(float *d_Dst, float *d_Src, int imageW, int imageH, int pitch)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    __shared__ float s_Data[COLUMNS_BLOCKDIM_X]
                           [(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];

    // Offset to the upper halo edge
    const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
    const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
    d_Src += baseY * pitch + baseX;
    d_Dst += baseY * pitch + baseX;

// Main data
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++) {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = d_Src[i * COLUMNS_BLOCKDIM_Y * pitch];
    }

// Upper halo
#pragma unroll

    for (int i = 0; i < COLUMNS_HALO_STEPS; i++) {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] =
            (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
    }

// Lower halo
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS;
         i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS;
         i++) {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] =
            (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
    }

    // Compute and store results
    cg::sync(cta);
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++) {
        float sum = 0;
#pragma unroll

        for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {
            sum += c_Kernel[KERNEL_RADIUS - j] * s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];
        }

        d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch] = sum;
    }
}

// --- end: verbatim from NVIDIA/cuda-samples ---

// --- begin: new code for this repository -- naive, non-tiled counterparts ---
//
// One thread per output pixel. Same 9-tap weighted sum, same left-to-right
// accumulation order (j = -KERNEL_RADIUS .. KERNEL_RADIUS), same
// "out-of-image -> 0" boundary rule as the tiled kernels above, but every
// input pixel (including ones a neighboring thread also needs) is read
// directly from global memory -- no shared-memory tile, no reuse.

__global__ void convolutionRowsNaiveKernel(float *d_Dst, float *d_Src, int imageW, int imageH)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= imageW || y >= imageH) return;

    float sum = 0;
#pragma unroll
    for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {
        int sx = x + j;
        float v = (sx >= 0 && sx < imageW) ? d_Src[y * imageW + sx] : 0.0f;
        sum += c_Kernel[KERNEL_RADIUS - j] * v;
    }
    d_Dst[y * imageW + x] = sum;
}

__global__ void convolutionColumnsNaiveKernel(float *d_Dst, float *d_Src, int imageW, int imageH)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= imageW || y >= imageH) return;

    float sum = 0;
#pragma unroll
    for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++) {
        int sy = y + j;
        float v = (sy >= 0 && sy < imageH) ? d_Src[sy * imageW + x] : 0.0f;
        sum += c_Kernel[KERNEL_RADIUS - j] * v;
    }
    d_Dst[y * imageW + x] = sum;
}

// --- end: new code for this repository ---

int main(int argc, char **argv) {
  const int imageW = 256, imageH = 256;
  const size_t numPix = (size_t)imageW * imageH;
  const size_t bytes = numPix * sizeof(float);
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";

  // Host-side setup: deterministic image and kernel weights.
  float *hSrc = (float *)malloc(bytes);
  for (int y = 0; y < imageH; ++y)
    for (int x = 0; x < imageW; ++x) hSrc[y * imageW + x] = (float)gen_image(x, y);

  float hKernel[KERNEL_LENGTH];
  for (int i = 0; i < KERNEL_LENGTH; ++i) hKernel[i] = (float)gen_kernel(i);

  CHECK(cudaMemcpyToSymbol(c_Kernel, hKernel, KERNEL_LENGTH * sizeof(float)));

  float *dSrc, *dBufTiled, *dFinalTiled, *dBufNaive, *dFinalNaive;
  CHECK(cudaMalloc(&dSrc, bytes));
  CHECK(cudaMalloc(&dBufTiled, bytes));
  CHECK(cudaMalloc(&dFinalTiled, bytes));
  CHECK(cudaMalloc(&dBufNaive, bytes));
  CHECK(cudaMalloc(&dFinalNaive, bytes));

  CHECK(cudaMemcpy(dSrc, hSrc, bytes, cudaMemcpyHostToDevice));

  // Shared-memory tiled path (upstream launch configuration).
  dim3 rowsBlocks(imageW / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X), imageH / ROWS_BLOCKDIM_Y);
  dim3 rowsThreads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y);
  convolutionRowsKernel<<<rowsBlocks, rowsThreads>>>(dBufTiled, dSrc, imageW, imageH, imageW);
  CHECK(cudaGetLastError());

  dim3 colsBlocks(imageW / COLUMNS_BLOCKDIM_X, imageH / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y));
  dim3 colsThreads(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);
  convolutionColumnsKernel<<<colsBlocks, colsThreads>>>(dFinalTiled, dBufTiled, imageW, imageH, imageW);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  // Naive, one-thread-per-pixel, direct-global-memory path.
  dim3 naiveThreads(16, 16);
  dim3 naiveBlocks(imageW / naiveThreads.x, imageH / naiveThreads.y);
  convolutionRowsNaiveKernel<<<naiveBlocks, naiveThreads>>>(dBufNaive, dSrc, imageW, imageH);
  CHECK(cudaGetLastError());

  convolutionColumnsNaiveKernel<<<naiveBlocks, naiveThreads>>>(dFinalNaive, dBufNaive, imageW, imageH);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  float *hFinalTiled = (float *)malloc(bytes);
  float *hFinalNaive = (float *)malloc(bytes);
  CHECK(cudaMemcpy(hFinalTiled, dFinalTiled, bytes, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(hFinalNaive, dFinalNaive, bytes, cudaMemcpyDeviceToHost));

  // CPU reference check.
  double *hImgD = (double *)malloc(numPix * sizeof(double));
  double *hKernelD = (double *)malloc(KERNEL_LENGTH * sizeof(double));
  double *hBufRef = (double *)malloc(numPix * sizeof(double));
  double *hFinalRef = (double *)malloc(numPix * sizeof(double));
  for (size_t i = 0; i < numPix; ++i) hImgD[i] = (double)hSrc[i];
  for (int i = 0; i < KERNEL_LENGTH; ++i) hKernelD[i] = (double)hKernel[i];

  reference_convolve_rows(hKernelD, hImgD, hBufRef, imageW, imageH);
  reference_convolve_columns(hKernelD, hBufRef, hFinalRef, imageW, imageH);

  double max_abs_err_tiled = 0.0, max_abs_err_naive = 0.0;
  for (size_t i = 0; i < numPix; ++i) {
    double dt = (double)hFinalTiled[i] - hFinalRef[i];
    double dn = (double)hFinalNaive[i] - hFinalRef[i];
    if (dt < 0) dt = -dt;
    if (dn < 0) dn = -dn;
    if (dt > max_abs_err_tiled) max_abs_err_tiled = dt;
    if (dn > max_abs_err_naive) max_abs_err_naive = dn;
  }

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (size_t i = 0; i < numPix; ++i) fprintf(f, "%.9g\n", hFinalTiled[i]);
  fclose(f);

  printf("separableConvHaloTiling done: imageW=%d, imageH=%d, KERNEL_RADIUS=%d, "
         "max_abs_error(tiled vs ref)=%.3e, max_abs_error(naive vs ref)=%.3e -> %s\n",
         imageW, imageH, KERNEL_RADIUS, max_abs_err_tiled, max_abs_err_naive, out);
  printf("%s\n", (max_abs_err_tiled == 0.0 && max_abs_err_naive == 0.0) ? "PASS" : "FAIL");

  cudaFree(dSrc);
  cudaFree(dBufTiled);
  cudaFree(dFinalTiled);
  cudaFree(dBufNaive);
  cudaFree(dFinalNaive);
  free(hSrc);
  free(hFinalTiled);
  free(hFinalNaive);
  free(hImgD);
  free(hKernelD);
  free(hBufRef);
  free(hFinalRef);
  return 0;
}
