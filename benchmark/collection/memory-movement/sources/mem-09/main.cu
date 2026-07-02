// layeredTextureGather: read a 3D-layered (cudaArrayLayered) texture with
// tex2DLayered() vs. read the identical data directly out of a flat
// global-memory array -- both computing the exact same
// `-src[layer]+layer` transform per element.
//
// The kernel below (transformKernel) is reproduced, unmodified except for
// formatting, from:
//   NVIDIA/cuda-samples, cpp/0_Introduction/simpleLayeredTexture/simpleLayeredTexture.cu
//   Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//   SPDX-License-Identifier: BSD-3-Clause
//
// transformKernel fetches its layer's texel through the CUDA texture
// pipeline (cudaArrayLayered array + cudaTextureObject_t + tex2DLayered),
// which is backed by an opaque, hardware-defined memory layout distinct
// from a linear buffer (and, on real hardware, is routed through the
// dedicated texture cache/samplers). transformKernelFlat -- new code
// written for this repository -- performs the *identical* arithmetic
// (`-src[idx] + layer`) but reads `idx` straight out of a flat,
// linearly-laid-out global-memory array with an ordinary pointer load.
// Both kernels touch the same logical elements and apply the same
// transform; only the memory layout/access mechanism used to fetch the
// source value differs -- the "layered texture vs. linear array" memory
// layout comparison this case isolates.
//
// The upstream sample defaults to `cudaFilterModeLinear` (bilinear)
// texture filtering, but samples exactly at texel centers
// (u = (x+0.5)/width), so with linear filtering the fetch would still
// land exactly on one texel with a bilinear weight of 1.0 in exact
// arithmetic -- except real GPU texture units use fixed-point
// interpolation weights, which can introduce tiny rounding error even at
// weight 1.0. To get a bit-exact oracle we instead configure the texture
// with `cudaFilterModePoint` (nearest/point sampling), which returns the
// backing texel's value with no interpolation at all, guaranteeing
// max_abs_error == 0 against a flat-array CPU reference.
//
// Deterministic input (replicated by reference.h):
//   data[i] = (i % 19) - 9 for i in [0, width*height), replicated
//   identically into all `num_layers` layers.
// width = height = 32, num_layers = 4 (matches the "8 x 8 threads per
// block, dimGrid = width/8 x height/8" launch configuration of the
// original sample).
//
// Output: argv[1] (default output/cuda_output.txt), one float per line,
// the texture-path result (`h_odata_tex`), width*height*num_layers lines.

#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// --- begin: verbatim from NVIDIA/cuda-samples simpleLayeredTexture.cu ---

__global__ void transformKernel(float *g_odata, int width, int height, int layer, cudaTextureObject_t tex)
{
    // calculate this thread's data point
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 0.5f offset and division are necessary to access the original data points
    // in the texture (such that bilinear interpolation will not be activated).
    // For details, see also CUDA Programming Guide, Appendix D
    float u = (x + 0.5f) / (float)width;
    float v = (y + 0.5f) / (float)height;

    // read from texture, do expected transformation and write to global memory
    g_odata[layer * width * height + y * width + x] = -tex2DLayered<float>(tex, u, v, layer) + layer;
}

// --- end: verbatim from NVIDIA/cuda-samples ---

// New kernel (not from upstream): identical "-src+layer" transform, but
// reading the source element directly out of a flat, linearly-laid-out
// global-memory array instead of through the texture/layered-array path.
__global__ void transformKernelFlat(float *g_odata, const float *g_idata, int width, int height, int layer)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int idx = layer * width * height + y * width + x;
    g_odata[idx] = -g_idata[idx] + layer;
}

int main(int argc, char **argv) {
  const unsigned int width = 32, height = 32, num_layers = 4;
  const unsigned int layer_elems = width * height;
  const unsigned int total_elems = layer_elems * num_layers;
  const size_t size = (size_t)total_elems * sizeof(float);
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";

  // Host input: same flat pattern replicated into every layer.
  float *h_data = (float *)malloc(size);
  for (unsigned int layer = 0; layer < num_layers; ++layer)
    for (unsigned int i = 0; i < layer_elems; ++i)
      h_data[layer * layer_elems + i] = gen_data((int)i);

  // --- flat-array path ---
  float *d_idata_flat, *d_odata_flat;
  CHECK(cudaMalloc(&d_idata_flat, size));
  CHECK(cudaMalloc(&d_odata_flat, size));
  CHECK(cudaMemcpy(d_idata_flat, h_data, size, cudaMemcpyHostToDevice));

  dim3 dimBlock(8, 8, 1);
  dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

  for (unsigned int layer = 0; layer < num_layers; ++layer) {
    transformKernelFlat<<<dimGrid, dimBlock>>>(d_odata_flat, d_idata_flat, width, height, layer);
    CHECK(cudaGetLastError());
  }
  CHECK(cudaDeviceSynchronize());

  float *h_odata_flat = (float *)malloc(size);
  CHECK(cudaMemcpy(h_odata_flat, d_odata_flat, size, cudaMemcpyDeviceToHost));

  // --- layered-texture path ---
  float *d_odata_tex = NULL;
  CHECK(cudaMalloc(&d_odata_tex, size));

  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  cudaArray *cu_3darray;
  CHECK(cudaMalloc3DArray(&cu_3darray, &channelDesc, make_cudaExtent(width, height, num_layers), cudaArrayLayered));

  cudaMemcpy3DParms myparms;
  memset(&myparms, 0, sizeof(myparms));
  myparms.srcPos = make_cudaPos(0, 0, 0);
  myparms.dstPos = make_cudaPos(0, 0, 0);
  myparms.srcPtr = make_cudaPitchedPtr(h_data, width * sizeof(float), width, height);
  myparms.dstArray = cu_3darray;
  myparms.extent = make_cudaExtent(width, height, num_layers);
  myparms.kind = cudaMemcpyHostToDevice;
  CHECK(cudaMemcpy3D(&myparms));

  cudaTextureObject_t tex;
  cudaResourceDesc texRes;
  memset(&texRes, 0, sizeof(cudaResourceDesc));
  texRes.resType = cudaResourceTypeArray;
  texRes.res.array.array = cu_3darray;

  cudaTextureDesc texDescr;
  memset(&texDescr, 0, sizeof(cudaTextureDesc));
  texDescr.normalizedCoords = true;
  // Point/nearest filtering (upstream defaults to cudaFilterModeLinear):
  // sampling exact texel centers with point filtering guarantees a
  // bit-exact fetch, so this case can use an exact-match oracle instead
  // of an interpolation-error tolerance.
  texDescr.filterMode = cudaFilterModePoint;
  texDescr.addressMode[0] = cudaAddressModeClamp;
  texDescr.addressMode[1] = cudaAddressModeClamp;
  texDescr.readMode = cudaReadModeElementType;

  CHECK(cudaCreateTextureObject(&tex, &texRes, &texDescr, NULL));

  for (unsigned int layer = 0; layer < num_layers; ++layer) {
    transformKernel<<<dimGrid, dimBlock>>>(d_odata_tex, width, height, layer, tex);
    CHECK(cudaGetLastError());
  }
  CHECK(cudaDeviceSynchronize());

  float *h_odata_tex = (float *)malloc(size);
  CHECK(cudaMemcpy(h_odata_tex, d_odata_tex, size, cudaMemcpyDeviceToHost));

  // CPU reference check.
  float *href = (float *)malloc(size);
  reference_layered_transform(href, (int)width, (int)height, (int)num_layers);

  double max_abs_err_tex = 0.0, max_abs_err_flat = 0.0;
  for (unsigned int i = 0; i < total_elems; ++i) {
    double d1 = (double)h_odata_tex[i] - (double)href[i];
    double d2 = (double)h_odata_flat[i] - (double)href[i];
    if (d1 < 0) d1 = -d1;
    if (d2 < 0) d2 = -d2;
    if (d1 > max_abs_err_tex) max_abs_err_tex = d1;
    if (d2 > max_abs_err_flat) max_abs_err_flat = d2;
  }

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (unsigned int i = 0; i < total_elems; ++i) fprintf(f, "%.9g\n", h_odata_tex[i]);
  fclose(f);

  printf("layeredTextureGather done: width=%u, height=%u, num_layers=%u, "
         "max_abs_error(tex vs ref)=%.3e, max_abs_error(flat vs ref)=%.3e -> %s\n",
         width, height, num_layers, max_abs_err_tex, max_abs_err_flat, out);
  printf("%s\n", (max_abs_err_tex == 0.0 && max_abs_err_flat == 0.0) ? "PASS" : "FAIL");

  CHECK(cudaDestroyTextureObject(tex));
  CHECK(cudaFreeArray(cu_3darray));
  cudaFree(d_odata_tex);
  cudaFree(d_idata_flat);
  cudaFree(d_odata_flat);
  free(h_data);
  free(h_odata_tex);
  free(h_odata_flat);
  free(href);
  return 0;
}
