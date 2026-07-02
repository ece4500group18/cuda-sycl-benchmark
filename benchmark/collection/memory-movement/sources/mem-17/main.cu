// textureInterpolationGather: rotate an image by reading it through a CUDA
// texture object (hardware bilinear interpolation, normalized coordinates,
// wrap addressing) vs. a hand-written software bilinear gather from a plain
// global-memory array at the same rotated coordinates.
//
// The __global__ kernel below (transformKernel) is reproduced, unmodified,
// from:
//   NVIDIA/cuda-samples, cpp/0_Introduction/simpleTexture/simpleTexture.cu
//   Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//   SPDX-License-Identifier: BSD-3-Clause
//
// "transformKernel": for each output pixel (x, y), computes rotated texture
//   coordinates (tu, tv) and reads the source image with
//   tex2D<float>(tex, tu + 0.5f, tv + 0.5f), where `tex` is a texture object
//   configured with normalizedCoords = true, filterMode = cudaFilterModeLinear,
//   addressMode = cudaAddressModeWrap. The texture unit converts the
//   normalized coordinate to texel space, wraps it into the image bounds,
//   and blends the 4 nearest texels using fixed-point interpolation weights
//   -- entirely in dedicated texture-sampling hardware.
//
// "softwareGatherKernel" (new code, written for this repository) performs
//   the identical rotation and then reproduces the texture unit's
//   normalize -> wrap -> unnormalize -> bilinear-blend pipeline by hand,
//   reading four explicit elements out of a plain `float*` global-memory
//   array (no texture/surface object, no cudaArray) and blending them with
//   ordinary IEEE-754 float arithmetic.
//
// Both kernels read the same source image and are meant to compute the same
// rotated, bilinearly-interpolated pixel value; only *how* the 2x2
// neighborhood is fetched and blended (fixed-function texture-sampling
// hardware with fixed-point interpolation weights, vs. an explicit
// global-memory gather blended with IEEE float arithmetic) differs. Because
// the texture unit's interpolation weights are quantized to a fixed-point
// fraction (not full float precision), the two paths are only guaranteed to
// agree to a small, hardware-precision-derived tolerance -- not bit-exactly
// -- which is documented and justified in README.md.
//
// Deterministic inputs (replicated by reference.h):
//   img[y*width+x] = ((x*3 + y) % 23) - 11
// width = height = 64, rotation angle = 0.5 rad (matches the original
// sample's fixed `angle` constant).
//
// Output: argv[1] (default output/output.txt), one float per line, the
// width*height rotated image produced by the texture-object path
// (transformKernel), row-major.

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

// --- begin: verbatim from NVIDIA/cuda-samples simpleTexture.cu ---

__global__ void transformKernel(float *outputData, int width, int height, float theta, cudaTextureObject_t tex)
{
    // calculate normalized texture coordinates
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    float u  = (float)x - (float)width / 2;
    float v  = (float)y - (float)height / 2;
    float tu = u * cosf(theta) - v * sinf(theta);
    float tv = v * cosf(theta) + u * sinf(theta);

    tu /= (float)width;
    tv /= (float)height;

    // read from texture and write to global memory
    outputData[y * width + x] = tex2D<float>(tex, tu + 0.5f, tv + 0.5f);
}

// --- end: verbatim from NVIDIA/cuda-samples ---

// --- begin: new code for this repository ---
//
// softwareGatherKernel reproduces, by hand, exactly what the texture unit
// above does for tex2D<float>(tex, tu + 0.5f, tv + 0.5f) with
// normalizedCoords = true, addressMode = cudaAddressModeWrap,
// filterMode = cudaFilterModeLinear:
//   1. wrap the normalized coordinate into [0, 1)
//   2. convert to texel space, where texel i's center is at (i + 0.5)
//   3. take the 2x2 neighborhood, wrapping texel indices into [0, dim)
//   4. bilinearly blend the 4 samples with the fractional offsets
// but does so by indexing a plain float* array in global memory instead of
// going through a cudaArray/texture object, and blends with ordinary
// (non-fixed-point) float arithmetic.
__global__ void softwareGatherKernel(float *outputData, const float *inputData,
                                      int width, int height, float theta)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    float u  = (float)x - (float)width / 2;
    float v  = (float)y - (float)height / 2;
    float tu = u * cosf(theta) - v * sinf(theta);
    float tv = v * cosf(theta) + u * sinf(theta);

    tu /= (float)width;
    tv /= (float)height;

    float un = tu + 0.5f;
    float vn = tv + 0.5f;

    // cudaAddressModeWrap on normalized coordinates: wrap into [0, 1).
    un = un - floorf(un);
    vn = vn - floorf(vn);

    // Un-normalize to texel space (texel i's center sits at i + 0.5).
    float fx = un * (float)width - 0.5f;
    float fy = vn * (float)height - 0.5f;

    int x0 = (int)floorf(fx);
    int y0 = (int)floorf(fy);
    float ax = fx - (float)x0;
    float ay = fy - (float)y0;

    int x1 = x0 + 1;
    int y1 = y0 + 1;

    // Wrap texel indices into [0, width) / [0, height), same as
    // cudaAddressModeWrap does for the hardware path.
    x0 = ((x0 % width) + width) % width;
    x1 = ((x1 % width) + width) % width;
    y0 = ((y0 % height) + height) % height;
    y1 = ((y1 % height) + height) % height;

    float p00 = inputData[y0 * width + x0];
    float p10 = inputData[y0 * width + x1];
    float p01 = inputData[y1 * width + x0];
    float p11 = inputData[y1 * width + x1];

    float top = p00 + ax * (p10 - p00);
    float bot = p01 + ax * (p11 - p01);
    outputData[y * width + x] = top + ay * (bot - top);
}

// --- end: new code for this repository ---

int main(int argc, char **argv) {
  const int width = 64, height = 64;   // multiple of dimBlock below
  const float angle = 0.5f;            // rotation angle, matches upstream default
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  const size_t size = (size_t)width * height * sizeof(float);

  float *hImg = (float *)malloc(size);
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      hImg[y * width + x] = gen_img(x, y);

  // --- texture-object path setup (mirrors upstream runTest()) ---
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  cudaArray *cuArray;
  CHECK(cudaMallocArray(&cuArray, &channelDesc, width, height));
  CHECK(cudaMemcpyToArray(cuArray, 0, 0, hImg, size, cudaMemcpyHostToDevice));

  cudaResourceDesc texRes;
  memset(&texRes, 0, sizeof(cudaResourceDesc));
  texRes.resType = cudaResourceTypeArray;
  texRes.res.array.array = cuArray;

  cudaTextureDesc texDescr;
  memset(&texDescr, 0, sizeof(cudaTextureDesc));
  texDescr.normalizedCoords = true;
  texDescr.filterMode = cudaFilterModeLinear;
  texDescr.addressMode[0] = cudaAddressModeWrap;
  texDescr.addressMode[1] = cudaAddressModeWrap;
  texDescr.readMode = cudaReadModeElementType;

  cudaTextureObject_t tex;
  CHECK(cudaCreateTextureObject(&tex, &texRes, &texDescr, NULL));

  // --- plain global-memory path setup ---
  float *dImgLinear;
  CHECK(cudaMalloc(&dImgLinear, size));
  CHECK(cudaMemcpy(dImgLinear, hImg, size, cudaMemcpyHostToDevice));

  float *dOutTex, *dOutSoft;
  CHECK(cudaMalloc(&dOutTex, size));
  CHECK(cudaMalloc(&dOutSoft, size));

  dim3 dimBlock(8, 8, 1);
  dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

  transformKernel<<<dimGrid, dimBlock>>>(dOutTex, width, height, angle, tex);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  softwareGatherKernel<<<dimGrid, dimBlock>>>(dOutSoft, dImgLinear, width, height, angle);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  float *hOutTex = (float *)malloc(size);
  float *hOutSoft = (float *)malloc(size);
  CHECK(cudaMemcpy(hOutTex, dOutTex, size, cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(hOutSoft, dOutSoft, size, cudaMemcpyDeviceToHost));

  // CPU reference check (double precision, same rotate+wrap+bilinear pipeline).
  double *href = (double *)malloc((size_t)width * height * sizeof(double));
  reference_rotate_bilinear_wrap(hImg, href, width, height, (double)angle);

  double max_abs_err_tex = 0.0, max_abs_err_soft = 0.0;
  for (int i = 0; i < width * height; ++i) {
    double d1 = (double)hOutTex[i] - href[i];
    double d2 = (double)hOutSoft[i] - href[i];
    if (d1 < 0) d1 = -d1;
    if (d2 < 0) d2 = -d2;
    if (d1 > max_abs_err_tex) max_abs_err_tex = d1;
    if (d2 > max_abs_err_soft) max_abs_err_soft = d2;
  }

  // Precision-derived tolerances (see README.md "Correctness" discussion):
  //   - the texture path's bilinear weights are quantized to an 8-bit
  //     fixed-point fraction in hardware, vs. full float precision in the
  //     software/reference paths; with image values spanning a range of 22
  //     (-11..11), the resulting worst-case quantization error is bounded by
  //     about 22/256 ~= 0.086, so we use 0.1 as a safe, documented bound.
  //   - the software-gather path uses the exact same formula and operand
  //     order as the reference, so the only disagreement is float (device
  //     cosf/sinf) vs. double (host cos/sin) rounding -- a much tighter,
  //     pure floating-point-rounding bound of 1e-3 is used.
  const double TOL_TEX = 0.1;
  const double TOL_SOFT = 1e-3;

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (int i = 0; i < width * height; ++i) fprintf(f, "%.9g\n", hOutTex[i]);
  fclose(f);

  printf("textureInterpolationGather done: width=%d, height=%d, "
         "max_abs_error(texture vs ref)=%.3e (tol=%.3e), "
         "max_abs_error(software vs ref)=%.3e (tol=%.3e) -> %s\n",
         width, height, max_abs_err_tex, TOL_TEX, max_abs_err_soft, TOL_SOFT, out);
  printf("%s\n", (max_abs_err_tex < TOL_TEX && max_abs_err_soft < TOL_SOFT) ? "PASS" : "FAIL");

  CHECK(cudaDestroyTextureObject(tex));
  CHECK(cudaFreeArray(cuArray));
  cudaFree(dImgLinear);
  cudaFree(dOutTex);
  cudaFree(dOutSoft);
  free(hImg);
  free(hOutTex);
  free(hOutSoft);
  free(href);
  return 0;
}
