// pitchLinearVsCudaArrayTexture: cudaMallocPitch (pitch-linear) allocation
// vs. cudaMallocArray (CUDA-array) allocation, both bound as texture
// objects with point filtering and wrap addressing, performing the
// identical per-pixel circular (periodic) column/row shift.
//
// The two __global__ kernels below are reproduced, unmodified, from:
//   NVIDIA/cuda-samples, cpp/0_Introduction/simplePitchLinearTexture/simplePitchLinearTexture.cu
//   https://github.com/NVIDIA/cuda-samples/blob/master/cpp/0_Introduction/simplePitchLinearTexture/simplePitchLinearTexture.cu
//   Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
//   SPDX-License-Identifier: BSD-3-Clause
//
// "shiftPitchLinear" reads its input through a texture object bound to a
// plain pitch-linear device allocation (cudaMallocPitch); "shiftArray"
// reads its input through a texture object bound to an opaque CUDA array
// (cudaMallocArray). Both textures use normalized coordinates,
// cudaFilterModePoint (nearest-texel, no interpolation) and
// cudaAddressModeWrap (periodic boundary condition) in both dimensions,
// and both kernels compute the exact same output:
//   odata[y][x] = src[(y + shiftY) % height][(x + shiftX) % width]
// -- i.e. a circular shift of the source grid -- via the identical
// texture-fetch expression `tex2D<float>(tex, (xid+shiftX)/(float)width,
// (yid+shiftY)/(float)height)`. Only the underlying memory layout the
// texture is bound to (linear pitched memory vs. a CUDA-array's
// opaque/blocked/swizzled layout optimized for 2D locality) differs.
//
// width and height are chosen as powers of two (64 x 32) so that every
// coordinate division and the wrap-addressing subtraction the texture
// unit performs (x/width, frac(.) * width, etc.) is an exact power-of-two
// float32 operation with zero rounding error; combined with an
// integer shift, the nearest-texel fetch is therefore guaranteed to land
// exactly on the intended source texel -- no accumulated floating-point
// drift -- so both kernels' output matches a CPU reference exactly
// (max_abs_error == 0), unlike the original sample (2048x2048, a
// non-power-of-two-friendly shift/tolerance combination -- it compares
// with a 0.15 relative tolerance instead).
//
// Deterministic input (replicated by reference.h):
//   src[y*w + x] = ((x*3 + y*5) % 17) - 8
// w = 64, h = 32, shiftX = 5, shiftY = 7, TILE_DIM = 16 (both w and h are
// multiples of TILE_DIM, matching the original sample's own constraint).
//
// Output: argv[1] (default output/output.txt), w*h floats (one per
// line, row-major), the shiftPitchLinear kernel's output.

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

// --- begin: verbatim from NVIDIA/cuda-samples simplePitchLinearTexture.cu ---

////////////////////////////////////////////////////////////////////////////////
// NB: (1) The second argument "pitch" is in elements, not bytes
//     (2) normalized coordinates are used (required for wrap address mode)
////////////////////////////////////////////////////////////////////////////////
//! Shifts matrix elements using pitch linear array
//! @param odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void
shiftPitchLinear(float *odata, int pitch, int width, int height, int shiftX, int shiftY, cudaTextureObject_t texRefPL)
{
    int xid = blockIdx.x * blockDim.x + threadIdx.x;
    int yid = blockIdx.y * blockDim.y + threadIdx.y;

    odata[yid * pitch + xid] = tex2D<float>(texRefPL, (xid + shiftX) / (float)width, (yid + shiftY) / (float)height);
}

////////////////////////////////////////////////////////////////////////////////
//! Shifts matrix elements using regular array
//! @param odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void
shiftArray(float *odata, int pitch, int width, int height, int shiftX, int shiftY, cudaTextureObject_t texRefArray)
{
    int xid = blockIdx.x * blockDim.x + threadIdx.x;
    int yid = blockIdx.y * blockDim.y + threadIdx.y;

    odata[yid * pitch + xid] = tex2D<float>(texRefArray, (xid + shiftX) / (float)width, (yid + shiftY) / (float)height);
}

// --- end: verbatim from NVIDIA/cuda-samples ---

int main(int argc, char **argv) {
  const int nx = 64;   // width  (power of two)
  const int ny = 32;   // height (power of two)
  const int TILE_DIM = 16;
  const int x_shift = 5;
  const int y_shift = 7;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";

  dim3 dimGrid(nx / TILE_DIM, ny / TILE_DIM), dimBlock(TILE_DIM, TILE_DIM);

  const size_t numel = (size_t)nx * ny;
  const size_t bytes = numel * sizeof(float);
  const size_t h_pitchBytes = (size_t)nx * sizeof(float);

  float *h_idata = (float *)malloc(bytes);
  float *h_odataPL = (float *)malloc(bytes);
  float *h_odataArr = (float *)malloc(bytes);
  float *gold = (float *)malloc(bytes);

  for (int y = 0; y < ny; ++y)
    for (int x = 0; x < nx; ++x) h_idata[y * nx + x] = gen_src(x, y);

  // Pitch-linear input allocation.
  float *d_idataPL;
  size_t d_pitchBytes;
  CHECK(cudaMallocPitch((void **)&d_idataPL, &d_pitchBytes, nx * sizeof(float), ny));

  // CUDA-array input allocation.
  cudaArray *d_idataArray;
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  CHECK(cudaMallocArray(&d_idataArray, &channelDesc, nx, ny));

  // Pitch-linear output allocations (one per kernel).
  float *d_odataPL, *d_odataArr;
  size_t d_opitchBytesPL, d_opitchBytesArr;
  CHECK(cudaMallocPitch((void **)&d_odataPL, &d_opitchBytesPL, nx * sizeof(float), ny));
  CHECK(cudaMallocPitch((void **)&d_odataArr, &d_opitchBytesArr, nx * sizeof(float), ny));

  CHECK(cudaMemcpy2D(d_idataPL, d_pitchBytes, h_idata, h_pitchBytes, nx * sizeof(float), ny,
                      cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy2DToArray(d_idataArray, 0, 0, h_idata, h_pitchBytes, nx * sizeof(float), ny,
                             cudaMemcpyHostToDevice));

  cudaTextureObject_t texRefPL, texRefArray;
  cudaResourceDesc texRes;
  cudaTextureDesc texDescr;

  memset(&texRes, 0, sizeof(texRes));
  texRes.resType = cudaResourceTypePitch2D;
  texRes.res.pitch2D.devPtr = d_idataPL;
  texRes.res.pitch2D.desc = channelDesc;
  texRes.res.pitch2D.width = nx;
  texRes.res.pitch2D.height = ny;
  // NB: must be the *actual* device pitch returned by cudaMallocPitch, not
  // the host row stride -- for small widths (like our 64-element rows)
  // the two commonly differ once the allocator's alignment padding kicks
  // in, unlike the original sample's 2048-wide buffers where they happen
  // to coincide.
  texRes.res.pitch2D.pitchInBytes = d_pitchBytes;

  memset(&texDescr, 0, sizeof(texDescr));
  texDescr.normalizedCoords = true;
  texDescr.filterMode = cudaFilterModePoint;
  texDescr.addressMode[0] = cudaAddressModeWrap;
  texDescr.addressMode[1] = cudaAddressModeWrap;
  texDescr.readMode = cudaReadModeElementType;

  CHECK(cudaCreateTextureObject(&texRefPL, &texRes, &texDescr, NULL));

  memset(&texRes, 0, sizeof(texRes));
  texRes.resType = cudaResourceTypeArray;
  texRes.res.array.array = d_idataArray;

  memset(&texDescr, 0, sizeof(texDescr));
  texDescr.normalizedCoords = true;
  texDescr.filterMode = cudaFilterModePoint;
  texDescr.addressMode[0] = cudaAddressModeWrap;
  texDescr.addressMode[1] = cudaAddressModeWrap;
  texDescr.readMode = cudaReadModeElementType;

  CHECK(cudaCreateTextureObject(&texRefArray, &texRes, &texDescr, NULL));

  CHECK(cudaMemset2D(d_odataPL, d_opitchBytesPL, 0, nx * sizeof(float), ny));
  shiftPitchLinear<<<dimGrid, dimBlock>>>(d_odataPL, (int)(d_opitchBytesPL / sizeof(float)), nx, ny, x_shift, y_shift,
                                          texRefPL);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  CHECK(cudaMemset2D(d_odataArr, d_opitchBytesArr, 0, nx * sizeof(float), ny));
  shiftArray<<<dimGrid, dimBlock>>>(d_odataArr, (int)(d_opitchBytesArr / sizeof(float)), nx, ny, x_shift, y_shift,
                                    texRefArray);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());

  CHECK(cudaMemcpy2D(h_odataPL, h_pitchBytes, d_odataPL, d_opitchBytesPL, nx * sizeof(float), ny,
                      cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy2D(h_odataArr, h_pitchBytes, d_odataArr, d_opitchBytesArr, nx * sizeof(float), ny,
                      cudaMemcpyDeviceToHost));

  // CPU reference check.
  reference_shift(gold, nx, ny, x_shift, y_shift);

  double max_abs_err_pl = 0.0, max_abs_err_arr = 0.0;
  for (size_t i = 0; i < numel; ++i) {
    double d1 = (double)h_odataPL[i] - (double)gold[i];
    double d2 = (double)h_odataArr[i] - (double)gold[i];
    if (d1 < 0) d1 = -d1;
    if (d2 < 0) d2 = -d2;
    if (d1 > max_abs_err_pl) max_abs_err_pl = d1;
    if (d2 > max_abs_err_arr) max_abs_err_arr = d2;
  }

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (size_t i = 0; i < numel; ++i) fprintf(f, "%.9g\n", h_odataPL[i]);
  fclose(f);

  printf("pitchLinearVsCudaArrayTexture done: w=%d, h=%d, "
         "max_abs_error(shiftPitchLinear vs ref)=%.3e, max_abs_error(shiftArray vs ref)=%.3e -> %s\n",
         nx, ny, max_abs_err_pl, max_abs_err_arr, out);
  printf("%s\n", (max_abs_err_pl == 0.0 && max_abs_err_arr == 0.0) ? "PASS" : "FAIL");

  CHECK(cudaDestroyTextureObject(texRefPL));
  CHECK(cudaDestroyTextureObject(texRefArray));
  cudaFree(d_idataPL);
  cudaFreeArray(d_idataArray);
  cudaFree(d_odataPL);
  cudaFree(d_odataArr);
  free(h_idata);
  free(h_odataPL);
  free(h_odataArr);
  free(gold);
  return 0;
}
