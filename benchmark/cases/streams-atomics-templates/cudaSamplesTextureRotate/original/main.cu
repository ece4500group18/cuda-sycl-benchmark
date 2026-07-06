// Image rotation through a CUDA texture object: normalized coordinates,
// linear filtering and wrap addressing.
//
// Extracted from NVIDIA/cuda-samples 0_Introduction/simpleTexture
// (simpleTexture.cu). Upstream: @ b7c5481c (BSD-3-Clause, NVIDIA).
// The transformKernel and the texture-object setup sequence
// (cudaChannelFormatDesc -> cudaMallocArray -> resource/texture descriptors
// -> cudaCreateTextureObject) are upstream code verbatim. The harness
// replaces the PGM file input with a deterministic hash image and dumps the
// rotated result. Linear filtering quantizes interpolation weights, so the
// verifier uses a small absolute tolerance.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}

// ---- upstream kernel (verbatim) ----------------------------------------------
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
// ---- end upstream kernel -------------------------------------------------------

int main(int argc, char **argv) {
  const unsigned int width = 128, height = 128;
  const float angle = 0.5f;  // upstream's rotation angle (radians)
  const char *out_path = (argc > 1) ? argv[1] : "output/output.txt";
  const unsigned int size = width * height * sizeof(float);

  // Deterministic smooth image replaces the PGM input (smoothness keeps the
  // linear-filter weight quantization error small and verifiable).
  float *hData = (float*)malloc(size);
  for (unsigned int y = 0; y < height; ++y)
    for (unsigned int x = 0; x < width; ++x)
      hData[y * width + x] = 0.5f + 0.25f * sinf(2.0f * (float)M_PI * x / width)
                                  + 0.25f * cosf(2.0f * (float)M_PI * y / height);

  float *dData = NULL;
  CK(cudaMalloc((void**)&dData, size));

  // ---- upstream texture-object setup (verbatim sequence) ----------------------
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  cudaArray *cuArray;
  CK(cudaMallocArray(&cuArray, &channelDesc, width, height));
  CK(cudaMemcpy2DToArray(cuArray, 0, 0, hData, width * sizeof(float),
                         width * sizeof(float), height, cudaMemcpyHostToDevice));

  cudaTextureObject_t tex;
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

  CK(cudaCreateTextureObject(&tex, &texRes, &texDescr, NULL));
  // ------------------------------------------------------------------------------

  dim3 dimBlock(8, 8, 1);
  dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

  transformKernel<<<dimGrid, dimBlock>>>(dData, width, height, angle, tex);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

  float *hOut = (float*)malloc(size);
  CK(cudaMemcpy(hOut, dData, size, cudaMemcpyDeviceToHost));

  FILE *f = fopen(out_path, "w");
  if (!f) { fprintf(stderr, "open %s\n", out_path); return 2; }
  for (unsigned int i = 0; i < width * height; ++i) fprintf(f, "%.9g\n", hOut[i]);
  fclose(f);

  CK(cudaDestroyTextureObject(tex));
  CK(cudaFreeArray(cuArray));
  cudaFree(dData);
  free(hData); free(hOut);
  return 0;
}
