// cufftC2C: forward 1D complex-to-complex FFT of n=4096 points with cuFFT.
//   input real[i]=h01(i,123), imag=0. Output: magnitude spectrum |X[k]|.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cufft.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}

__host__ __device__ static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

int main(int argc, char **argv) {
  const int n = 4096;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  cufftComplex *h = (cufftComplex *)malloc((size_t)n * sizeof(cufftComplex));
  for (int i = 0; i < n; ++i) { h[i].x = h01(i, 123); h[i].y = 0.0f; }
  cufftComplex *d; CK(cudaMalloc(&d, (size_t)n * sizeof(cufftComplex)));
  CK(cudaMemcpy(d, h, (size_t)n * sizeof(cufftComplex), cudaMemcpyHostToDevice));
  cufftHandle plan;
  if (cufftPlan1d(&plan, n, CUFFT_C2C, 1) != CUFFT_SUCCESS) { fprintf(stderr,"cufftPlan1d failed\n"); return 2; }
  cufftExecC2C(plan, d, d, CUFFT_FORWARD);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h, d, (size_t)n * sizeof(cufftComplex), cudaMemcpyDeviceToHost));
  cufftDestroy(plan);
  FILE *f = fopen(out, "w"); if (!f) { fprintf(stderr, "open %s\n", out); return 2; }
  for (int i = 0; i < n; ++i) fprintf(f, "%.9g\n", sqrtf(h[i].x * h[i].x + h[i].y * h[i].y)); fclose(f);
  printf("cufftC2C done: n=%d -> %s\n", n, out);
  cudaFree(d); free(h);
  return 0;
}
