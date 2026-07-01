#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA %s @%d\n", cudaGetErrorString(e), __LINE__); return 2; \
} } while (0)

__host__ __device__ static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

__host__ __device__ static inline float hs(unsigned i, unsigned s) {
  return 2.0f * h01(i, s) - 1.0f;
}

static void write_vec(const char *path, const float *data, int n) {
  FILE *f = fopen(path, "w");
  if (!f) { fprintf(stderr, "open %s\n", path); exit(2); }
  for (int i = 0; i < n; ++i) fprintf(f, "%.9g\n", data[i]);
  fclose(f);
}

__global__ void dequant_q8(const int8_t *q, const float *scale, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = (float)q[i] * scale[i / 32];
}

int main(int argc, char **argv) {
  const int blocks_q = 4096, qk = 32, n = blocks_q * qk;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  int8_t *hq=(int8_t*)malloc((size_t)n*sizeof(int8_t));
  float *hscl=(float*)malloc((size_t)blocks_q*sizeof(float)), *hy=(float*)malloc((size_t)n*sizeof(float));
  for (int b=0;b<blocks_q;++b) hscl[b] = 0.02f + 0.03f * h01(b, 77);
  for (int i=0;i<n;++i) hq[i] = (int8_t)lrintf(127.0f * hs(i, 123));
  int8_t *dq; float *ds,*dy; CK(cudaMalloc(&dq,(size_t)n)); CK(cudaMalloc(&ds,(size_t)blocks_q*sizeof(float))); CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));
  CK(cudaMemcpy(dq,hq,(size_t)n,cudaMemcpyHostToDevice)); CK(cudaMemcpy(ds,hscl,(size_t)blocks_q*sizeof(float),cudaMemcpyHostToDevice));
  int tpb=256, grid=(n+tpb-1)/tpb; dequant_q8<<<grid,tpb>>>(dq,ds,dy,n);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));
  write_vec(out, hy, n);
  cudaFree(dq); cudaFree(ds); cudaFree(dy); free(hq); free(hscl); free(hy); return 0;
}
