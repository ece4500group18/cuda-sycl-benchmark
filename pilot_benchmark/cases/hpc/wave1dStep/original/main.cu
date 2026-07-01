#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}

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

__global__ void wave_step(const float *prev, const float *cur, float *next, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float left = cur[i > 0 ? i - 1 : i];
    float right = cur[i + 1 < n ? i + 1 : i];
    next[i] = 2.0f * cur[i] - prev[i] + 0.1f * (left - 2.0f * cur[i] + right);
  }
}

int main(int argc, char **argv) {
  const int n = 262144;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  size_t bytes = (size_t)n * sizeof(float);
  float *hp=(float*)malloc(bytes), *hc=(float*)malloc(bytes), *hn=(float*)malloc(bytes);
  for (int i=0;i<n;++i) { hp[i]=hs(i,13); hc[i]=hs(i,14); }
  float *dp,*dc,*dn; CK(cudaMalloc(&dp,bytes)); CK(cudaMalloc(&dc,bytes)); CK(cudaMalloc(&dn,bytes));
  CK(cudaMemcpy(dp,hp,bytes,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dc,hc,bytes,cudaMemcpyHostToDevice));
  int tpb=256, blocks=(n+tpb-1)/tpb;
  wave_step<<<blocks,tpb>>>(dp,dc,dn,n);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hn,dn,bytes,cudaMemcpyDeviceToHost));
  write_vec(out, hn, n);
  cudaFree(dp); cudaFree(dc); cudaFree(dn); free(hp); free(hc); free(hn); return 0;
}
