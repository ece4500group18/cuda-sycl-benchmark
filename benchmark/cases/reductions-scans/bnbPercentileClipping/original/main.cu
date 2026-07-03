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

__global__ void clip_blockwise(const float *g, float *y, int blocks_n) {
  __shared__ float s[64];
  int b=blockIdx.x, t=threadIdx.x;
  float v=g[b*64+t];
  s[t]=fabsf(v); __syncthreads();
  for (int stride=32; stride>0; stride>>=1) { if(t<stride) s[t]=fmaxf(s[t],s[t+stride]); __syncthreads(); }
  float thr=0.7f*s[0];
  y[b*64+t]=fminf(thr,fmaxf(-thr,v));
}

int main(int argc, char **argv) {
  const int blocks_n=4096, block=64, n=blocks_n*block; const char *out=(argc>1)?argv[1]:"output/output.txt";
  float *hg=(float*)malloc((size_t)n*sizeof(float)), *hy=(float*)malloc((size_t)n*sizeof(float));
  for(int i=0;i<n;++i) hg[i]=8.0f*hs(i,123);
  float *dg,*dy; CK(cudaMalloc(&dg,(size_t)n*sizeof(float))); CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));
  CK(cudaMemcpy(dg,hg,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));
  clip_blockwise<<<blocks_n,64>>>(dg,dy,blocks_n);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(out,hy,n);
  cudaFree(dg); cudaFree(dy); free(hg); free(hy); return 0;
}
