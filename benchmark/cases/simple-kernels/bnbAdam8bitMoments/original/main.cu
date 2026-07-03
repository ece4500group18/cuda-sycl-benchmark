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

__global__ void adam8bit(const float *p, const float *g, const int8_t *mq, const int8_t *vq, float *out, int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<n) {
    float m=(float)mq[i]*0.002f;
    float v=fabsf((float)vq[i])*0.0002f + 0.001f;
    float m2=0.9f*m + 0.1f*g[i];
    float v2=0.999f*v + 0.001f*g[i]*g[i];
    out[i]=p[i] - 0.001f*m2/(sqrtf(v2)+1.0e-8f);
  }
}

int main(int argc, char **argv) {
  const int n=262144; const char *out=(argc>1)?argv[1]:"output/output.txt";
  float *hp=(float*)malloc((size_t)n*sizeof(float)), *hg=(float*)malloc((size_t)n*sizeof(float)), *hy=(float*)malloc((size_t)n*sizeof(float));
  int8_t *hm=(int8_t*)malloc((size_t)n), *hv=(int8_t*)malloc((size_t)n);
  for (int i=0;i<n;++i) { hp[i]=hs(i,11); hg[i]=hs(i,22); hm[i]=(int8_t)lrintf(60.0f*hs(i,33)); hv[i]=(int8_t)lrintf(60.0f*hs(i,44)); }
  float *dp,*dg,*dy; int8_t *dm,*dv; CK(cudaMalloc(&dp,(size_t)n*sizeof(float))); CK(cudaMalloc(&dg,(size_t)n*sizeof(float))); CK(cudaMalloc(&dm,(size_t)n)); CK(cudaMalloc(&dv,(size_t)n)); CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));
  CK(cudaMemcpy(dp,hp,(size_t)n*sizeof(float),cudaMemcpyHostToDevice)); CK(cudaMemcpy(dg,hg,(size_t)n*sizeof(float),cudaMemcpyHostToDevice)); CK(cudaMemcpy(dm,hm,(size_t)n,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dv,hv,(size_t)n,cudaMemcpyHostToDevice));
  int tpb=256, grid=(n+tpb-1)/tpb; adam8bit<<<grid,tpb>>>(dp,dg,dm,dv,dy,n);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(out,hy,n);
  cudaFree(dp); cudaFree(dg); cudaFree(dm); cudaFree(dv); cudaFree(dy); free(hp); free(hg); free(hm); free(hv); free(hy); return 0;
}
