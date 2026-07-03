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

__global__ void adam_update(const float *p, const float *m, const float *v, const float *g, float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float m2 = 0.9f * m[i] + 0.1f * g[i];
    float v2 = 0.999f * v[i] + 0.001f * g[i] * g[i];
    out[i] = p[i] - 0.001f * m2 / (sqrtf(v2) + 1.0e-8f);
  }
}

int main(int argc, char **argv) {
  const int n = 262144;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  size_t bytes = (size_t)n * sizeof(float);
  float *hp=(float*)malloc(bytes), *hm=(float*)malloc(bytes), *hv=(float*)malloc(bytes), *hg=(float*)malloc(bytes), *hy=(float*)malloc(bytes);
  for (int i=0;i<n;++i) { hp[i]=hs(i,11); hm[i]=0.1f*hs(i,22); hv[i]=0.01f+0.01f*h01(i,33); hg[i]=hs(i,44); }
  float *dp,*dm,*dv,*dg,*dy; CK(cudaMalloc(&dp,bytes)); CK(cudaMalloc(&dm,bytes)); CK(cudaMalloc(&dv,bytes)); CK(cudaMalloc(&dg,bytes)); CK(cudaMalloc(&dy,bytes));
  CK(cudaMemcpy(dp,hp,bytes,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dm,hm,bytes,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dv,hv,bytes,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dg,hg,bytes,cudaMemcpyHostToDevice));
  int tpb=256, blocks=(n+tpb-1)/tpb;
  adam_update<<<blocks,tpb>>>(dp,dm,dv,dg,dy,n);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,bytes,cudaMemcpyDeviceToHost));
  write_vec(out, hy, n);
  cudaFree(dp); cudaFree(dm); cudaFree(dv); cudaFree(dg); cudaFree(dy); free(hp); free(hm); free(hv); free(hg); free(hy); return 0;
}
