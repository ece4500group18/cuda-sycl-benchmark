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

__global__ void gelu_tanh_kernel(const float *x, const float *g, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v=x[i]; float u=0.7978845608f*(v+0.044715f*v*v*v); y[i]=0.5f*v*(1.0f+tanhf(u));
  }
}

int main(int argc, char **argv) {
  const int n = 1048576;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  size_t bytes = (size_t)n * sizeof(float);
  float *hx=(float*)malloc(bytes), *hg=(float*)malloc(bytes), *hy=(float*)malloc(bytes);
  for (int i=0;i<n;++i) { hx[i] = 5.0f * hs(i, 123); hg[i] = 1.0f * hs(i, 321); }
  float *dx,*dg,*dy; CK(cudaMalloc(&dx,bytes)); CK(cudaMalloc(&dg,bytes)); CK(cudaMalloc(&dy,bytes));
  CK(cudaMemcpy(dx,hx,bytes,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dg,hg,bytes,cudaMemcpyHostToDevice));
  int tpb=256, blocks=(n+tpb-1)/tpb;
  gelu_tanh_kernel<<<blocks,tpb>>>(dx,dg,dy,n);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,bytes,cudaMemcpyDeviceToHost));
  write_vec(out, hy, n);
  cudaFree(dx); cudaFree(dg); cudaFree(dy); free(hx); free(hg); free(hy); return 0;
}
