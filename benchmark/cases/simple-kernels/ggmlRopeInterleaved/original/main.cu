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

__global__ void rope_kernel(const float *x, float *y, int tokens, int dim) {
  int pair = blockIdx.x * blockDim.x + threadIdx.x;
  int pairs = tokens * (dim / 2);
  if (pair < pairs) {
    int t = pair / (dim / 2);
    int p = pair % (dim / 2);
    int i0 = t * dim + 2 * p;
    int i1 = i0 + 1;
    float theta = (float)((t * 17) % 2048) * powf(10000.0f, -2.0f * (float)p / (float)dim);
    float c = cosf(theta), s = sinf(theta);
    float a = x[i0], b = x[i1];
    y[i0] = a * c - b * s;
    y[i1] = a * s + b * c;
  }
}

int main(int argc, char **argv) {
  const int tokens = 512, dim = 128, n = tokens * dim;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  float *hx=(float*)malloc((size_t)n*sizeof(float)), *hy=(float*)malloc((size_t)n*sizeof(float));
  for (int i=0;i<n;++i) hx[i] = hs(i, 123);
  float *dx,*dy; CK(cudaMalloc(&dx,(size_t)n*sizeof(float))); CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));
  CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));
  int total = tokens * (dim / 2); int tpb=256, blocks=(total+tpb-1)/tpb;
  rope_kernel<<<blocks,tpb>>>(dx,dy,tokens,dim);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));
  write_vec(out, hy, n);
  cudaFree(dx); cudaFree(dy); free(hx); free(hy); return 0;
}
