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

__global__ void bias_relu(const float *x, const float *b, float *y, int rows, int cols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int n = rows * cols;
  if (idx < n) {
    float v = x[idx] + b[idx % cols];
    y[idx] = v > 0.0f ? v : 0.0f;
  }
}

int main(int argc, char **argv) {
  const int rows = 512, cols = 256, n = rows * cols;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  float *hx=(float*)malloc((size_t)n*sizeof(float)), *hb=(float*)malloc((size_t)cols*sizeof(float)), *hy=(float*)malloc((size_t)n*sizeof(float));
  for (int i=0;i<n;++i) hx[i] = hs(i, 123);
  for (int i=0;i<cols;++i) hb[i] = 0.25f * hs(i, 321);
  float *dx,*db,*dy; CK(cudaMalloc(&dx,(size_t)n*sizeof(float))); CK(cudaMalloc(&db,(size_t)cols*sizeof(float))); CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));
  CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(db,hb,(size_t)cols*sizeof(float),cudaMemcpyHostToDevice));
  int tpb=256, blocks=(n+tpb-1)/tpb;
  bias_relu<<<blocks,tpb>>>(dx,db,dy,rows,cols);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));
  write_vec(out, hy, n);
  cudaFree(dx); cudaFree(db); cudaFree(dy); free(hx); free(hb); free(hy); return 0;
}
