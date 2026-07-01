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

__global__ void concat_rows(const float *a, const float *b, float *y, int rows, int ca, int cb) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * (ca + cb);
  if (idx < total) {
    int row = idx / (ca + cb);
    int col = idx % (ca + cb);
    y[idx] = (col < ca) ? a[row * ca + col] : b[row * cb + (col - ca)];
  }
}

int main(int argc, char **argv) {
  const int rows=512, ca=192, cb=64, na=rows*ca, nb=rows*cb, n=rows*(ca+cb);
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  float *ha=(float*)malloc((size_t)na*sizeof(float)), *hb=(float*)malloc((size_t)nb*sizeof(float)), *hy=(float*)malloc((size_t)n*sizeof(float));
  for (int i=0;i<na;++i) ha[i] = hs(i, 123);
  for (int i=0;i<nb;++i) hb[i] = 2.0f * hs(i, 321);
  float *da,*db,*dy; CK(cudaMalloc(&da,(size_t)na*sizeof(float))); CK(cudaMalloc(&db,(size_t)nb*sizeof(float))); CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));
  CK(cudaMemcpy(da,ha,(size_t)na*sizeof(float),cudaMemcpyHostToDevice)); CK(cudaMemcpy(db,hb,(size_t)nb*sizeof(float),cudaMemcpyHostToDevice));
  int tpb=256, grid=(n+tpb-1)/tpb; concat_rows<<<grid,tpb>>>(da,db,dy,rows,ca,cb);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));
  write_vec(out, hy, n);
  cudaFree(da); cudaFree(db); cudaFree(dy); free(ha); free(hb); free(hy); return 0;
}
