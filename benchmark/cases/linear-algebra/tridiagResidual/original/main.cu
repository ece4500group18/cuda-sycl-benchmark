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

__global__ void residual(const float *x, const float *b, float *r, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float xm = x[i > 0 ? i - 1 : i];
    float xp = x[i + 1 < n ? i + 1 : i];
    r[i] = b[i] - (2.0f * x[i] - xm - xp);
  }
}

int main(int argc, char **argv) {
  const int n = 262144;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  size_t bytes = (size_t)n * sizeof(float);
  float *hx=(float*)malloc(bytes), *hb=(float*)malloc(bytes), *hr=(float*)malloc(bytes);
  for (int i=0;i<n;++i) { hx[i]=hs(i,123); hb[i]=0.5f*hs(i,456); }
  float *dx,*db,*dr; CK(cudaMalloc(&dx,bytes)); CK(cudaMalloc(&db,bytes)); CK(cudaMalloc(&dr,bytes));
  CK(cudaMemcpy(dx,hx,bytes,cudaMemcpyHostToDevice)); CK(cudaMemcpy(db,hb,bytes,cudaMemcpyHostToDevice));
  int tpb=256, blocks=(n+tpb-1)/tpb;
  residual<<<blocks,tpb>>>(dx,db,dr,n);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hr,dr,bytes,cudaMemcpyDeviceToHost));
  write_vec(out, hr, n);
  cudaFree(dx); cudaFree(db); cudaFree(dr); free(hx); free(hb); free(hr); return 0;
}
