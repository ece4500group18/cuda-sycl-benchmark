// dotProduct: sum(a*b) over n = 1048576 hashed floats via reduction.
// a[i] = h01(i, 123) ; b[i] = h01(i, 321)
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}

__host__ __device__ static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

__global__ void dotKernel(const float *a, const float *b, float *partial, int n) {
  extern __shared__ float s[];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  s[tid] = (i < n) ? a[i] * b[i] : 0.0f;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) s[tid] += s[tid + stride];
    __syncthreads();
  }
  if (tid == 0) partial[blockIdx.x] = s[0];
}

int main(int argc, char **argv) {
  const int n = 1048576;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  size_t bytes = (size_t)n * sizeof(float);
  float *ha = (float *)malloc(bytes), *hb = (float *)malloc(bytes);
  for (int i = 0; i < n; ++i) { ha[i] = h01(i, 123); hb[i] = h01(i, 321); }

  int tpb = 256, blocks = (n + tpb - 1) / tpb;
  float *da, *db, *dpart;
  CK(cudaMalloc(&da, bytes)); CK(cudaMalloc(&db, bytes));
  CK(cudaMalloc(&dpart, blocks * sizeof(float)));
  CK(cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice));
  dotKernel<<<blocks, tpb, tpb * sizeof(float)>>>(da, db, dpart, n);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  float *hpart = (float *)malloc(blocks * sizeof(float));
  CK(cudaMemcpy(hpart, dpart, blocks * sizeof(float), cudaMemcpyDeviceToHost));
  double total = 0.0;
  for (int b = 0; b < blocks; ++b) total += hpart[b];

  FILE *f = fopen(out, "w"); if (!f) { fprintf(stderr, "open %s\n", out); return 2; }
  fprintf(f, "%.9g\n", (float)total); fclose(f);
  printf("dotProduct done: n=%d dot=%.6f -> %s\n", n, total, out);
  cudaFree(da); cudaFree(db); cudaFree(dpart); free(ha); free(hb); free(hpart);
  return 0;
}
