// scanBlock: inclusive prefix sum over a single block of n = 1024 hashed
// floats (Hillis-Steele).  in[i] = h01(i, 123)
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}

__host__ __device__ static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

__global__ void scanKernel(const float *in, float *out, int n) {
  extern __shared__ float t[];
  int tid = threadIdx.x;
  t[tid] = (tid < n) ? in[tid] : 0.0f;
  __syncthreads();
  for (int off = 1; off < n; off <<= 1) {
    float v = (tid >= off) ? t[tid - off] : 0.0f;
    __syncthreads();
    t[tid] += v;
    __syncthreads();
  }
  if (tid < n) out[tid] = t[tid];
}

int main(int argc, char **argv) {
  const int n = 1024;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  size_t bytes = (size_t)n * sizeof(float);
  float *hin = (float *)malloc(bytes), *ho = (float *)malloc(bytes);
  for (int i = 0; i < n; ++i) hin[i] = h01(i, 123);

  float *din, *dout; CK(cudaMalloc(&din, bytes)); CK(cudaMalloc(&dout, bytes));
  CK(cudaMemcpy(din, hin, bytes, cudaMemcpyHostToDevice));
  scanKernel<<<1, n, n * sizeof(float)>>>(din, dout, n);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(ho, dout, bytes, cudaMemcpyDeviceToHost));

  FILE *f = fopen(out, "w"); if (!f) { fprintf(stderr, "open %s\n", out); return 2; }
  for (int i = 0; i < n; ++i) fprintf(f, "%.9g\n", ho[i]); fclose(f);
  printf("scanBlock done: n=%d -> %s\n", n, out);
  cudaFree(din); cudaFree(dout); free(hin); free(ho);
  return 0;
}
