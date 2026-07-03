// finiteDiff1d: central difference out[i] = 0.5*(in[i+1] - in[i-1]) (h=1).
// Edge-clamped neighbors. in[i] = h01(i, 123), n = 100000.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}

__host__ __device__ static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}
__device__ static inline int cl(int v, int hi){ return v<0?0:(v>hi?hi:v); }

__global__ void fd1d(const float *in, float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = 0.5f * (in[cl(i+1,n-1)] - in[cl(i-1,n-1)]);
}

int main(int argc, char **argv) {
  const int n = 100000;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  size_t bytes = (size_t)n * sizeof(float);
  float *hin = (float *)malloc(bytes), *ho = (float *)malloc(bytes);
  for (int i = 0; i < n; ++i) hin[i] = h01(i, 123);
  float *din, *dout; CK(cudaMalloc(&din, bytes)); CK(cudaMalloc(&dout, bytes));
  CK(cudaMemcpy(din, hin, bytes, cudaMemcpyHostToDevice));
  int tpb = 256, blocks = (n + tpb - 1) / tpb;
  fd1d<<<blocks, tpb>>>(din, dout, n);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(ho, dout, bytes, cudaMemcpyDeviceToHost));
  FILE *f = fopen(out, "w"); if (!f) { fprintf(stderr, "open %s\n", out); return 2; }
  for (int i = 0; i < n; ++i) fprintf(f, "%.9g\n", ho[i]); fclose(f);
  printf("finiteDiff1d done: n=%d -> %s\n", n, out);
  cudaFree(din); cudaFree(dout); free(hin); free(ho);
  return 0;
}
