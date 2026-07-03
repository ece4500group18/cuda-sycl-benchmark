// monteCarloPi: estimate pi by sampling the unit square with hashed points.
//   x_i = h01(i, 1), y_i = h01(i, 2); count hits where x^2 + y^2 < 1.
//   pi ~= 4 * hits / N.  N = 4194304 samples. Output: the single pi estimate.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}

__host__ __device__ static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

__global__ void mcKernel(int n, unsigned long long *count) {
  unsigned long long local = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    float x = h01(i, 1), y = h01(i, 2);
    if (x * x + y * y < 1.0f) local++;
  }
  atomicAdd(count, local);
}

int main(int argc, char **argv) {
  const int n = 4194304;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  unsigned long long *dcount, hcount = 0;
  CK(cudaMalloc(&dcount, sizeof(unsigned long long)));
  CK(cudaMemcpy(dcount, &hcount, sizeof(hcount), cudaMemcpyHostToDevice));
  mcKernel<<<256, 256>>>(n, dcount);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(&hcount, dcount, sizeof(hcount), cudaMemcpyDeviceToHost));
  double pi = 4.0 * (double)hcount / (double)n;

  FILE *f = fopen(out, "w"); if (!f) { fprintf(stderr, "open %s\n", out); return 2; }
  fprintf(f, "%.9g\n", pi); fclose(f);
  printf("monteCarloPi done: n=%d pi=%.6f -> %s\n", n, pi, out);
  cudaFree(dcount);
  return 0;
}
