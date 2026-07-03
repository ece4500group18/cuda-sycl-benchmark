// bitonicSort: ascending bitonic sort of n = 1024 hashed floats in one block.
// in[i] = h01(i, 123)
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}

__host__ __device__ static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

__global__ void bitonic(float *data, int n) {
  extern __shared__ float s[];
  int t = threadIdx.x;
  s[t] = data[t];
  __syncthreads();
  for (int k = 2; k <= n; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      int ixj = t ^ j;
      if (ixj > t) {
        bool ascending = ((t & k) == 0);
        if ((ascending && s[t] > s[ixj]) || (!ascending && s[t] < s[ixj])) {
          float tmp = s[t]; s[t] = s[ixj]; s[ixj] = tmp;
        }
      }
      __syncthreads();
    }
  }
  data[t] = s[t];
}

int main(int argc, char **argv) {
  const int n = 1024;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  size_t bytes = (size_t)n * sizeof(float);
  float *h = (float *)malloc(bytes);
  for (int i = 0; i < n; ++i) h[i] = h01(i, 123);

  float *d; CK(cudaMalloc(&d, bytes));
  CK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice));
  bitonic<<<1, n, bytes>>>(d, n);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost));

  FILE *f = fopen(out, "w"); if (!f) { fprintf(stderr, "open %s\n", out); return 2; }
  for (int i = 0; i < n; ++i) fprintf(f, "%.9g\n", h[i]); fclose(f);
  printf("bitonicSort done: n=%d -> %s\n", n, out);
  cudaFree(d); free(h);
  return 0;
}
