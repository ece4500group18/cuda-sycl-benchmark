// histogram: 256-bin histogram of n = 1048576 hashed indices using shared-mem
// atomics then a global atomic merge.  bin index = floor(h01(i,123) * 256).
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}
#define NBINS 256

__host__ __device__ static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

__global__ void histKernel(const int *idx, int n, int *bins) {
  __shared__ int sh[NBINS];
  int t = threadIdx.x;
  for (int b = t; b < NBINS; b += blockDim.x) sh[b] = 0;
  __syncthreads();
  for (int i = blockIdx.x * blockDim.x + t; i < n; i += gridDim.x * blockDim.x)
    atomicAdd(&sh[idx[i]], 1);
  __syncthreads();
  for (int b = t; b < NBINS; b += blockDim.x) atomicAdd(&bins[b], sh[b]);
}

int main(int argc, char **argv) {
  const int n = 1048576;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  int *hidx = (int *)malloc((size_t)n * sizeof(int));
  for (int i = 0; i < n; ++i) {
    int b = (int)(h01(i, 123) * NBINS);
    hidx[i] = b < 0 ? 0 : (b >= NBINS ? NBINS - 1 : b);
  }
  int *didx, *dbins;
  CK(cudaMalloc(&didx, (size_t)n * sizeof(int)));
  CK(cudaMalloc(&dbins, NBINS * sizeof(int)));
  CK(cudaMemcpy(didx, hidx, (size_t)n * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemset(dbins, 0, NBINS * sizeof(int)));
  int tpb = 256, blocks = 256;
  histKernel<<<blocks, tpb>>>(didx, n, dbins);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  int hbins[NBINS];
  CK(cudaMemcpy(hbins, dbins, NBINS * sizeof(int), cudaMemcpyDeviceToHost));

  FILE *f = fopen(out, "w"); if (!f) { fprintf(stderr, "open %s\n", out); return 2; }
  for (int b = 0; b < NBINS; ++b) fprintf(f, "%d\n", hbins[b]); fclose(f);
  printf("histogram done: n=%d bins=%d -> %s\n", n, NBINS, out);
  cudaFree(didx); cudaFree(dbins); free(hidx);
  return 0;
}
