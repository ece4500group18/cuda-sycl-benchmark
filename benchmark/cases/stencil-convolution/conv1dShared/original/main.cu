// conv1dShared: radius-3 fixed-weight 1D convolution with a shared-memory halo.
// in[i] = h01(i, 123). Boundary: replicate edge (clamp index to [0, n-1]).
// weights W = [1,2,3,4,3,2,1] / 16.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}
#define R 3

__host__ __device__ static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

__constant__ float W[2 * R + 1];

__device__ static inline int clampi(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

__global__ void conv1d(const float *in, float *out, int n) {
  extern __shared__ float s[];
  int bd = blockDim.x, t = threadIdx.x;
  int base = blockIdx.x * bd;
  int gid = base + t;
  s[t + R] = in[clampi(gid, 0, n - 1)];
  if (t < R) {
    s[t] = in[clampi(base - R + t, 0, n - 1)];
    s[t + R + bd] = in[clampi(base + bd + t, 0, n - 1)];
  }
  __syncthreads();
  if (gid < n) {
    float acc = 0.0f;
    for (int k = 0; k < 2 * R + 1; ++k) acc += W[k] * s[t + k];
    out[gid] = acc;
  }
}

int main(int argc, char **argv) {
  const int n = 100000;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  size_t bytes = (size_t)n * sizeof(float);
  float hW[2 * R + 1] = {1.f/16, 2.f/16, 3.f/16, 4.f/16, 3.f/16, 2.f/16, 1.f/16};
  float *hin = (float *)malloc(bytes), *ho = (float *)malloc(bytes);
  for (int i = 0; i < n; ++i) hin[i] = h01(i, 123);

  float *din, *dout; CK(cudaMalloc(&din, bytes)); CK(cudaMalloc(&dout, bytes));
  CK(cudaMemcpy(din, hin, bytes, cudaMemcpyHostToDevice));
  CK(cudaMemcpyToSymbol(W, hW, sizeof(hW)));
  int tpb = 256, blocks = (n + tpb - 1) / tpb;
  size_t shmem = (tpb + 2 * R) * sizeof(float);
  conv1d<<<blocks, tpb, shmem>>>(din, dout, n);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(ho, dout, bytes, cudaMemcpyDeviceToHost));

  FILE *f = fopen(out, "w"); if (!f) { fprintf(stderr, "open %s\n", out); return 2; }
  for (int i = 0; i < n; ++i) fprintf(f, "%.9g\n", ho[i]); fclose(f);
  printf("conv1dShared done: n=%d -> %s\n", n, out);
  cudaFree(din); cudaFree(dout); free(hin); free(ho);
  return 0;
}
