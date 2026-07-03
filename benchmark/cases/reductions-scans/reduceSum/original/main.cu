// reduceSum: parallel sum reduction of n = 1048576 hashed floats.
// Block reduction in shared memory; host sums the per-block partials.
// in[i] = h01(i, 123)   (see h01 below / verify_lib gen_hash01)
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}

__host__ __device__ static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

__global__ void reduceSumKernel(const float *in, float *partial, int n) {
  extern __shared__ float s[];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  s[tid] = (i < n) ? in[i] : 0.0f;
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
  float *hin = (float *)malloc(bytes);
  for (int i = 0; i < n; ++i) hin[i] = h01(i, 123);

  int tpb = 256, blocks = (n + tpb - 1) / tpb;
  float *din, *dpart; CK(cudaMalloc(&din, bytes)); CK(cudaMalloc(&dpart, blocks * sizeof(float)));
  CK(cudaMemcpy(din, hin, bytes, cudaMemcpyHostToDevice));
  reduceSumKernel<<<blocks, tpb, tpb * sizeof(float)>>>(din, dpart, n);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  float *hpart = (float *)malloc(blocks * sizeof(float));
  CK(cudaMemcpy(hpart, dpart, blocks * sizeof(float), cudaMemcpyDeviceToHost));
  double total = 0.0;
  for (int b = 0; b < blocks; ++b) total += hpart[b];

  FILE *f = fopen(out, "w"); if (!f) { fprintf(stderr, "open %s\n", out); return 2; }
  fprintf(f, "%.9g\n", (float)total); fclose(f);
  printf("reduceSum done: n=%d sum=%.6f -> %s\n", n, total, out);
  cudaFree(din); cudaFree(dpart); free(hin); free(hpart);
  return 0;
}
