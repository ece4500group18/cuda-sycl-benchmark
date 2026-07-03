// tiledMatmul: shared-memory 16x16 tiled GEMM C = A*B, N = 256.
// A[idx] = h01(idx, 123) ; B[idx] = h01(idx, 321)  (row-major)
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}
#define TILE 16

__host__ __device__ static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

__global__ void tiledMM(const float *A, const float *B, float *C, int N) {
  __shared__ float As[TILE][TILE], Bs[TILE][TILE];
  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;
  float acc = 0.0f;
  for (int m = 0; m < N / TILE; ++m) {
    As[threadIdx.y][threadIdx.x] = A[row * N + m * TILE + threadIdx.x];
    Bs[threadIdx.y][threadIdx.x] = B[(m * TILE + threadIdx.y) * N + col];
    __syncthreads();
    for (int k = 0; k < TILE; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    __syncthreads();
  }
  C[row * N + col] = acc;
}

int main(int argc, char **argv) {
  const int N = 256, total = N * N;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  size_t bytes = (size_t)total * sizeof(float);
  float *hA = (float *)malloc(bytes), *hB = (float *)malloc(bytes), *hC = (float *)malloc(bytes);
  for (int i = 0; i < total; ++i) { hA[i] = h01(i, 123); hB[i] = h01(i, 321); }

  float *dA, *dB, *dC;
  CK(cudaMalloc(&dA, bytes)); CK(cudaMalloc(&dB, bytes)); CK(cudaMalloc(&dC, bytes));
  CK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));
  dim3 block(TILE, TILE), grid(N / TILE, N / TILE);
  tiledMM<<<grid, block>>>(dA, dB, dC, N);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));

  FILE *f = fopen(out, "w"); if (!f) { fprintf(stderr, "open %s\n", out); return 2; }
  for (int i = 0; i < total; ++i) fprintf(f, "%.9g\n", hC[i]); fclose(f);
  printf("tiledMatmul done: N=%d -> %s\n", N, out);
  cudaFree(dA); cudaFree(dB); cudaFree(dC); free(hA); free(hB); free(hC);
  return 0;
}
