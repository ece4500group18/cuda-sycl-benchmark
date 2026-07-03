// matrixMul: naive square matrix multiply C = A * B, N = 128 (row-major).
// Deterministic inputs (replicated by tests/verify.py):
//   A[idx] = ((idx % 17) - 8) * 0.25f
//   B[idx] = ((idx % 23) - 11) * 0.5f   for idx in [0, N*N)
// C (row-major, N*N values, one per line) is written to argv[1].
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(call)                                                            \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__),   \
              __FILE__, __LINE__);                                             \
      return 2;                                                                \
    }                                                                          \
  } while (0)

__global__ void matmul(const float *A, const float *B, float *C, int N) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < N && col < N) {
    float s = 0.0f;
    for (int k = 0; k < N; ++k) s += A[row * N + k] * B[k * N + col];
    C[row * N + col] = s;
  }
}

static float genA(int i) { return ((i % 17) - 8) * 0.25f; }
static float genB(int i) { return ((i % 23) - 11) * 0.5f; }

int main(int argc, char **argv) {
  const int N = 128;
  const int total = N * N;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  const size_t bytes = (size_t)total * sizeof(float);

  float *hA = (float *)malloc(bytes);
  float *hB = (float *)malloc(bytes);
  float *hC = (float *)malloc(bytes);
  for (int i = 0; i < total; ++i) {
    hA[i] = genA(i);
    hB[i] = genB(i);
  }

  float *dA, *dB, *dC;
  CHECK(cudaMalloc(&dA, bytes));
  CHECK(cudaMalloc(&dB, bytes));
  CHECK(cudaMalloc(&dC, bytes));
  CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
  matmul<<<grid, block>>>(dA, dB, dC, N);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (int i = 0; i < total; ++i) fprintf(f, "%.9g\n", hC[i]);
  fclose(f);
  printf("matrixMul done: N=%d -> %s\n", N, out);

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  free(hA);
  free(hB);
  free(hC);
  return 0;
}
