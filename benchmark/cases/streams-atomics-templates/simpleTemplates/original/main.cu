// simpleTemplates: a templated CUDA kernel addConst<T> applied to floats.
// out = in + k, with k = 3.0f, n = 50000.
// Deterministic input (replicated by tests/verify.py):
//   in[i] = ((i % 17) - 8) * 0.25f
// The result is written one value per line to argv[1].
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

template <typename T>
__global__ void addConst(T *data, T k, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) data[i] = data[i] + k;
}

static float genA(int i) { return ((i % 17) - 8) * 0.25f; }

int main(int argc, char **argv) {
  const int n = 50000;
  const float k = 3.0f;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  const size_t bytes = (size_t)n * sizeof(float);

  float *h = (float *)malloc(bytes);
  for (int i = 0; i < n; ++i) h[i] = genA(i);

  float *d;
  CHECK(cudaMalloc(&d, bytes));
  CHECK(cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice));

  const int tpb = 256;
  const int blocks = (n + tpb - 1) / tpb;
  addConst<float><<<blocks, tpb>>>(d, k, n);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost));

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (int i = 0; i < n; ++i) fprintf(f, "%.9g\n", h[i]);
  fclose(f);
  printf("simpleTemplates done: n=%d k=%.1f -> %s\n", n, k, out);

  cudaFree(d);
  free(h);
  return 0;
}
