// saxpy: y = alpha*x + y, alpha = 2.5f, n = 100000.
// Deterministic inputs (replicated by tests/verify.py):
//   x[i] = ((i % 17) - 8) * 0.25f
//   y[i] = ((i % 23) - 11) * 0.5f
// The updated y vector is written one value per line to argv[1].
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

__global__ void saxpy(float a, const float *x, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}

static float genA(int i) { return ((i % 17) - 8) * 0.25f; }
static float genB(int i) { return ((i % 23) - 11) * 0.5f; }

int main(int argc, char **argv) {
  const int n = 100000;
  const float alpha = 2.5f;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  const size_t bytes = (size_t)n * sizeof(float);

  float *hx = (float *)malloc(bytes);
  float *hy = (float *)malloc(bytes);
  for (int i = 0; i < n; ++i) {
    hx[i] = genA(i);
    hy[i] = genB(i);
  }

  float *dx, *dy;
  CHECK(cudaMalloc(&dx, bytes));
  CHECK(cudaMalloc(&dy, bytes));
  CHECK(cudaMemcpy(dx, hx, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dy, hy, bytes, cudaMemcpyHostToDevice));

  const int tpb = 256;
  const int blocks = (n + tpb - 1) / tpb;
  saxpy<<<blocks, tpb>>>(alpha, dx, dy, n);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(hy, dy, bytes, cudaMemcpyDeviceToHost));

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (int i = 0; i < n; ++i) fprintf(f, "%.9g\n", hy[i]);
  fclose(f);
  printf("saxpy done: n=%d alpha=%.1f -> %s\n", n, alpha, out);

  cudaFree(dx);
  cudaFree(dy);
  free(hx);
  free(hy);
  return 0;
}
