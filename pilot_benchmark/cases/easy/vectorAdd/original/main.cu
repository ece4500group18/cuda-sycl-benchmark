// vectorAdd: element-wise C = A + B.
// Canonical deterministic inputs (replicated by tests/verify.py):
//   A[i] = ((i % 17) - 8) * 0.25f
//   B[i] = ((i % 23) - 11) * 0.5f
// The result vector C is written, one value per line, to argv[1]
// (default: output/output.txt) so the host-side verifier can check it.
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

__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

static float genA(int i) { return ((i % 17) - 8) * 0.25f; }
static float genB(int i) { return ((i % 23) - 11) * 0.5f; }

int main(int argc, char **argv) {
  const int n = 100000;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  const size_t bytes = (size_t)n * sizeof(float);

  float *ha = (float *)malloc(bytes);
  float *hb = (float *)malloc(bytes);
  float *hc = (float *)malloc(bytes);
  for (int i = 0; i < n; ++i) {
    ha[i] = genA(i);
    hb[i] = genB(i);
  }

  float *da, *db, *dc;
  CHECK(cudaMalloc(&da, bytes));
  CHECK(cudaMalloc(&db, bytes));
  CHECK(cudaMalloc(&dc, bytes));
  CHECK(cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice));

  const int tpb = 256;
  const int blocks = (n + tpb - 1) / tpb;
  vectorAdd<<<blocks, tpb>>>(da, db, dc, n);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(hc, dc, bytes, cudaMemcpyDeviceToHost));

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (int i = 0; i < n; ++i) fprintf(f, "%.9g\n", hc[i]);
  fclose(f);
  printf("vectorAdd done: n=%d -> %s\n", n, out);

  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);
  free(ha);
  free(hb);
  free(hc);
  return 0;
}
