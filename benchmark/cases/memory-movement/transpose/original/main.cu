// transpose: out = in^T for a rows x cols matrix (row-major), rows=192 cols=128.
// Deterministic input (replicated by tests/verify.py):
//   in[idx] = ((idx % 17) - 8) * 0.25f   for idx in [0, rows*cols)
// out is cols x rows (row-major), written one value per line to argv[1].
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

__global__ void transpose(const float *in, float *out, int rows, int cols) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;  // column index
  int y = blockIdx.y * blockDim.y + threadIdx.y;  // row index
  if (x < cols && y < rows) out[x * rows + y] = in[y * cols + x];
}

static float genA(int i) { return ((i % 17) - 8) * 0.25f; }

int main(int argc, char **argv) {
  const int rows = 192, cols = 128;
  const int total = rows * cols;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  const size_t bytes = (size_t)total * sizeof(float);

  float *hin = (float *)malloc(bytes);
  float *hout = (float *)malloc(bytes);
  for (int i = 0; i < total; ++i) hin[i] = genA(i);

  float *din, *dout;
  CHECK(cudaMalloc(&din, bytes));
  CHECK(cudaMalloc(&dout, bytes));
  CHECK(cudaMemcpy(din, hin, bytes, cudaMemcpyHostToDevice));

  dim3 block(16, 16);
  dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
  transpose<<<grid, block>>>(din, dout, rows, cols);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(hout, dout, bytes, cudaMemcpyDeviceToHost));

  FILE *f = fopen(out, "w");
  if (!f) {
    fprintf(stderr, "cannot open %s for writing\n", out);
    return 2;
  }
  for (int i = 0; i < total; ++i) fprintf(f, "%.9g\n", hout[i]);
  fclose(f);
  printf("transpose done: %dx%d -> %s\n", rows, cols, out);

  cudaFree(din);
  cudaFree(dout);
  free(hin);
  free(hout);
  return 0;
}
