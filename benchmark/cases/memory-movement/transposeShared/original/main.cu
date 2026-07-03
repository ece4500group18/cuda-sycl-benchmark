// transposeShared: tiled 256x256 transpose using shared memory.
// in[idx] = h01(idx, 123) ; out is cols x rows (row-major).
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

__global__ void transposeShared(const float *in, float *out, int rows, int cols) {
  __shared__ float tile[TILE][TILE + 1];
  int x = blockIdx.x * TILE + threadIdx.x;  // column
  int y = blockIdx.y * TILE + threadIdx.y;  // row
  if (x < cols && y < rows) tile[threadIdx.y][threadIdx.x] = in[y * cols + x];
  __syncthreads();
  int tx = blockIdx.y * TILE + threadIdx.x;  // row index in source -> col in out
  int ty = blockIdx.x * TILE + threadIdx.y;  // col index in source -> row in out
  if (tx < rows && ty < cols) out[ty * rows + tx] = tile[threadIdx.x][threadIdx.y];
}

int main(int argc, char **argv) {
  const int rows = 256, cols = 256, total = rows * cols;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  size_t bytes = (size_t)total * sizeof(float);
  float *hin = (float *)malloc(bytes), *ho = (float *)malloc(bytes);
  for (int i = 0; i < total; ++i) hin[i] = h01(i, 123);

  float *din, *dout; CK(cudaMalloc(&din, bytes)); CK(cudaMalloc(&dout, bytes));
  CK(cudaMemcpy(din, hin, bytes, cudaMemcpyHostToDevice));
  dim3 block(TILE, TILE);
  dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);
  transposeShared<<<grid, block>>>(din, dout, rows, cols);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(ho, dout, bytes, cudaMemcpyDeviceToHost));

  FILE *f = fopen(out, "w"); if (!f) { fprintf(stderr, "open %s\n", out); return 2; }
  for (int i = 0; i < total; ++i) fprintf(f, "%.9g\n", ho[i]); fclose(f);
  printf("transposeShared done: %dx%d -> %s\n", rows, cols, out);
  cudaFree(din); cudaFree(dout); free(hin); free(ho);
  return 0;
}
