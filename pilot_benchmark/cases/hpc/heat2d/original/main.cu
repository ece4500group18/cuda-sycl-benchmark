// heat2d: explicit (FTCS) 2D heat equation. alpha = 0.2.
// Interior: u_new = u + alpha*((up+down+left+right) - 4*u). Boundary fixed.
// Grid ny=nx=128, K=50 steps. u0[idx] = h01(idx, 123).
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}

__host__ __device__ static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

__global__ void heat(const float *u, float *un, int ny, int nx, float alpha) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < nx && y < ny) {
    int idx = y * nx + x;
    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1) {
      un[idx] = u[idx];
    } else {
      float lap = (((u[(y-1)*nx+x] + u[(y+1)*nx+x]) + u[y*nx+x-1]) + u[y*nx+x+1])
                  - 4.0f * u[idx];
      un[idx] = u[idx] + alpha * lap;
    }
  }
}

int main(int argc, char **argv) {
  const int ny = 128, nx = 128, K = 50, total = ny * nx;
  const float alpha = 0.2f;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  size_t bytes = (size_t)total * sizeof(float);
  float *hu = (float *)malloc(bytes);
  for (int i = 0; i < total; ++i) hu[i] = h01(i, 123);
  float *du, *dun; CK(cudaMalloc(&du, bytes)); CK(cudaMalloc(&dun, bytes));
  CK(cudaMemcpy(du, hu, bytes, cudaMemcpyHostToDevice));
  dim3 block(16, 16), grid((nx + 15) / 16, (ny + 15) / 16);
  for (int it = 0; it < K; ++it) {
    heat<<<grid, block>>>(du, dun, ny, nx, alpha);
    CK(cudaGetLastError());
    float *tmp = du; du = dun; dun = tmp;
  }
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hu, du, bytes, cudaMemcpyDeviceToHost));
  FILE *f = fopen(out, "w"); if (!f) { fprintf(stderr, "open %s\n", out); return 2; }
  for (int i = 0; i < total; ++i) fprintf(f, "%.9g\n", hu[i]); fclose(f);
  printf("heat2d done: %dx%d K=%d -> %s\n", ny, nx, K, out);
  cudaFree(du); cudaFree(dun); free(hu);
  return 0;
}
