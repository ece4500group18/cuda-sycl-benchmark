// stencil2d: 5-point stencil out = 0.2*(c + up + down + left + right).
// Edge-clamped. in[idx] = h01(idx, 123). Grid ny=nx=256 (row-major).
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}

__host__ __device__ static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}
__device__ static inline int cl(int v, int hi){ return v<0?0:(v>hi?hi:v); }

__global__ void stencil2d(const float *in, float *out, int ny, int nx) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < nx && y < ny) {
    float c  = in[y * nx + x];
    float up = in[cl(y-1,ny-1) * nx + x];
    float dn = in[cl(y+1,ny-1) * nx + x];
    float lf = in[y * nx + cl(x-1,nx-1)];
    float rt = in[y * nx + cl(x+1,nx-1)];
    out[y * nx + x] = 0.2f * (c + up + dn + lf + rt);
  }
}

int main(int argc, char **argv) {
  const int ny = 256, nx = 256, total = ny * nx;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  size_t bytes = (size_t)total * sizeof(float);
  float *hin = (float *)malloc(bytes), *ho = (float *)malloc(bytes);
  for (int i = 0; i < total; ++i) hin[i] = h01(i, 123);
  float *din, *dout; CK(cudaMalloc(&din, bytes)); CK(cudaMalloc(&dout, bytes));
  CK(cudaMemcpy(din, hin, bytes, cudaMemcpyHostToDevice));
  dim3 block(16, 16), grid((nx + 15) / 16, (ny + 15) / 16);
  stencil2d<<<grid, block>>>(din, dout, ny, nx);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(ho, dout, bytes, cudaMemcpyDeviceToHost));
  FILE *f = fopen(out, "w"); if (!f) { fprintf(stderr, "open %s\n", out); return 2; }
  for (int i = 0; i < total; ++i) fprintf(f, "%.9g\n", ho[i]); fclose(f);
  printf("stencil2d done: %dx%d -> %s\n", ny, nx, out);
  cudaFree(din); cudaFree(dout); free(hin); free(ho);
  return 0;
}
