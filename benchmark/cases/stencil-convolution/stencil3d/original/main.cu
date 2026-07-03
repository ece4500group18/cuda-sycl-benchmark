// stencil3d: 7-point stencil out = (c + 6 neighbors) / 7. Edge-clamped.
// in[idx] = h01(idx, 123). Grid nz=ny=nx=64 (row-major, z outermost).
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

__global__ void stencil3d(const float *in, float *out, int nz, int ny, int nx) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x < nx && y < ny && z < nz) {
    int idx = (z * ny + y) * nx + x;
    float c  = in[idx];
    float xm = in[(z * ny + y) * nx + cl(x-1,nx-1)];
    float xp = in[(z * ny + y) * nx + cl(x+1,nx-1)];
    float ym = in[(z * ny + cl(y-1,ny-1)) * nx + x];
    float yp = in[(z * ny + cl(y+1,ny-1)) * nx + x];
    float zm = in[(cl(z-1,nz-1) * ny + y) * nx + x];
    float zp = in[(cl(z+1,nz-1) * ny + y) * nx + x];
    out[idx] = (c + xm + xp + ym + yp + zm + zp) / 7.0f;
  }
}

int main(int argc, char **argv) {
  const int nz = 64, ny = 64, nx = 64, total = nz * ny * nx;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  size_t bytes = (size_t)total * sizeof(float);
  float *hin = (float *)malloc(bytes), *ho = (float *)malloc(bytes);
  for (int i = 0; i < total; ++i) hin[i] = h01(i, 123);
  float *din, *dout; CK(cudaMalloc(&din, bytes)); CK(cudaMalloc(&dout, bytes));
  CK(cudaMemcpy(din, hin, bytes, cudaMemcpyHostToDevice));
  dim3 block(8, 8, 8);
  dim3 grid((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);
  stencil3d<<<grid, block>>>(din, dout, nz, ny, nx);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(ho, dout, bytes, cudaMemcpyDeviceToHost));
  FILE *f = fopen(out, "w"); if (!f) { fprintf(stderr, "open %s\n", out); return 2; }
  for (int i = 0; i < total; ++i) fprintf(f, "%.9g\n", ho[i]); fclose(f);
  printf("stencil3d done: %dx%dx%d -> %s\n", nz, ny, nx, out);
  cudaFree(din); cudaFree(dout); free(hin); free(ho);
  return 0;
}
