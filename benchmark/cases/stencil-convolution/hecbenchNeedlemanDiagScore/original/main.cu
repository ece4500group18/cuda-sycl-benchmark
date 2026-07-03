#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA %s @%d\n", cudaGetErrorString(e), __LINE__); return 2; } } while (0)

__host__ __device__ static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}
__host__ __device__ static inline float hs(unsigned i, unsigned s) {
  return 2.0f * h01(i, s) - 1.0f;
}
static void write_vec(const char *path, const float *data, int n) {
  FILE *f = fopen(path, "w");
  if (!f) { fprintf(stderr, "open %s\n", path); exit(2); }
  for (int i = 0; i < n; ++i) fprintf(f, "%.9g\n", data[i]);
  fclose(f);
}

__global__ void nw_diag(float*y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){int a=(i*7+3)&15,b=(i*11+5)&15,up=(i*13)&63,left=(i*17)&63,diag=(i*19)&63;int match=(a==b)?2:-1;int best=max(diag+match,max(up-1,left-1));y[i]=(float)best;}}
int main(int argc,char**argv){const int n=4096;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hy=(float*)malloc((size_t)n*sizeof(float));float*dy;CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));nw_diag<<<(n+255)/256,256>>>(dy,n);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(dy);free(hy);return 0;}
