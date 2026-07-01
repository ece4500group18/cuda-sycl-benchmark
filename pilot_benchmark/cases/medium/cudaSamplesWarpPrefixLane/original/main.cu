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

__global__ void warp_scan(const float*x,float*y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float v=x[i];int lane=threadIdx.x&31;for(int off=1;off<32;off<<=1){float u=__shfl_up_sync(0xffffffff,v,off);if(lane>=off)v+=u;}y[i]=v;}}
int main(int argc,char**argv){const int n=32768;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i)hx[i]=0.01f*hs(i,123);float*dx,*dy;CK(cudaMalloc(&dx,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));warp_scan<<<(n+255)/256,256>>>(dx,dy,n);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
