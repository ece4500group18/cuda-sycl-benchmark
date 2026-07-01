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

__global__ void fw(float*y,int n,int k){int idx=blockIdx.x*blockDim.x+threadIdx.x,total=n*n;if(idx<total){int i=idx/n,j=idx%n;int cur=(i*13+j*7)&1023;int via=((i*13+k*7)&1023)+((k*13+j*7)&1023);y[idx]=(float)min(cur,via);}}
int main(int argc,char**argv){const int n=256,k=64,total=n*n;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hy=(float*)malloc((size_t)total*sizeof(float));float*dy;CK(cudaMalloc(&dy,(size_t)total*sizeof(float)));fw<<<(total+255)/256,256>>>(dy,n,k);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)total*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,total);cudaFree(dy);free(hy);return 0;}
