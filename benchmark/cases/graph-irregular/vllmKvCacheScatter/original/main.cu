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

__global__ void kv_scatter(const float *src, float *cache, int tokens, int heads, int dim) {
  int idx=blockIdx.x*blockDim.x+threadIdx.x,n=tokens*heads*dim;
  if(idx<n){int d=idx%dim;int tmp=idx/dim;int h=tmp%heads;int t=tmp/heads;int page=(t*37)%tokens;cache[(h*tokens+page)*dim+d]=src[idx];}
}
int main(int argc,char**argv){const int tokens=512,heads=8,dim=64,n=tokens*heads*dim;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hsrc=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)calloc((size_t)n,sizeof(float));for(int i=0;i<n;++i)hsrc[i]=hs(i,123);float*ds,*dc;CK(cudaMalloc(&ds,(size_t)n*sizeof(float)));CK(cudaMalloc(&dc,(size_t)n*sizeof(float)));CK(cudaMemcpy(ds,hsrc,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));CK(cudaMemset(dc,0,(size_t)n*sizeof(float)));kv_scatter<<<(n+255)/256,256>>>(ds,dc,tokens,heads,dim);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dc,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(ds);cudaFree(dc);free(hsrc);free(hy);return 0;}
