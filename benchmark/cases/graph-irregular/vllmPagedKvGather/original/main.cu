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

__global__ void kv_gather(const float *cache, float *out, int tokens, int heads, int dim) {
  int idx=blockIdx.x*blockDim.x+threadIdx.x, n=tokens*heads*dim;
  if(idx<n){ int d=idx%dim; int tmp=idx/dim; int h=tmp%heads; int t=tmp/heads; int page=(t*37)%tokens; out[idx]=cache[(h*tokens+page)*dim+d]; }
}
int main(int argc,char**argv){ const int tokens=512,heads=8,dim=64,n=tokens*heads*dim; const char*outp=(argc>1)?argv[1]:"output/output.txt";
float *hc=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float)); for(int i=0;i<n;++i) hc[i]=hs(i,123);
float *dc,*dy; CK(cudaMalloc(&dc,(size_t)n*sizeof(float))); CK(cudaMalloc(&dy,(size_t)n*sizeof(float))); CK(cudaMemcpy(dc,hc,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));
int tpb=256,grid=(n+tpb-1)/tpb; kv_gather<<<grid,tpb>>>(dc,dy,tokens,heads,dim); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(outp,hy,n); cudaFree(dc); cudaFree(dy); free(hc); free(hy); return 0; }
