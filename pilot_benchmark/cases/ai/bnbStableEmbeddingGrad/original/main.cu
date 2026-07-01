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

__global__ void embed_grad(const int *idx, const float *grad, float *table, int tokens, int dim, int vocab) {
  int p=blockIdx.x*blockDim.x+threadIdx.x, n=tokens*dim;
  if(p<n){ int t=p/dim, d=p%dim; atomicAdd(&table[idx[t]*dim+d], grad[p]); }
}
int main(int argc,char**argv){ const int tokens=4096,dim=64,vocab=1024,n=tokens*dim,total=vocab*dim; const char*out=(argc>1)?argv[1]:"output/output.txt";
int *hi=(int*)malloc((size_t)tokens*sizeof(int)); float *hg=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)calloc((size_t)total,sizeof(float));
for(int t=0;t<tokens;++t) hi[t]=(int)floorf(h01(t,77)*vocab); for(int i=0;i<n;++i) hg[i]=0.01f*hs(i,123);
int*di; float*dg,*dt; CK(cudaMalloc(&di,(size_t)tokens*sizeof(int))); CK(cudaMalloc(&dg,(size_t)n*sizeof(float))); CK(cudaMalloc(&dt,(size_t)total*sizeof(float)));
CK(cudaMemcpy(di,hi,(size_t)tokens*sizeof(int),cudaMemcpyHostToDevice)); CK(cudaMemcpy(dg,hg,(size_t)n*sizeof(float),cudaMemcpyHostToDevice)); CK(cudaMemset(dt,0,(size_t)total*sizeof(float)));
int tpb=256,grid=(n+tpb-1)/tpb; embed_grad<<<grid,tpb>>>(di,dg,dt,tokens,dim,vocab); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
CK(cudaMemcpy(hy,dt,(size_t)total*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(out,hy,total); cudaFree(di); cudaFree(dg); cudaFree(dt); free(hi); free(hg); free(hy); return 0; }
