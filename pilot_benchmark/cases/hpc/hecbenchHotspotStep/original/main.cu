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

__global__ void hotspot(const float*t,const float*p,float*y,int H,int W){int idx=blockIdx.x*blockDim.x+threadIdx.x,n=H*W;if(idx<n){int r=idx/W,c=idx%W;int up=max(r-1,0),dn=min(r+1,H-1),lf=max(c-1,0),rt=min(c+1,W-1);y[idx]=t[idx]+0.05f*(t[up*W+c]+t[dn*W+c]+t[r*W+lf]+t[r*W+rt]-4.0f*t[idx])+0.01f*p[idx];}}
int main(int argc,char**argv){const int H=256,W=256,n=H*W;const char*out=(argc>1)?argv[1]:"output/output.txt";float*ht=(float*)malloc((size_t)n*sizeof(float)),*hp=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i){ht[i]=hs(i,123);hp[i]=h01(i,321);}
float*dt,*dp,*dy;CK(cudaMalloc(&dt,(size_t)n*sizeof(float)));CK(cudaMalloc(&dp,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));CK(cudaMemcpy(dt,ht,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));CK(cudaMemcpy(dp,hp,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));int tpb=256,grid=(n+tpb-1)/tpb;hotspot<<<grid,tpb>>>(dt,dp,dy,H,W);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(dt);cudaFree(dp);cudaFree(dy);free(ht);free(hp);free(hy);return 0;}
