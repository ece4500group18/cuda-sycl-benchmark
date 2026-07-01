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

__global__ void path_step(const float*prev,const float*cost,float*y,int H,int W){int idx=blockIdx.x*blockDim.x+threadIdx.x,n=H*W;if(idx<n){int c=idx%W;float l=prev[max(c-1,0)],m=prev[c],r=prev[min(c+1,W-1)];y[idx]=cost[idx]+fminf(m,fminf(l,r));}}
int main(int argc,char**argv){const int H=256,W=256,n=H*W;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hp=(float*)malloc((size_t)W*sizeof(float)),*hc=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<W;++i)hp[i]=h01(i,77);for(int i=0;i<n;++i)hc[i]=h01(i,123);float*dp,*dc,*dy;CK(cudaMalloc(&dp,(size_t)W*sizeof(float)));CK(cudaMalloc(&dc,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));CK(cudaMemcpy(dp,hp,(size_t)W*sizeof(float),cudaMemcpyHostToDevice));CK(cudaMemcpy(dc,hc,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));path_step<<<(n+255)/256,256>>>(dp,dc,dy,H,W);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(dp);cudaFree(dc);cudaFree(dy);free(hp);free(hc);free(hy);return 0;}
