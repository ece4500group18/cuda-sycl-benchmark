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

__global__ void path_step(const float*prev,const float*cost,float*y,int rows,int cols){int c=blockIdx.x*blockDim.x+threadIdx.x;if(c<cols){float m=prev[c];if(c>0)m=fminf(m,prev[c-1]);if(c+1<cols)m=fminf(m,prev[c+1]);y[c]=cost[c]+m;}}
int main(int argc,char**argv){const int rows=512,cols=256;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hp=(float*)malloc((size_t)cols*sizeof(float)),*hc=(float*)malloc((size_t)cols*sizeof(float)),*hy=(float*)malloc((size_t)cols*sizeof(float));for(int i=0;i<cols;++i){hp[i]=(float)((i*7)%31);hc[i]=(float)((i*13)%17);}
float*dp,*dc,*dy;CK(cudaMalloc(&dp,(size_t)cols*sizeof(float)));CK(cudaMalloc(&dc,(size_t)cols*sizeof(float)));CK(cudaMalloc(&dy,(size_t)cols*sizeof(float)));CK(cudaMemcpy(dp,hp,(size_t)cols*sizeof(float),cudaMemcpyHostToDevice));CK(cudaMemcpy(dc,hc,(size_t)cols*sizeof(float),cudaMemcpyHostToDevice));int tpb=256,grid=(cols+tpb-1)/tpb;path_step<<<grid,tpb>>>(dp,dc,dy,rows,cols);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)cols*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,cols);cudaFree(dp);cudaFree(dc);cudaFree(dy);free(hp);free(hc);free(hy);return 0;}
