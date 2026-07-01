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

__global__ void nw_diag(const float*up,const float*left,const float*diag,float*y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float match=diag[i]+((i%7)==0?2.0f:-1.0f);float del=up[i]-1.0f;float ins=left[i]-1.0f;y[i]=fmaxf(match,fmaxf(del,ins));}}
int main(int argc,char**argv){const int n=256;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hu=(float*)malloc((size_t)n*sizeof(float)),*hl=(float*)malloc((size_t)n*sizeof(float)),*hd=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i){hu[i]=(float)(i%17);hl[i]=(float)(i%13);hd[i]=(float)(i%11);}
float*du,*dl,*dd,*dy;CK(cudaMalloc(&du,(size_t)n*sizeof(float)));CK(cudaMalloc(&dl,(size_t)n*sizeof(float)));CK(cudaMalloc(&dd,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));CK(cudaMemcpy(du,hu,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));CK(cudaMemcpy(dl,hl,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));CK(cudaMemcpy(dd,hd,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));int tpb=256,grid=(n+tpb-1)/tpb;nw_diag<<<grid,tpb>>>(du,dl,dd,dy,n);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(du);cudaFree(dl);cudaFree(dd);cudaFree(dy);free(hu);free(hl);free(hd);free(hy);return 0;}
