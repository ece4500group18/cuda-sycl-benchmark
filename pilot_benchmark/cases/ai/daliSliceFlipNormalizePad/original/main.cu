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

__global__ void slice_flip_norm(const float*x,float*y,int N,int H,int W,int C){int outH=48,outW=48,total=N*outH*outW*C;int idx=blockIdx.x*blockDim.x+threadIdx.x;if(idx<total){int c=idx%C;int tmp=idx/C;int ow=tmp%outW;tmp/=outW;int oh=tmp%outH;int n=tmp/outH;int ih=oh+8,iw=55-ow;float mean=0.1f*(float)c,stdv=0.5f+0.1f*(float)c;y[idx]=(x[((n*H+ih)*W+iw)*C+c]-mean)/stdv;}}
int main(int argc,char**argv){const int N=4,H=64,W=64,C=3,inN=N*H*W*C,outN=N*48*48*C;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)inN*sizeof(float)),*hy=(float*)malloc((size_t)outN*sizeof(float));for(int i=0;i<inN;++i)hx[i]=h01(i,123);
float*dx,*dy;CK(cudaMalloc(&dx,(size_t)inN*sizeof(float)));CK(cudaMalloc(&dy,(size_t)outN*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)inN*sizeof(float),cudaMemcpyHostToDevice));int tpb=256,grid=(outN+tpb-1)/tpb;slice_flip_norm<<<grid,tpb>>>(dx,dy,N,H,W,C);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)outN*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,outN);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
