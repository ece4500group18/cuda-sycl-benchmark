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

__device__ float bs_normcdf(float x){return 0.5f*(1.0f+erff(x*0.70710678118f));}
__global__ void black_scholes(const float*S,float*K,float*T,float*y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float r=0.02f,sig=0.3f;float sqrtT=sqrtf(T[i]);float d1=(logf(S[i]/K[i])+(r+0.5f*sig*sig)*T[i])/(sig*sqrtT);float d2=d1-sig*sqrtT;y[i]=S[i]*bs_normcdf(d1)-K[i]*expf(-r*T[i])*bs_normcdf(d2);}}
int main(int argc,char**argv){const int n=262144;const char*out=(argc>1)?argv[1]:"output/output.txt";float*S=(float*)malloc((size_t)n*sizeof(float)),*K=(float*)malloc((size_t)n*sizeof(float)),*T=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i){S[i]=10.0f+20.0f*h01(i,1);K[i]=10.0f+20.0f*h01(i,2);T[i]=0.25f+2.0f*h01(i,3);}
float*dS,*dK,*dT,*dy;CK(cudaMalloc(&dS,(size_t)n*sizeof(float)));CK(cudaMalloc(&dK,(size_t)n*sizeof(float)));CK(cudaMalloc(&dT,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));CK(cudaMemcpy(dS,S,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));CK(cudaMemcpy(dK,K,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));CK(cudaMemcpy(dT,T,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));int tpb=256,grid=(n+tpb-1)/tpb;black_scholes<<<grid,tpb>>>(dS,dK,dT,dy,n);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(dS);cudaFree(dK);cudaFree(dT);cudaFree(dy);free(S);free(K);free(T);free(hy);return 0;}
