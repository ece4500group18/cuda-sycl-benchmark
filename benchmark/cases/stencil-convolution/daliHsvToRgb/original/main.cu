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

__global__ void hsv_rgb(const float*h,const float*s,const float*v,float*y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float hh=h[i]*6.0f;int k=(int)floorf(hh);float f=hh-k,p=v[i]*(1.0f-s[i]),q=v[i]*(1.0f-s[i]*f),t=v[i]*(1.0f-s[i]*(1.0f-f));float r,g,b;switch(k%6){case 0:r=v[i];g=t;b=p;break;case 1:r=q;g=v[i];b=p;break;case 2:r=p;g=v[i];b=t;break;case 3:r=p;g=q;b=v[i];break;case 4:r=t;g=p;b=v[i];break;default:r=v[i];g=p;b=q;}y[3*i]=r;y[3*i+1]=g;y[3*i+2]=b;}}
int main(int argc,char**argv){const int n=65536;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hh=(float*)malloc((size_t)n*sizeof(float)),*hsat=(float*)malloc((size_t)n*sizeof(float)),*hv=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)3*n*sizeof(float));for(int i=0;i<n;++i){hh[i]=h01(i,11);hsat[i]=h01(i,22);hv[i]=h01(i,33);}
float*dh,*ds,*dv,*dy;CK(cudaMalloc(&dh,(size_t)n*sizeof(float)));CK(cudaMalloc(&ds,(size_t)n*sizeof(float)));CK(cudaMalloc(&dv,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)3*n*sizeof(float)));CK(cudaMemcpy(dh,hh,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));CK(cudaMemcpy(ds,hsat,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));CK(cudaMemcpy(dv,hv,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));int tpb=256,grid=(n+tpb-1)/tpb;hsv_rgb<<<grid,tpb>>>(dh,ds,dv,dy,n);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)3*n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,3*n);cudaFree(dh);cudaFree(ds);cudaFree(dv);cudaFree(dy);free(hh);free(hsat);free(hv);free(hy);return 0;}
