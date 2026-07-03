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

__global__ void fw_step(const float*x,float*y,int n,int k){int idx=blockIdx.x*blockDim.x+threadIdx.x,total=n*n;if(idx<total){int i=idx/n,j=idx%n;float via=x[i*n+k]+x[k*n+j];y[idx]=fminf(x[idx],via);}}
int main(int argc,char**argv){const int n=256,k=37,total=n*n;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)total*sizeof(float)),*hy=(float*)malloc((size_t)total*sizeof(float));for(int i=0;i<total;++i)hx[i]=(float)((i*17+23)%251);for(int i=0;i<n;++i)hx[i*n+i]=0.0f;
float*dx,*dy;CK(cudaMalloc(&dx,(size_t)total*sizeof(float)));CK(cudaMalloc(&dy,(size_t)total*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)total*sizeof(float),cudaMemcpyHostToDevice));int tpb=256,grid=(total+tpb-1)/tpb;fw_step<<<grid,tpb>>>(dx,dy,n,k);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)total*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,total);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
