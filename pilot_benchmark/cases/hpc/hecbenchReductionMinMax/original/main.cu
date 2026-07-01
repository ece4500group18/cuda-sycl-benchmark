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

__global__ void minmax_kernel(const float*x,int*mn,int*mx,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){int v=(int)lrintf(1000000.0f*x[i]);atomicMin(mn,v);atomicMax(mx,v);}}
int main(int argc,char**argv){const int n=1048576;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i)hx[i]=hs(i,123);float*dx;int *dmn,*dmx,hmn=2147483647,hmx=-2147483647;CK(cudaMalloc(&dx,(size_t)n*sizeof(float)));CK(cudaMalloc(&dmn,sizeof(int)));CK(cudaMalloc(&dmx,sizeof(int)));CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));CK(cudaMemcpy(dmn,&hmn,sizeof(int),cudaMemcpyHostToDevice));CK(cudaMemcpy(dmx,&hmx,sizeof(int),cudaMemcpyHostToDevice));minmax_kernel<<<(n+255)/256,256>>>(dx,dmn,dmx,n);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(&hmn,dmn,sizeof(int),cudaMemcpyDeviceToHost));CK(cudaMemcpy(&hmx,dmx,sizeof(int),cudaMemcpyDeviceToHost));float outv[2]={(float)hmn/1000000.0f,(float)hmx/1000000.0f};write_vec(out,outv,2);cudaFree(dx);cudaFree(dmn);cudaFree(dmx);free(hx);return 0;}
