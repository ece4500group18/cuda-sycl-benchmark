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

__global__ void block_sum(const float*x,float*y,int n){__shared__ float s[256];int t=threadIdx.x,i=blockIdx.x*blockDim.x+t;s[t]=(i<n)?x[i]:0.0f;__syncthreads();for(int stride=128;stride>0;stride>>=1){if(t<stride)s[t]+=s[t+stride];__syncthreads();}if(t==0)y[blockIdx.x]=s[0];}
int main(int argc,char**argv){const int n=65536,B=256,outn=n/B;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)outn*sizeof(float));for(int i=0;i<n;++i)hx[i]=0.01f*hs(i,123);float*dx,*dy;CK(cudaMalloc(&dx,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)outn*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));block_sum<<<outn,B>>>(dx,dy,n);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)outn*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,outn);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
