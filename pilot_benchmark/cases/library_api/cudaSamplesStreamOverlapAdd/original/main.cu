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

__global__ void addv(const float*a,const float*b,float*c,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)c[i]=a[i]+b[i];}
int main(int argc,char**argv){const int n=1048576,half=n/2;const char*out=(argc>1)?argv[1]:"output/output.txt";float *ha,*hb,*hc;CK(cudaMallocHost(&ha,(size_t)n*sizeof(float)));CK(cudaMallocHost(&hb,(size_t)n*sizeof(float)));CK(cudaMallocHost(&hc,(size_t)n*sizeof(float)));for(int i=0;i<n;++i){ha[i]=hs(i,123);hb[i]=hs(i,321);}
float *da[2],*db[2],*dc[2];cudaStream_t st[2];for(int s=0;s<2;++s){CK(cudaStreamCreate(&st[s]));CK(cudaMalloc(&da[s],(size_t)half*sizeof(float)));CK(cudaMalloc(&db[s],(size_t)half*sizeof(float)));CK(cudaMalloc(&dc[s],(size_t)half*sizeof(float)));CK(cudaMemcpyAsync(da[s],ha+s*half,(size_t)half*sizeof(float),cudaMemcpyHostToDevice,st[s]));CK(cudaMemcpyAsync(db[s],hb+s*half,(size_t)half*sizeof(float),cudaMemcpyHostToDevice,st[s]));addv<<<(half+255)/256,256,0,st[s]>>>(da[s],db[s],dc[s],half);CK(cudaMemcpyAsync(hc+s*half,dc[s],(size_t)half*sizeof(float),cudaMemcpyDeviceToHost,st[s]));}
CK(cudaDeviceSynchronize());write_vec(out,hc,n);for(int s=0;s<2;++s){cudaFree(da[s]);cudaFree(db[s]);cudaFree(dc[s]);cudaStreamDestroy(st[s]);}cudaFreeHost(ha);cudaFreeHost(hb);cudaFreeHost(hc);return 0;}
