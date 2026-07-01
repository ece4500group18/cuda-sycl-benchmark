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

__global__ void crop_pad(const float*x,float*y,int H0,int W0,int H1,int W1){int idx=blockIdx.x*blockDim.x+threadIdx.x,n=H1*W1;if(idx<n){int r=idx/W1,c=idx%W1;float v=0.0f;if(r<32&&c<32){int sr=r,sc=31-c;v=x[sr*W0+sc];}y[idx]=v;}}
int main(int argc,char**argv){const int H0=32,W0=32,H1=48,W1=48,n0=H0*W0,n1=H1*W1;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)n0*sizeof(float)),*hy=(float*)malloc((size_t)n1*sizeof(float));for(int i=0;i<n0;++i)hx[i]=h01(i,123);float*dx,*dy;CK(cudaMalloc(&dx,(size_t)n0*sizeof(float)));CK(cudaMalloc(&dy,(size_t)n1*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)n0*sizeof(float),cudaMemcpyHostToDevice));crop_pad<<<(n1+255)/256,256>>>(dx,dy,H0,W0,H1,W1);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n1*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n1);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
