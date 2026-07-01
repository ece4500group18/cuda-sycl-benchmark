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

__global__ void dct_dc_ac(const float*x,float*y,int blocks){int b=blockIdx.x;float dc=0.0f,ac=0.0f;for(int r=0;r<8;++r){for(int c=0;c<8;++c){float v=x[b*64+r*8+c];dc+=v;ac+=v*cosf((float)(2*c+1)*3.14159265f/16.0f);}}y[2*b]=dc/8.0f;y[2*b+1]=0.5f*ac;}
int main(int argc,char**argv){const int blocks=1024,n=blocks*64;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)2*blocks*sizeof(float));for(int i=0;i<n;++i)hx[i]=h01(i,123);
float*dx,*dy;CK(cudaMalloc(&dx,(size_t)n*sizeof(float)));CK(cudaMalloc(&dy,(size_t)2*blocks*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));dct_dc_ac<<<blocks,1>>>(dx,dy,blocks);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)2*blocks*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,2*blocks);cudaFree(dx);cudaFree(dy);free(hx);free(hy);return 0;}
