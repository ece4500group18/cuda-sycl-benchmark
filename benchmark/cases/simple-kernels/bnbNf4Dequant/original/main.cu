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

__constant__ float lut[16];
__global__ void nf4_dequant(const unsigned char *q, float *y, int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n){ unsigned char p=q[i/2]; int nib=(i&1)?(p>>4):(p&15); y[i]=lut[nib]; }
}
int main(int argc,char**argv){ const int n=131072; const char*out=(argc>1)?argv[1]:"output/output.txt";
float hlut[16]={-1.0f,-0.696f,-0.525f,-0.394f,-0.284f,-0.184f,-0.091f,0.0f,0.079f,0.161f,0.246f,0.338f,0.441f,0.563f,0.723f,1.0f};
unsigned char *hq=(unsigned char*)calloc((size_t)n/2,1); float *hy=(float*)malloc((size_t)n*sizeof(float));
for(int i=0;i<n;++i){ unsigned char nib=(unsigned char)min(15,max(0,(int)floorf(16.0f*h01(i,123)))); if(i&1) hq[i/2]|=(nib<<4); else hq[i/2]|=nib; }
CK(cudaMemcpyToSymbol(lut,hlut,16*sizeof(float))); unsigned char*dq; float*dy; CK(cudaMalloc(&dq,(size_t)n/2)); CK(cudaMalloc(&dy,(size_t)n*sizeof(float))); CK(cudaMemcpy(dq,hq,(size_t)n/2,cudaMemcpyHostToDevice));
int tpb=256,grid=(n+tpb-1)/tpb; nf4_dequant<<<grid,tpb>>>(dq,dy,n); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));
write_vec(out,hy,n); cudaFree(dq); cudaFree(dy); free(hq); free(hy); return 0; }
