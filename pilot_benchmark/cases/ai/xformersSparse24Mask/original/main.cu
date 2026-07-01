#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA %s @%d\n", cudaGetErrorString(e), __LINE__); return 2; \
} } while (0)

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

__global__ void sparse24_mask(const float *x, float *y, int groups) {
  int g=blockIdx.x*blockDim.x+threadIdx.x;
  if(g<groups){
    float v0=x[4*g+0], v1=x[4*g+1], v2=x[4*g+2], v3=x[4*g+3];
    float a0=fabsf(v0), a1=fabsf(v1), a2=fabsf(v2), a3=fabsf(v3);
    int m0=0, m1=1; float b0=a0, b1=a1;
    if(b1>b0){float tb=b0; b0=b1; b1=tb; m0=1; m1=0;}
    if(a2>b0){b1=b0; m1=m0; b0=a2; m0=2;} else if(a2>b1){b1=a2; m1=2;}
    if(a3>b0){b1=b0; m1=m0; b0=a3; m0=3;} else if(a3>b1){b1=a3; m1=3;}
    y[4*g+0]=(m0==0||m1==0)?v0:0.0f;
    y[4*g+1]=(m0==1||m1==1)?v1:0.0f;
    y[4*g+2]=(m0==2||m1==2)?v2:0.0f;
    y[4*g+3]=(m0==3||m1==3)?v3:0.0f;
  }
}

int main(int argc, char **argv) {
  const int groups=65536, n=groups*4; const char *out=(argc>1)?argv[1]:"output/output.txt";
  float *hx=(float*)malloc((size_t)n*sizeof(float)), *hy=(float*)malloc((size_t)n*sizeof(float));
  for(int i=0;i<n;++i) hx[i]=hs(i,123);
  float *dx,*dy; CK(cudaMalloc(&dx,(size_t)n*sizeof(float))); CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));
  CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));
  int tpb=256, grid=(groups+tpb-1)/tpb; sparse24_mask<<<grid,tpb>>>(dx,dy,groups);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(out,hy,n);
  cudaFree(dx); cudaFree(dy); free(hx); free(hy); return 0;
}
