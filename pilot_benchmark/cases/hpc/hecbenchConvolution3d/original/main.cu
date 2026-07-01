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

__global__ void conv3d7(const float *x, float *y, int n) {
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  int total=n*n*n;
  if(idx<total){
    int k=idx%n; int j=(idx/n)%n; int i=idx/(n*n);
    int im=max(i-1,0), ip=min(i+1,n-1), jm=max(j-1,0), jp=min(j+1,n-1), km=max(k-1,0), kp=min(k+1,n-1);
    y[idx]=0.4f*x[idx]+0.1f*(x[(im*n+j)*n+k]+x[(ip*n+j)*n+k]+x[(i*n+jm)*n+k]+x[(i*n+jp)*n+k]+x[(i*n+j)*n+km]+x[(i*n+j)*n+kp]);
  }
}

int main(int argc, char **argv) {
  const int n=32, total=n*n*n; const char *out=(argc>1)?argv[1]:"output/output.txt";
  float *hx=(float*)malloc((size_t)total*sizeof(float)), *hy=(float*)malloc((size_t)total*sizeof(float));
  for(int i=0;i<total;++i) hx[i]=hs(i,123);
  float *dx,*dy; CK(cudaMalloc(&dx,(size_t)total*sizeof(float))); CK(cudaMalloc(&dy,(size_t)total*sizeof(float)));
  CK(cudaMemcpy(dx,hx,(size_t)total*sizeof(float),cudaMemcpyHostToDevice));
  int tpb=256, grid=(total+tpb-1)/tpb; conv3d7<<<grid,tpb>>>(dx,dy,n);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)total*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(out,hy,total);
  cudaFree(dx); cudaFree(dy); free(hx); free(hy); return 0;
}
