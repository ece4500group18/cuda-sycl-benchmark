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

__global__ void fdtd3d(const float *x, float *y, int nx, int ny, int nz) {
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  int n=nx*ny*nz;
  if(idx<n){
    int k=idx%nz; int j=(idx/nz)%ny; int i=idx/(ny*nz);
    int im=max(i-1,0), jm=max(j-1,0), km=max(k-1,0);
    y[idx]=x[idx]+0.1f*(x[(im*ny+j)*nz+k]+x[(i*ny+jm)*nz+k]+x[(i*ny+j)*nz+km]-3.0f*x[idx]);
  }
}

int main(int argc, char **argv) {
  const int nx=48, ny=48, nz=32, n=nx*ny*nz; const char *out=(argc>1)?argv[1]:"output/output.txt";
  float *hx=(float*)malloc((size_t)n*sizeof(float)), *hy=(float*)malloc((size_t)n*sizeof(float));
  for(int i=0;i<n;++i) hx[i]=hs(i,123);
  float *dx,*dy; CK(cudaMalloc(&dx,(size_t)n*sizeof(float))); CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));
  CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));
  int tpb=256, grid=(n+tpb-1)/tpb; fdtd3d<<<grid,tpb>>>(dx,dy,nx,ny,nz);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(out,hy,n);
  cudaFree(dx); cudaFree(dy); free(hx); free(hy); return 0;
}
