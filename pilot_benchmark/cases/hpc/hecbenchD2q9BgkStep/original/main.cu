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

__global__ void d2q9_step(const float *f, float *rho_out, int h, int w) {
  int cell=blockIdx.x*blockDim.x+threadIdx.x;
  int cells=h*w;
  if(cell<cells){
    float rho=0.0f;
    for(int q=0;q<9;++q) rho += f[q*cells + cell];
    float ux=(f[1*cells+cell]-f[3*cells+cell]+f[5*cells+cell]-f[6*cells+cell]-f[7*cells+cell]+f[8*cells+cell])/rho;
    float uy=(f[2*cells+cell]-f[4*cells+cell]+f[5*cells+cell]+f[6*cells+cell]-f[7*cells+cell]-f[8*cells+cell])/rho;
    rho_out[cell]=rho + 0.01f*(ux+uy);
  }
}

int main(int argc, char **argv) {
  const int h=128, w=128, q=9, cells=h*w, n=q*cells; const char *out=(argc>1)?argv[1]:"output/output.txt";
  float *hf=(float*)malloc((size_t)n*sizeof(float)), *hy=(float*)malloc((size_t)cells*sizeof(float));
  for(int i=0;i<n;++i) hf[i]=0.2f+0.01f*h01(i,123);
  float *df,*dy; CK(cudaMalloc(&df,(size_t)n*sizeof(float))); CK(cudaMalloc(&dy,(size_t)cells*sizeof(float)));
  CK(cudaMemcpy(df,hf,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));
  int tpb=256, grid=(cells+tpb-1)/tpb; d2q9_step<<<grid,tpb>>>(df,dy,h,w);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)cells*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(out,hy,cells);
  cudaFree(df); cudaFree(dy); free(hf); free(hy); return 0;
}
