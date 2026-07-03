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

__global__ void layer_forward(const float *input, const float *weights, float *out, int in_n, int hid_n) {
  int h=blockIdx.x*blockDim.x+threadIdx.x;
  if(h<hid_n){
    float sum=weights[h*(in_n+1)];
    for(int i=0;i<in_n;++i) sum += input[i]*weights[h*(in_n+1)+i+1];
    out[h]=1.0f/(1.0f+expf(-sum));
  }
}

int main(int argc, char **argv) {
  const int in_n=512, hid_n=256; const char *outp=(argc>1)?argv[1]:"output/output.txt";
  float *hi=(float*)malloc((size_t)in_n*sizeof(float)), *hw=(float*)malloc((size_t)hid_n*(in_n+1)*sizeof(float)), *hy=(float*)malloc((size_t)hid_n*sizeof(float));
  for(int i=0;i<in_n;++i) hi[i]=hs(i,123);
  for(int i=0;i<hid_n*(in_n+1);++i) hw[i]=0.01f*hs(i,321);
  float *di,*dw,*dy; CK(cudaMalloc(&di,(size_t)in_n*sizeof(float))); CK(cudaMalloc(&dw,(size_t)hid_n*(in_n+1)*sizeof(float))); CK(cudaMalloc(&dy,(size_t)hid_n*sizeof(float)));
  CK(cudaMemcpy(di,hi,(size_t)in_n*sizeof(float),cudaMemcpyHostToDevice)); CK(cudaMemcpy(dw,hw,(size_t)hid_n*(in_n+1)*sizeof(float),cudaMemcpyHostToDevice));
  int tpb=128, grid=(hid_n+tpb-1)/tpb; layer_forward<<<grid,tpb>>>(di,dw,dy,in_n,hid_n);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)hid_n*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(outp,hy,hid_n);
  cudaFree(di); cudaFree(dw); cudaFree(dy); free(hi); free(hw); free(hy); return 0;
}
