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

__global__ void per_token_q8(const float *x, float *y, int rows, int cols) {
  extern __shared__ float s[];
  int row=blockIdx.x, tid=threadIdx.x;
  float m=0.0f;
  for (int c=tid;c<cols;c+=blockDim.x) m=fmaxf(m, fabsf(x[row*cols+c]));
  s[tid]=m; __syncthreads();
  for (int stride=blockDim.x/2; stride>0; stride>>=1) {
    if (tid < stride) s[tid]=fmaxf(s[tid],s[tid+stride]);
    __syncthreads();
  }
  float scale=s[0]/127.0f + 1.0e-12f;
  for (int c=tid;c<cols;c+=blockDim.x) {
    int q=(int)lrintf(x[row*cols+c]/scale);
    q=max(-127,min(127,q));
    y[row*cols+c]=(float)q*scale;
  }
}

int main(int argc, char **argv) {
  const int rows=512, cols=256, n=rows*cols;
  const char *out=(argc>1)?argv[1]:"output/output.txt";
  float *hx=(float*)malloc((size_t)n*sizeof(float)), *hy=(float*)malloc((size_t)n*sizeof(float));
  for (int i=0;i<n;++i) hx[i]=4.0f*hs(i,123);
  float *dx,*dy; CK(cudaMalloc(&dx,(size_t)n*sizeof(float))); CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));
  CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));
  per_token_q8<<<rows,256,256*sizeof(float)>>>(dx,dy,rows,cols);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(out,hy,n);
  cudaFree(dx); cudaFree(dy); free(hx); free(hy); return 0;
}
