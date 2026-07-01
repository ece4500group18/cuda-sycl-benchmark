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

__global__ void residual_layernorm(const float *x, const float *r, const float *gamma, const float *beta, float *y, int rows, int cols) {
  extern __shared__ float s[];
  int row=blockIdx.x, tid=threadIdx.x;
  float sum=0.0f, sum2=0.0f;
  for (int c=tid;c<cols;c+=blockDim.x) {
    float v=x[row*cols+c]+r[row*cols+c];
    sum += v; sum2 += v*v;
  }
  s[tid] = sum; s[tid + blockDim.x] = sum2; __syncthreads();
  for (int stride=blockDim.x/2; stride>0; stride>>=1) {
    if (tid < stride) { s[tid] += s[tid+stride]; s[tid+blockDim.x] += s[tid+blockDim.x+stride]; }
    __syncthreads();
  }
  float mean=s[0]/(float)cols;
  float var=s[blockDim.x]/(float)cols - mean*mean;
  float inv=rsqrtf(var + 1.0e-5f);
  for (int c=tid;c<cols;c+=blockDim.x) {
    float v=x[row*cols+c]+r[row*cols+c];
    y[row*cols+c] = (v - mean) * inv * gamma[c] + beta[c];
  }
}

int main(int argc, char **argv) {
  const int rows=256, cols=512, n=rows*cols;
  const char *out=(argc>1)?argv[1]:"output/output.txt";
  float *hx=(float*)malloc((size_t)n*sizeof(float)), *hr=(float*)malloc((size_t)n*sizeof(float)), *hg=(float*)malloc((size_t)cols*sizeof(float)), *hb=(float*)malloc((size_t)cols*sizeof(float)), *hy=(float*)malloc((size_t)n*sizeof(float));
  for (int i=0;i<n;++i) { hx[i]=2.0f*hs(i,123); hr[i]=0.25f*hs(i,321); }
  for (int i=0;i<cols;++i) { hg[i]=1.0f+0.1f*hs(i,55); hb[i]=0.01f*hs(i,66); }
  float *dx,*dr,*dg,*db,*dy; CK(cudaMalloc(&dx,(size_t)n*sizeof(float))); CK(cudaMalloc(&dr,(size_t)n*sizeof(float))); CK(cudaMalloc(&dg,(size_t)cols*sizeof(float))); CK(cudaMalloc(&db,(size_t)cols*sizeof(float))); CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));
  CK(cudaMemcpy(dx,hx,(size_t)n*sizeof(float),cudaMemcpyHostToDevice)); CK(cudaMemcpy(dr,hr,(size_t)n*sizeof(float),cudaMemcpyHostToDevice)); CK(cudaMemcpy(dg,hg,(size_t)cols*sizeof(float),cudaMemcpyHostToDevice)); CK(cudaMemcpy(db,hb,(size_t)cols*sizeof(float),cudaMemcpyHostToDevice));
  residual_layernorm<<<rows,256,512*sizeof(float)>>>(dx,dr,dg,db,dy,rows,cols);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(out,hy,n);
  cudaFree(dx); cudaFree(dr); cudaFree(dg); cudaFree(db); cudaFree(dy); free(hx); free(hr); free(hg); free(hb); free(hy); return 0;
}
