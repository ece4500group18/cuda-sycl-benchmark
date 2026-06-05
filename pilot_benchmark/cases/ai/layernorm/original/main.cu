// layernorm: per-row LayerNorm with gamma=1, beta=0, eps=1e-5.
//   rows=512, cols=1024. x[idx] = h01(idx, 123)
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}

__host__ __device__ static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

__global__ void layernorm(const float *x, float *y, int rows, int cols) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r < rows) {
    const float *xr = x + (size_t)r * cols;
    float *yr = y + (size_t)r * cols;
    float mean = 0.0f;
    for (int c = 0; c < cols; ++c) mean += xr[c];
    mean /= cols;
    float var = 0.0f;
    for (int c = 0; c < cols; ++c) { float d = xr[c] - mean; var += d * d; }
    var /= cols;
    float inv = rsqrtf(var + 1e-5f);
    for (int c = 0; c < cols; ++c) yr[c] = (xr[c] - mean) * inv;
  }
}

int main(int argc, char **argv) {
  const int rows = 512, cols = 1024, total = rows * cols;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  size_t bytes = (size_t)total * sizeof(float);
  float *hx=(float*)malloc(bytes),*hy=(float*)malloc(bytes);
  for (int i=0;i<total;++i) hx[i]=h01(i,123);
  float *dx,*dy; CK(cudaMalloc(&dx,bytes)); CK(cudaMalloc(&dy,bytes));
  CK(cudaMemcpy(dx,hx,bytes,cudaMemcpyHostToDevice));
  int tpb=128, blocks=(rows+tpb-1)/tpb;
  layernorm<<<blocks,tpb>>>(dx,dy,rows,cols);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,bytes,cudaMemcpyDeviceToHost));
  FILE *f=fopen(out,"w"); if(!f){fprintf(stderr,"open %s\n",out);return 2;}
  for(int i=0;i<total;++i) fprintf(f,"%.9g\n",hy[i]); fclose(f);
  printf("layernorm done: %dx%d -> %s\n", rows, cols, out);
  cudaFree(dx);cudaFree(dy);free(hx);free(hy);
  return 0;
}
