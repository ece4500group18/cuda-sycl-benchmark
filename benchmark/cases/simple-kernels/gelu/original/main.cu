// gelu: tanh-approximation GELU activation, n = 1048576.
//   x = (2*h01(i,123) - 1) * 3   (range about [-3, 3))
//   gelu(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}

__host__ __device__ static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

__global__ void gelu(const float *x, float *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = x[i];
    const float k = 0.7978845608028654f;  // sqrt(2/pi)
    float inner = k * (v + 0.044715f * v * v * v);
    y[i] = 0.5f * v * (1.0f + tanhf(inner));
  }
}

int main(int argc, char **argv) {
  const int n = 1048576;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  size_t bytes = (size_t)n * sizeof(float);
  float *hx=(float*)malloc(bytes),*hy=(float*)malloc(bytes);
  for (int i=0;i<n;++i) hx[i]=(2.0f*h01(i,123)-1.0f)*3.0f;
  float *dx,*dy; CK(cudaMalloc(&dx,bytes)); CK(cudaMalloc(&dy,bytes));
  CK(cudaMemcpy(dx,hx,bytes,cudaMemcpyHostToDevice));
  int tpb=256, blocks=(n+tpb-1)/tpb;
  gelu<<<blocks,tpb>>>(dx,dy,n);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,bytes,cudaMemcpyDeviceToHost));
  FILE *f=fopen(out,"w"); if(!f){fprintf(stderr,"open %s\n",out);return 2;}
  for(int i=0;i<n;++i) fprintf(f,"%.9g\n",hy[i]); fclose(f);
  printf("gelu done: n=%d -> %s\n", n, out);
  cudaFree(dx);cudaFree(dy);free(hx);free(hy);
  return 0;
}
