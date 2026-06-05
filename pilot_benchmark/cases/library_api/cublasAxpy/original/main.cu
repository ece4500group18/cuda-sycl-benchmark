// cublasAxpy: y = alpha*x + y via cuBLAS SAXPY. alpha=2.0, n=1048576.
//   x[i]=h01(i,123), y[i]=h01(i,321)
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}

__host__ __device__ static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

int main(int argc, char **argv) {
  const int n = 1048576;
  const float alpha = 2.0f;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  size_t bytes = (size_t)n * sizeof(float);
  float *hx=(float*)malloc(bytes),*hy=(float*)malloc(bytes);
  for (int i=0;i<n;++i){ hx[i]=h01(i,123); hy[i]=h01(i,321); }
  float *dx,*dy; CK(cudaMalloc(&dx,bytes)); CK(cudaMalloc(&dy,bytes));
  CK(cudaMemcpy(dx,hx,bytes,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dy,hy,bytes,cudaMemcpyHostToDevice));
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { fprintf(stderr,"cublasCreate failed\n"); return 2; }
  cublasSaxpy(handle, n, &alpha, dx, 1, dy, 1);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,bytes,cudaMemcpyDeviceToHost));
  cublasDestroy(handle);
  FILE *f=fopen(out,"w"); if(!f){fprintf(stderr,"open %s\n",out);return 2;}
  for(int i=0;i<n;++i) fprintf(f,"%.9g\n",hy[i]); fclose(f);
  printf("cublasAxpy done: n=%d -> %s\n", n, out);
  cudaFree(dx);cudaFree(dy);free(hx);free(hy);
  return 0;
}
