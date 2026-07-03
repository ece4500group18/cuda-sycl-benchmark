// cudaMemcpyAsyncPinned: pinned-host async H2D/D2H around a scale kernel.
//   out = 3 * a, n=1048576.  a[i]=((i%17)-8)*0.25f
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}

__global__ void scale3(const float *a, float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = 3.0f * a[i];
}
static float genA(int i){ return ((i % 17) - 8) * 0.25f; }

int main(int argc, char **argv) {
  const int n = 1048576;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  size_t bytes = (size_t)n * sizeof(float);
  float *ha,*ho;
  CK(cudaHostAlloc(&ha,bytes,cudaHostAllocDefault));
  CK(cudaHostAlloc(&ho,bytes,cudaHostAllocDefault));
  for (int i=0;i<n;++i) ha[i]=genA(i);
  float *da,*dz; CK(cudaMalloc(&da,bytes)); CK(cudaMalloc(&dz,bytes));
  cudaStream_t st; CK(cudaStreamCreate(&st));
  int tpb=256, blocks=(n+tpb-1)/tpb;
  CK(cudaMemcpyAsync(da,ha,bytes,cudaMemcpyHostToDevice,st));
  scale3<<<blocks,tpb,0,st>>>(da,dz,n);
  CK(cudaMemcpyAsync(ho,dz,bytes,cudaMemcpyDeviceToHost,st));
  CK(cudaStreamSynchronize(st));
  cudaStreamDestroy(st);
  FILE *f=fopen(out,"w"); if(!f){fprintf(stderr,"open %s\n",out);return 2;}
  for(int i=0;i<n;++i) fprintf(f,"%.9g\n",ho[i]); fclose(f);
  printf("cudaMemcpyAsyncPinned done: n=%d -> %s\n", n, out);
  cudaFree(da);cudaFree(dz);cudaFreeHost(ha);cudaFreeHost(ho);
  return 0;
}
