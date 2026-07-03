// reverseArray: out[i] = in[n-1-i], n = 100000.  in[i] = ((i % 17) - 8) * 0.25f
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}

__global__ void reverseArray(const float *in, float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[n - 1 - i];
}
static float genA(int i){ return ((i % 17) - 8) * 0.25f; }

int main(int argc, char **argv) {
  const int n = 100000;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  size_t bytes = (size_t)n * sizeof(float);
  float *hin=(float*)malloc(bytes),*ho=(float*)malloc(bytes);
  for (int i=0;i<n;++i) hin[i]=genA(i);
  float *din,*dout; CK(cudaMalloc(&din,bytes)); CK(cudaMalloc(&dout,bytes));
  CK(cudaMemcpy(din,hin,bytes,cudaMemcpyHostToDevice));
  int tpb=256, blocks=(n+tpb-1)/tpb;
  reverseArray<<<blocks,tpb>>>(din,dout,n);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(ho,dout,bytes,cudaMemcpyDeviceToHost));
  FILE *f=fopen(out,"w"); if(!f){fprintf(stderr,"open %s\n",out);return 2;}
  for(int i=0;i<n;++i) fprintf(f,"%.9g\n",ho[i]); fclose(f);
  printf("reverseArray done: n=%d -> %s\n", n, out);
  cudaFree(din); cudaFree(dout); free(hin); free(ho);
  return 0;
}
