// cudaGraph: capture a scale kernel into a CUDA graph, then launch it.
//   out = 2*a + 1, n=1048576.  a[i]=((i%17)-8)*0.25f
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}

__global__ void affine(const float *a, float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = 2.0f * a[i] + 1.0f;
}
static float genA(int i){ return ((i % 17) - 8) * 0.25f; }

int main(int argc, char **argv) {
  const int n = 1048576;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  size_t bytes = (size_t)n * sizeof(float);
  float *ha=(float*)malloc(bytes),*ho=(float*)malloc(bytes);
  for (int i=0;i<n;++i) ha[i]=genA(i);
  float *da,*dz; CK(cudaMalloc(&da,bytes)); CK(cudaMalloc(&dz,bytes));
  CK(cudaMemcpy(da,ha,bytes,cudaMemcpyHostToDevice));
  cudaStream_t st; CK(cudaStreamCreate(&st));
  int tpb=256, blocks=(n+tpb-1)/tpb;

  cudaGraph_t graph; cudaGraphExec_t exec;
  CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
  affine<<<blocks,tpb,0,st>>>(da,dz,n);
  CK(cudaStreamEndCapture(st, &graph));
  CK(cudaGraphInstantiate(&exec, graph, NULL, NULL, 0));
  CK(cudaGraphLaunch(exec, st));
  CK(cudaStreamSynchronize(st));
  CK(cudaMemcpy(ho,dz,bytes,cudaMemcpyDeviceToHost));
  cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); cudaStreamDestroy(st);

  FILE *f=fopen(out,"w"); if(!f){fprintf(stderr,"open %s\n",out);return 2;}
  for(int i=0;i<n;++i) fprintf(f,"%.9g\n",ho[i]); fclose(f);
  printf("cudaGraph done: n=%d -> %s\n", n, out);
  cudaFree(da);cudaFree(dz);free(ha);free(ho);
  return 0;
}
