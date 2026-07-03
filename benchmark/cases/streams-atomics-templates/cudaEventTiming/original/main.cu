// cudaEventTiming: time a vector add (C = A + B) with CUDA events.
// Correctness is the add result. n=1048576.
//   A[i]=((i%17)-8)*0.25f, B[i]=((i%23)-11)*0.5f
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}

__global__ void vadd(const float *a, const float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
static float genA(int i){ return ((i % 17) - 8) * 0.25f; }
static float genB(int i){ return ((i % 23) - 11) * 0.5f; }

int main(int argc, char **argv) {
  const int n = 1048576;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  size_t bytes = (size_t)n * sizeof(float);
  float *ha=(float*)malloc(bytes),*hb=(float*)malloc(bytes),*hc=(float*)malloc(bytes);
  for (int i=0;i<n;++i){ ha[i]=genA(i); hb[i]=genB(i); }
  float *da,*db,*dc; CK(cudaMalloc(&da,bytes)); CK(cudaMalloc(&db,bytes)); CK(cudaMalloc(&dc,bytes));
  CK(cudaMemcpy(da,ha,bytes,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(db,hb,bytes,cudaMemcpyHostToDevice));
  cudaEvent_t start, stop; CK(cudaEventCreate(&start)); CK(cudaEventCreate(&stop));
  int tpb=256, blocks=(n+tpb-1)/tpb;
  CK(cudaEventRecord(start));
  vadd<<<blocks,tpb>>>(da,db,dc,n);
  CK(cudaEventRecord(stop));
  CK(cudaEventSynchronize(stop));
  float ms=0; CK(cudaEventElapsedTime(&ms,start,stop));
  CK(cudaMemcpy(hc,dc,bytes,cudaMemcpyDeviceToHost));
  cudaEventDestroy(start); cudaEventDestroy(stop);
  FILE *f=fopen(out,"w"); if(!f){fprintf(stderr,"open %s\n",out);return 2;}
  for(int i=0;i<n;++i) fprintf(f,"%.9g\n",hc[i]); fclose(f);
  printf("cudaEventTiming done: n=%d kernel=%.3f ms -> %s\n", n, ms, out);
  cudaFree(da);cudaFree(db);cudaFree(dc);free(ha);free(hb);free(hc);
  return 0;
}
