// cudaStream: vector add (C = A + B) split across 4 streams with async copies.
// n=1048576.  A[i]=((i%17)-8)*0.25f, B[i]=((i%23)-11)*0.5f
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}
#define NS 4

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
  float *ha,*hb,*hc;
  CK(cudaHostAlloc(&ha,bytes,cudaHostAllocDefault));
  CK(cudaHostAlloc(&hb,bytes,cudaHostAllocDefault));
  CK(cudaHostAlloc(&hc,bytes,cudaHostAllocDefault));
  for (int i=0;i<n;++i){ ha[i]=genA(i); hb[i]=genB(i); }
  float *da,*db,*dc; CK(cudaMalloc(&da,bytes)); CK(cudaMalloc(&db,bytes)); CK(cudaMalloc(&dc,bytes));
  cudaStream_t s[NS]; for (int k=0;k<NS;++k) CK(cudaStreamCreate(&s[k]));
  int chunk = (n + NS - 1) / NS, tpb = 256;
  for (int k=0;k<NS;++k){
    int off=k*chunk, len=(off+chunk<=n)?chunk:(n-off); if(len<=0) continue;
    size_t cb=(size_t)len*sizeof(float);
    CK(cudaMemcpyAsync(da+off,ha+off,cb,cudaMemcpyHostToDevice,s[k]));
    CK(cudaMemcpyAsync(db+off,hb+off,cb,cudaMemcpyHostToDevice,s[k]));
    vadd<<<(len+tpb-1)/tpb,tpb,0,s[k]>>>(da+off,db+off,dc+off,len);
    CK(cudaMemcpyAsync(hc+off,dc+off,cb,cudaMemcpyDeviceToHost,s[k]));
  }
  CK(cudaDeviceSynchronize());
  for (int k=0;k<NS;++k) cudaStreamDestroy(s[k]);
  FILE *f=fopen(out,"w"); if(!f){fprintf(stderr,"open %s\n",out);return 2;}
  for(int i=0;i<n;++i) fprintf(f,"%.9g\n",hc[i]); fclose(f);
  printf("cudaStream done: n=%d streams=%d -> %s\n", n, NS, out);
  cudaFree(da);cudaFree(db);cudaFree(dc);
  cudaFreeHost(ha);cudaFreeHost(hb);cudaFreeHost(hc);
  return 0;
}
