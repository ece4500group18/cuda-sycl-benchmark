#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA %s @%d\n", cudaGetErrorString(e), __LINE__); return 2; } } while (0)

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

__global__ void bgemm(const float*A,const float*B,float*C,int batch,int M,int N,int K){ int b=blockIdx.z,row=threadIdx.y,col=threadIdx.x; if(row<M&&col<N){float acc=0.0f; for(int k=0;k<K;++k) acc+=A[(b*M+row)*K+k]*B[(b*K+k)*N+col]; C[(b*M+row)*N+col]=acc;}}
int main(int argc,char**argv){const int batch=32,M=16,N=16,K=16,nc=batch*M*N; const char*out=(argc>1)?argv[1]:"output/output.txt";
float *ha=(float*)malloc((size_t)batch*M*K*sizeof(float)),*hb=(float*)malloc((size_t)batch*K*N*sizeof(float)),*hc=(float*)malloc((size_t)nc*sizeof(float));
for(int i=0;i<batch*M*K;++i)ha[i]=0.2f*hs(i,123); for(int i=0;i<batch*K*N;++i)hb[i]=0.2f*hs(i,321);
float*da,*db,*dc; CK(cudaMalloc(&da,(size_t)batch*M*K*sizeof(float))); CK(cudaMalloc(&db,(size_t)batch*K*N*sizeof(float))); CK(cudaMalloc(&dc,(size_t)nc*sizeof(float)));
CK(cudaMemcpy(da,ha,(size_t)batch*M*K*sizeof(float),cudaMemcpyHostToDevice)); CK(cudaMemcpy(db,hb,(size_t)batch*K*N*sizeof(float),cudaMemcpyHostToDevice)); bgemm<<<dim3(1,1,batch),dim3(N,M)>>>(da,db,dc,batch,M,N,K);
CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(hc,dc,(size_t)nc*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(out,hc,nc);
cudaFree(da); cudaFree(db); cudaFree(dc); free(ha); free(hb); free(hc); return 0;}
