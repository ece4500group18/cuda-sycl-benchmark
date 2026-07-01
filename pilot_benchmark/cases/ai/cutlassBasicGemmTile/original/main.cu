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

__global__ void gemm_tile(const float *A,const float *B,float*C,int M,int N,int K){
  __shared__ float As[16][16], Bs[16][16]; int row=blockIdx.y*16+threadIdx.y, col=blockIdx.x*16+threadIdx.x; float acc=0.0f;
  for(int t=0;t<K;t+=16){ As[threadIdx.y][threadIdx.x]=(row<M&&t+threadIdx.x<K)?A[row*K+t+threadIdx.x]:0.0f; Bs[threadIdx.y][threadIdx.x]=(t+threadIdx.y<K&&col<N)?B[(t+threadIdx.y)*N+col]:0.0f; __syncthreads(); for(int k=0;k<16;++k) acc+=As[threadIdx.y][k]*Bs[k][threadIdx.x]; __syncthreads(); }
  if(row<M&&col<N) C[row*N+col]=acc;
}
int main(int argc,char**argv){ const int M=128,N=128,K=64,nc=M*N; const char*out=(argc>1)?argv[1]:"output/output.txt";
float *ha=(float*)malloc((size_t)M*K*sizeof(float)),*hb=(float*)malloc((size_t)K*N*sizeof(float)),*hc=(float*)malloc((size_t)nc*sizeof(float));
for(int i=0;i<M*K;++i)ha[i]=0.1f*hs(i,123); for(int i=0;i<K*N;++i)hb[i]=0.1f*hs(i,321);
float *da,*db,*dc; CK(cudaMalloc(&da,(size_t)M*K*sizeof(float))); CK(cudaMalloc(&db,(size_t)K*N*sizeof(float))); CK(cudaMalloc(&dc,(size_t)nc*sizeof(float)));
CK(cudaMemcpy(da,ha,(size_t)M*K*sizeof(float),cudaMemcpyHostToDevice)); CK(cudaMemcpy(db,hb,(size_t)K*N*sizeof(float),cudaMemcpyHostToDevice)); gemm_tile<<<dim3((N+15)/16,(M+15)/16),dim3(16,16)>>>(da,db,dc,M,N,K);
CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(hc,dc,(size_t)nc*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(out,hc,nc);
cudaFree(da); cudaFree(db); cudaFree(dc); free(ha); free(hb); free(hc); return 0; }
