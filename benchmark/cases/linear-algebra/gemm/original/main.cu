// gemm: C = A*B + bias (per-column bias), square N = 128.
//   A[idx]=h01(idx,123), B[idx]=h01(idx,321), bias[c]=h01(c,7)
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}

__host__ __device__ static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

__global__ void gemm(const float *A, const float *B, const float *bias,
                     float *C, int N) {
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (r < N && c < N) {
    float s = 0.0f;
    for (int k = 0; k < N; ++k) s += A[r * N + k] * B[k * N + c];
    C[r * N + c] = s + bias[c];
  }
}

int main(int argc, char **argv) {
  const int N = 128, total = N * N;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  size_t bytes = (size_t)total * sizeof(float);
  float *hA=(float*)malloc(bytes),*hB=(float*)malloc(bytes),*hC=(float*)malloc(bytes);
  float *hbias=(float*)malloc((size_t)N*sizeof(float));
  for (int i=0;i<total;++i){ hA[i]=h01(i,123); hB[i]=h01(i,321); }
  for (int c=0;c<N;++c) hbias[c]=h01(c,7);
  float *dA,*dB,*dC,*dbias;
  CK(cudaMalloc(&dA,bytes)); CK(cudaMalloc(&dB,bytes)); CK(cudaMalloc(&dC,bytes));
  CK(cudaMalloc(&dbias,(size_t)N*sizeof(float)));
  CK(cudaMemcpy(dA,hA,bytes,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB,hB,bytes,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dbias,hbias,(size_t)N*sizeof(float),cudaMemcpyHostToDevice));
  dim3 block(16,16), grid((N+15)/16,(N+15)/16);
  gemm<<<grid,block>>>(dA,dB,dbias,dC,N);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hC,dC,bytes,cudaMemcpyDeviceToHost));
  FILE *f=fopen(out,"w"); if(!f){fprintf(stderr,"open %s\n",out);return 2;}
  for(int i=0;i<total;++i) fprintf(f,"%.9g\n",hC[i]); fclose(f);
  printf("gemm done: N=%d -> %s\n", N, out);
  cudaFree(dA);cudaFree(dB);cudaFree(dC);cudaFree(dbias);free(hA);free(hB);free(hC);free(hbias);
  return 0;
}
