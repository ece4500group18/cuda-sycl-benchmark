// cublasGemm: row-major C = A*B via cuBLAS SGEMM (column-major trick).
//   N=256. A[idx]=h01(idx,123), B[idx]=h01(idx,321).
// cuBLAS is column-major; computing C_cm = B*A yields row-major A*B.
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
  const int N = 256, total = N * N;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  size_t bytes = (size_t)total * sizeof(float);
  float *hA=(float*)malloc(bytes),*hB=(float*)malloc(bytes),*hC=(float*)malloc(bytes);
  for (int i=0;i<total;++i){ hA[i]=h01(i,123); hB[i]=h01(i,321); }
  float *dA,*dB,*dC;
  CK(cudaMalloc(&dA,bytes)); CK(cudaMalloc(&dB,bytes)); CK(cudaMalloc(&dC,bytes));
  CK(cudaMemcpy(dA,hA,bytes,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB,hB,bytes,cudaMemcpyHostToDevice));
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { fprintf(stderr,"cublasCreate failed\n"); return 2; }
  const float alpha = 1.0f, beta = 0.0f;
  // C_cm = dB_cm * dA_cm = B^T * A^T = (A*B)^T -> row-major C = A*B
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, dB, N, dA, N, &beta, dC, N);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hC,dC,bytes,cudaMemcpyDeviceToHost));
  cublasDestroy(handle);
  FILE *f=fopen(out,"w"); if(!f){fprintf(stderr,"open %s\n",out);return 2;}
  for(int i=0;i<total;++i) fprintf(f,"%.9g\n",hC[i]); fclose(f);
  printf("cublasGemm done: N=%d -> %s\n", N, out);
  cudaFree(dA);cudaFree(dB);cudaFree(dC);free(hA);free(hB);free(hC);
  return 0;
}
