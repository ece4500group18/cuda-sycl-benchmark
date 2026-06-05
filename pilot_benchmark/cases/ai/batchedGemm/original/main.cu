// batchedGemm: C_b = A_b * B_b for b in [0,batch), N = 64, batch = 8.
//   A_b[idx]=h01(idx, 100+b), B_b[idx]=h01(idx, 200+b)
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}

__host__ __device__ static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

__global__ void bgemm(const float *A, const float *B, float *C, int batch, int N) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  int b = blockIdx.z;
  if (r < N && c < N && b < batch) {
    const float *Ab = A + (size_t)b * N * N;
    const float *Bb = B + (size_t)b * N * N;
    float s = 0.0f;
    for (int k = 0; k < N; ++k) s += Ab[r * N + k] * Bb[k * N + c];
    C[(size_t)b * N * N + r * N + c] = s;
  }
}

int main(int argc, char **argv) {
  const int batch = 8, N = 64, perB = N * N, total = batch * perB;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  size_t bytes = (size_t)total * sizeof(float);
  float *hA=(float*)malloc(bytes),*hB=(float*)malloc(bytes),*hC=(float*)malloc(bytes);
  for (int b=0;b<batch;++b) for (int i=0;i<perB;++i){
    hA[b*perB+i]=h01(i,100+b); hB[b*perB+i]=h01(i,200+b);
  }
  float *dA,*dB,*dC;
  CK(cudaMalloc(&dA,bytes)); CK(cudaMalloc(&dB,bytes)); CK(cudaMalloc(&dC,bytes));
  CK(cudaMemcpy(dA,hA,bytes,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB,hB,bytes,cudaMemcpyHostToDevice));
  dim3 block(16,16), grid((N+15)/16,(N+15)/16,batch);
  bgemm<<<grid,block>>>(dA,dB,dC,batch,N);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hC,dC,bytes,cudaMemcpyDeviceToHost));
  FILE *f=fopen(out,"w"); if(!f){fprintf(stderr,"open %s\n",out);return 2;}
  for(int i=0;i<total;++i) fprintf(f,"%.9g\n",hC[i]); fclose(f);
  printf("batchedGemm done: batch=%d N=%d -> %s\n", batch, N, out);
  cudaFree(dA);cudaFree(dB);cudaFree(dC);free(hA);free(hB);free(hC);
  return 0;
}
