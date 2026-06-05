// attention: single-head scaled dot-product attention. seq=128, dim=64.
//   O = softmax(Q K^T / sqrt(dim)) V
//   Q[idx]=h01(idx,11), K[idx]=h01(idx,22), V[idx]=h01(idx,33)
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}
#define SEQ 128

__host__ __device__ static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

__global__ void attention(const float *Q, const float *K, const float *Vv,
                          float *O, int seq, int dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < seq) {
    float scores[SEQ];
    float scale = rsqrtf((float)dim);
    float m = -1e30f;
    for (int j = 0; j < seq; ++j) {
      float s = 0.0f;
      for (int d = 0; d < dim; ++d) s += Q[i * dim + d] * K[j * dim + d];
      s *= scale; scores[j] = s; if (s > m) m = s;
    }
    float sum = 0.0f;
    for (int j = 0; j < seq; ++j) { scores[j] = expf(scores[j] - m); sum += scores[j]; }
    for (int d = 0; d < dim; ++d) {
      float acc = 0.0f;
      for (int j = 0; j < seq; ++j) acc += scores[j] * Vv[j * dim + d];
      O[i * dim + d] = acc / sum;
    }
  }
}

int main(int argc, char **argv) {
  const int seq = SEQ, dim = 64, total = seq * dim;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  size_t bytes = (size_t)total * sizeof(float);
  float *hQ=(float*)malloc(bytes),*hK=(float*)malloc(bytes),*hV=(float*)malloc(bytes),*hO=(float*)malloc(bytes);
  for (int i=0;i<total;++i){ hQ[i]=h01(i,11); hK[i]=h01(i,22); hV[i]=h01(i,33); }
  float *dQ,*dK,*dV,*dO;
  CK(cudaMalloc(&dQ,bytes)); CK(cudaMalloc(&dK,bytes)); CK(cudaMalloc(&dV,bytes)); CK(cudaMalloc(&dO,bytes));
  CK(cudaMemcpy(dQ,hQ,bytes,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dK,hK,bytes,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dV,hV,bytes,cudaMemcpyHostToDevice));
  int tpb=64, blocks=(seq+tpb-1)/tpb;
  attention<<<blocks,tpb>>>(dQ,dK,dV,dO,seq,dim);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hO,dO,bytes,cudaMemcpyDeviceToHost));
  FILE *f=fopen(out,"w"); if(!f){fprintf(stderr,"open %s\n",out);return 2;}
  for(int i=0;i<total;++i) fprintf(f,"%.9g\n",hO[i]); fclose(f);
  printf("attention done: seq=%d dim=%d -> %s\n", seq, dim, out);
  cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dO);free(hQ);free(hK);free(hV);free(hO);
  return 0;
}
