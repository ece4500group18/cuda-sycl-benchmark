#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA %s @%d\n", cudaGetErrorString(e), __LINE__); return 2; \
} } while (0)

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

__global__ void attn_scores(const float *q, const float *k, float *scores, int queries, int keys, int dim) {
  int pair = blockIdx.x;
  int qi = pair / keys;
  int ki = pair % keys;
  extern __shared__ float s[];
  int tid = threadIdx.x;
  float sum = 0.0f;
  for (int d=tid; d<dim; d+=blockDim.x) sum += q[qi*dim+d] * k[ki*dim+d];
  s[tid] = sum; __syncthreads();
  for (int stride=blockDim.x/2; stride>0; stride>>=1) {
    if (tid < stride) s[tid] += s[tid+stride];
    __syncthreads();
  }
  if (tid == 0) scores[pair] = s[0] * rsqrtf((float)dim);
}

int main(int argc, char **argv) {
  const int queries=128, keys=64, dim=64, nq=queries*dim, nk=keys*dim, ns=queries*keys;
  const char *out=(argc>1)?argv[1]:"output/output.txt";
  float *hq=(float*)malloc((size_t)nq*sizeof(float)), *hk=(float*)malloc((size_t)nk*sizeof(float)), *hy=(float*)malloc((size_t)ns*sizeof(float));
  for (int i=0;i<nq;++i) hq[i]=hs(i,123);
  for (int i=0;i<nk;++i) hk[i]=hs(i,321);
  float *dq,*dk,*dy; CK(cudaMalloc(&dq,(size_t)nq*sizeof(float))); CK(cudaMalloc(&dk,(size_t)nk*sizeof(float))); CK(cudaMalloc(&dy,(size_t)ns*sizeof(float)));
  CK(cudaMemcpy(dq,hq,(size_t)nq*sizeof(float),cudaMemcpyHostToDevice)); CK(cudaMemcpy(dk,hk,(size_t)nk*sizeof(float),cudaMemcpyHostToDevice));
  attn_scores<<<ns,128,128*sizeof(float)>>>(dq,dk,dy,queries,keys,dim);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)ns*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(out,hy,ns);
  cudaFree(dq); cudaFree(dk); cudaFree(dy); free(hq); free(hk); free(hy); return 0;
}
