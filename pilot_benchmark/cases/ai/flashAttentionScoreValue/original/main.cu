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

__global__ void attention_tile(const float *q, const float *k, const float *v, float *out, int queries, int keys, int dim) {
  __shared__ float score[64];
  int qi=blockIdx.x, tid=threadIdx.x;
  if(tid<keys){
    float s=0.0f;
    for(int d=0;d<dim;++d) s += q[qi*dim+d] * k[tid*dim+d];
    score[tid]=s*rsqrtf((float)dim);
  }
  __syncthreads();
  float m=-3.402823e38f;
  for(int i=0;i<keys;++i) m=fmaxf(m,score[i]);
  float denom=0.0f;
  for(int i=0;i<keys;++i){ score[i]=expf(score[i]-m); denom += score[i]; }
  if(tid<dim){
    float acc=0.0f;
    for(int kk=0;kk<keys;++kk) acc += (score[kk]/denom) * v[kk*dim+tid];
    out[qi*dim+tid]=acc;
  }
}

int main(int argc, char **argv) {
  const int queries=64, keys=64, dim=32, nq=queries*dim, nk=keys*dim, no=queries*dim;
  const char *outp=(argc>1)?argv[1]:"output/output.txt";
  float *hq=(float*)malloc((size_t)nq*sizeof(float)), *hk=(float*)malloc((size_t)nk*sizeof(float)), *hv=(float*)malloc((size_t)nk*sizeof(float)), *hy=(float*)malloc((size_t)no*sizeof(float));
  for(int i=0;i<nq;++i) hq[i]=hs(i,123);
  for(int i=0;i<nk;++i){ hk[i]=hs(i,321); hv[i]=hs(i,777); }
  float *dq,*dk,*dv,*dy; CK(cudaMalloc(&dq,(size_t)nq*sizeof(float))); CK(cudaMalloc(&dk,(size_t)nk*sizeof(float))); CK(cudaMalloc(&dv,(size_t)nk*sizeof(float))); CK(cudaMalloc(&dy,(size_t)no*sizeof(float)));
  CK(cudaMemcpy(dq,hq,(size_t)nq*sizeof(float),cudaMemcpyHostToDevice)); CK(cudaMemcpy(dk,hk,(size_t)nk*sizeof(float),cudaMemcpyHostToDevice)); CK(cudaMemcpy(dv,hv,(size_t)nk*sizeof(float),cudaMemcpyHostToDevice));
  attention_tile<<<queries,64>>>(dq,dk,dv,dy,queries,keys,dim);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)no*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(outp,hy,no);
  cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dy); free(hq); free(hk); free(hv); free(hy); return 0;
}
