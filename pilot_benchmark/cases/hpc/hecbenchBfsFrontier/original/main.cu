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

__global__ void bfs_step(const int *src, const int *dst, const int *frontier, int *dist, int *next, int edges) {
  int e=blockIdx.x*blockDim.x+threadIdx.x;
  if(e<edges){
    int u=src[e], v=dst[e];
    if(frontier[u]){
      int old=atomicMin(&dist[v], dist[u]+1);
      if(old>dist[u]+1) next[v]=1;
    }
  }
}

int main(int argc, char **argv) {
  const int nodes=4096, deg=3, edges=nodes*deg; const char *out=(argc>1)?argv[1]:"output/output.txt";
  int *hs=(int*)malloc((size_t)edges*sizeof(int)), *hd=(int*)malloc((size_t)edges*sizeof(int)), *hf=(int*)calloc(nodes,sizeof(int)), *hdis=(int*)malloc((size_t)nodes*sizeof(int)), *hn=(int*)calloc(nodes,sizeof(int));
  for(int i=0;i<nodes;++i){ hdis[i]=1000000; if(i%97==0){hf[i]=1; hdis[i]=0;} }
  for(int i=0;i<nodes;++i){ hs[3*i]=i; hd[3*i]=(i+1)%nodes; hs[3*i+1]=i; hd[3*i+1]=(i+17)%nodes; hs[3*i+2]=i; hd[3*i+2]=(i*13+7)%nodes; }
  int *ds,*dd,*df,*ddi,*dn; CK(cudaMalloc(&ds,(size_t)edges*sizeof(int))); CK(cudaMalloc(&dd,(size_t)edges*sizeof(int))); CK(cudaMalloc(&df,(size_t)nodes*sizeof(int))); CK(cudaMalloc(&ddi,(size_t)nodes*sizeof(int))); CK(cudaMalloc(&dn,(size_t)nodes*sizeof(int)));
  CK(cudaMemcpy(ds,hs,(size_t)edges*sizeof(int),cudaMemcpyHostToDevice)); CK(cudaMemcpy(dd,hd,(size_t)edges*sizeof(int),cudaMemcpyHostToDevice)); CK(cudaMemcpy(df,hf,(size_t)nodes*sizeof(int),cudaMemcpyHostToDevice)); CK(cudaMemcpy(ddi,hdis,(size_t)nodes*sizeof(int),cudaMemcpyHostToDevice)); CK(cudaMemset(dn,0,(size_t)nodes*sizeof(int)));
  int tpb=256, grid=(edges+tpb-1)/tpb; bfs_step<<<grid,tpb>>>(ds,dd,df,ddi,dn,edges);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hdis,ddi,(size_t)nodes*sizeof(int),cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hn,dn,(size_t)nodes*sizeof(int),cudaMemcpyDeviceToHost));
  float *outv=(float*)malloc((size_t)nodes*sizeof(float)); for(int i=0;i<nodes;++i) outv[i]=(float)(hdis[i]<1000000?hdis[i]:-1) + 0.01f*(float)hn[i];
  write_vec(out,outv,nodes);
  cudaFree(ds); cudaFree(dd); cudaFree(df); cudaFree(ddi); cudaFree(dn); free(hs); free(hd); free(hf); free(hdis); free(hn); free(outv); return 0;
}
