// embedding: gather rows from a [vocab, dim] table by token ids.
//   vocab=10000, dim=128, num_ids=4096.
//   table[v*dim+d] = h01(v*dim+d, 123)
//   ids[j] = floor(h01(j, 777) * vocab)
//   out[j*dim+d] = table[ids[j]*dim + d]
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}

__host__ __device__ static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

__global__ void embed(const float *table, const int *ids, float *out,
                      int num_ids, int dim) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  int total = num_ids * dim;
  if (e < total) {
    int j = e / dim, d = e % dim;
    out[e] = table[(size_t)ids[j] * dim + d];
  }
}

int main(int argc, char **argv) {
  const int vocab = 10000, dim = 128, num_ids = 4096;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  size_t tsz = (size_t)vocab * dim, osz = (size_t)num_ids * dim;
  float *htab=(float*)malloc(tsz*sizeof(float));
  float *hout=(float*)malloc(osz*sizeof(float));
  int *hids=(int*)malloc((size_t)num_ids*sizeof(int));
  for (size_t i=0;i<tsz;++i) htab[i]=h01((unsigned)i,123);
  for (int j=0;j<num_ids;++j){ int v=(int)(h01(j,777)*vocab); hids[j]=v<0?0:(v>=vocab?vocab-1:v); }
  float *dtab,*dout; int *dids;
  CK(cudaMalloc(&dtab,tsz*sizeof(float))); CK(cudaMalloc(&dout,osz*sizeof(float)));
  CK(cudaMalloc(&dids,(size_t)num_ids*sizeof(int)));
  CK(cudaMemcpy(dtab,htab,tsz*sizeof(float),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dids,hids,(size_t)num_ids*sizeof(int),cudaMemcpyHostToDevice));
  int tpb=256, blocks=(int)((osz+tpb-1)/tpb);
  embed<<<blocks,tpb>>>(dtab,dids,dout,num_ids,dim);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hout,dout,osz*sizeof(float),cudaMemcpyDeviceToHost));
  FILE *f=fopen(out,"w"); if(!f){fprintf(stderr,"open %s\n",out);return 2;}
  for(size_t i=0;i<osz;++i) fprintf(f,"%.9g\n",hout[i]); fclose(f);
  printf("embedding done: ids=%d dim=%d -> %s\n", num_ids, dim, out);
  cudaFree(dtab);cudaFree(dout);cudaFree(dids);free(htab);free(hout);free(hids);
  return 0;
}
