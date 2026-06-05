// topk: top-8 largest values per row, sorted descending. rows=256, cols=512.
//   x[idx] = h01(idx, 123).  Output: rows*8 values.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}
#define K 8

__host__ __device__ static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

__global__ void topk(const float *x, float *out, int rows, int cols) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r < rows) {
    const float *xr = x + (size_t)r * cols;
    int chosen[K];
    for (int t = 0; t < K; ++t) {
      float best = -1e30f; int bi = -1;
      for (int c = 0; c < cols; ++c) {
        bool taken = false;
        for (int u = 0; u < t; ++u) if (chosen[u] == c) { taken = true; break; }
        if (!taken && xr[c] > best) { best = xr[c]; bi = c; }
      }
      chosen[t] = bi;
      out[(size_t)r * K + t] = best;
    }
  }
}

int main(int argc, char **argv) {
  const int rows = 256, cols = 512, total = rows * cols;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  size_t bytes = (size_t)total * sizeof(float);
  size_t osz = (size_t)rows * K;
  float *hx=(float*)malloc(bytes),*ho=(float*)malloc(osz*sizeof(float));
  for (int i=0;i<total;++i) hx[i]=h01(i,123);
  float *dx,*dout; CK(cudaMalloc(&dx,bytes)); CK(cudaMalloc(&dout,osz*sizeof(float)));
  CK(cudaMemcpy(dx,hx,bytes,cudaMemcpyHostToDevice));
  int tpb=128, blocks=(rows+tpb-1)/tpb;
  topk<<<blocks,tpb>>>(dx,dout,rows,cols);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(ho,dout,osz*sizeof(float),cudaMemcpyDeviceToHost));
  FILE *f=fopen(out,"w"); if(!f){fprintf(stderr,"open %s\n",out);return 2;}
  for(size_t i=0;i<osz;++i) fprintf(f,"%.9g\n",ho[i]); fclose(f);
  printf("topk done: rows=%d cols=%d k=%d -> %s\n", rows, cols, K, out);
  cudaFree(dx);cudaFree(dout);free(hx);free(ho);
  return 0;
}
