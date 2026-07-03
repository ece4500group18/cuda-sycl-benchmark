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

__global__ void hist(const int*b,float*out,int n,int bins){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)atomicAdd(&out[b[i]],1.0f);}
int main(int argc,char**argv){const int n=262144,bins=64;const char*outp=(argc>1)?argv[1]:"output/output.txt";int*hb=(int*)malloc((size_t)n*sizeof(int));float*hy=(float*)calloc((size_t)bins,sizeof(float));for(int i=0;i<n;++i)hb[i]=(int)floorf(h01(i,123)*bins);
int*db;float*dy;CK(cudaMalloc(&db,(size_t)n*sizeof(int)));CK(cudaMalloc(&dy,(size_t)bins*sizeof(float)));CK(cudaMemcpy(db,hb,(size_t)n*sizeof(int),cudaMemcpyHostToDevice));CK(cudaMemset(dy,0,(size_t)bins*sizeof(float)));int tpb=256,grid=(n+tpb-1)/tpb;hist<<<grid,tpb>>>(db,dy,n,bins);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)bins*sizeof(float),cudaMemcpyDeviceToHost));write_vec(outp,hy,bins);cudaFree(db);cudaFree(dy);free(hb);free(hy);return 0;}
