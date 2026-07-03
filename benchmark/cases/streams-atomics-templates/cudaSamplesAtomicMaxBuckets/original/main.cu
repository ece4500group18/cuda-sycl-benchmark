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

__global__ void bucket_max(int*out,int n,int buckets){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){int b=(i*17+13)&(buckets-1);int v=(i*29+7)&65535;atomicMax(&out[b],v);}}
int main(int argc,char**argv){const int n=262144,buckets=256;const char*outp=(argc>1)?argv[1]:"output/output.txt";int*db;CK(cudaMalloc(&db,(size_t)buckets*sizeof(int)));CK(cudaMemset(db,0,(size_t)buckets*sizeof(int)));bucket_max<<<(n+255)/256,256>>>(db,n,buckets);CK(cudaGetLastError());CK(cudaDeviceSynchronize());int*hb=(int*)malloc((size_t)buckets*sizeof(int));float*hy=(float*)malloc((size_t)buckets*sizeof(float));CK(cudaMemcpy(hb,db,(size_t)buckets*sizeof(int),cudaMemcpyDeviceToHost));for(int i=0;i<buckets;++i)hy[i]=(float)hb[i];write_vec(outp,hy,buckets);cudaFree(db);free(hb);free(hy);return 0;}
