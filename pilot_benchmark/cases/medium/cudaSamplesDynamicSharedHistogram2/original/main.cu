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

__global__ void shist(int*out,int n,int bins){extern __shared__ int s[];for(int b=threadIdx.x;b<bins;b+=blockDim.x)s[b]=0;__syncthreads();int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){int b=(i*17+3)&(bins-1);atomicAdd(&s[b],1);}__syncthreads();for(int b=threadIdx.x;b<bins;b+=blockDim.x)atomicAdd(&out[b],s[b]);}
int main(int argc,char**argv){const int n=65536,bins=16;const char*outp=(argc>1)?argv[1]:"output/output.txt";int*db;CK(cudaMalloc(&db,(size_t)bins*sizeof(int)));CK(cudaMemset(db,0,(size_t)bins*sizeof(int)));shist<<<(n+255)/256,256,bins*sizeof(int)>>>(db,n,bins);CK(cudaGetLastError());CK(cudaDeviceSynchronize());int*hb=(int*)malloc((size_t)bins*sizeof(int));float*hy=(float*)malloc((size_t)bins*sizeof(float));CK(cudaMemcpy(hb,db,(size_t)bins*sizeof(int),cudaMemcpyDeviceToHost));for(int i=0;i<bins;++i)hy[i]=(float)hb[i];write_vec(outp,hy,bins);cudaFree(db);free(hb);free(hy);return 0;}
