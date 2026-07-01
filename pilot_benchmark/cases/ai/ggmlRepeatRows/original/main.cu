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

__global__ void repeat_rows(const float *src, float *dst, int rows, int src_rows, int cols) {
  int idx=blockIdx.x*blockDim.x+threadIdx.x, n=rows*cols;
  if(idx<n){ int r=idx/cols, c=idx%cols; dst[idx]=src[(r%src_rows)*cols+c]; }
}
int main(int argc,char**argv){const int rows=1024,src_rows=256,cols=128,n=rows*cols,ns=src_rows*cols;const char*out=(argc>1)?argv[1]:"output/output.txt";
float *hsr=(float*)malloc((size_t)ns*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<ns;++i)hsr[i]=hs(i,123);
float *ds,*dy;CK(cudaMalloc(&ds,(size_t)ns*sizeof(float)));CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));CK(cudaMemcpy(ds,hsr,(size_t)ns*sizeof(float),cudaMemcpyHostToDevice));
repeat_rows<<<(n+255)/256,256>>>(ds,dy,rows,src_rows,cols);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);
cudaFree(ds);cudaFree(dy);free(hsr);free(hy);return 0;}
