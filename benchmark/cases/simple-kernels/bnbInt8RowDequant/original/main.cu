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

__global__ void deq(const signed char *q,const float *scale,float *y,int rows,int cols){int idx=blockIdx.x*blockDim.x+threadIdx.x,n=rows*cols;if(idx<n){int r=idx/cols;y[idx]=(float)q[idx]*scale[r];}}
int main(int argc,char**argv){const int rows=1024,cols=128,n=rows*cols;const char*out=(argc>1)?argv[1]:"output/output.txt";signed char*hq=(signed char*)malloc((size_t)n);float*hscl=(float*)malloc((size_t)rows*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i)hq[i]=(signed char)((int)floorf(h01(i,123)*255.0f)-127);for(int r=0;r<rows;++r)hscl[r]=0.001f+0.01f*h01(r,77);
signed char*dq;float*ds,*dy;CK(cudaMalloc(&dq,(size_t)n));CK(cudaMalloc(&ds,(size_t)rows*sizeof(float)));CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));CK(cudaMemcpy(dq,hq,(size_t)n,cudaMemcpyHostToDevice));CK(cudaMemcpy(ds,hscl,(size_t)rows*sizeof(float),cudaMemcpyHostToDevice));deq<<<(n+255)/256,256>>>(dq,ds,dy,rows,cols);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(dq);cudaFree(ds);cudaFree(dy);free(hq);free(hscl);free(hy);return 0;}
