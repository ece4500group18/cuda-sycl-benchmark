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

__global__ void mandel(float*y,int W,int H,int maxit){int idx=blockIdx.x*blockDim.x+threadIdx.x,n=W*H;if(idx<n){int px=idx%W,py=idx/W;float cr=-2.0f+3.0f*(float)px/(float)(W-1),ci=-1.5f+3.0f*(float)py/(float)(H-1),zr=0.0f,zi=0.0f;int it=0;while(it<maxit&&zr*zr+zi*zi<=4.0f){float nzr=zr*zr-zi*zi+cr;zi=2.0f*zr*zi+ci;zr=nzr;++it;}y[idx]=(float)it;}}
int main(int argc,char**argv){const int W=256,H=256,maxit=64,n=W*H;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hy=(float*)malloc((size_t)n*sizeof(float));float*dy;CK(cudaMalloc(&dy,(size_t)n*sizeof(float)));mandel<<<(n+255)/256,256>>>(dy,W,H,maxit);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,n);cudaFree(dy);free(hy);return 0;}
