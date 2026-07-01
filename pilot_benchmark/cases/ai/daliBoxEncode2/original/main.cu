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

__global__ void boxenc(float*y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float x1=h01(i,11)*0.5f,y1=h01(i,22)*0.5f,w=0.1f+0.4f*h01(i,33),h=0.1f+0.4f*h01(i,44);y[4*i]=x1+0.5f*w;y[4*i+1]=y1+0.5f*h;y[4*i+2]=logf(w);y[4*i+3]=logf(h);}}
int main(int argc,char**argv){const int n=4096;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hy=(float*)malloc((size_t)4*n*sizeof(float));float*dy;CK(cudaMalloc(&dy,(size_t)4*n*sizeof(float)));boxenc<<<(n+255)/256,256>>>(dy,n);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)4*n*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,4*n);cudaFree(dy);free(hy);return 0;}
