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

__global__ void dist_center(const float*x,const float*c,float*y,int pts,int dim){int p=blockIdx.x*blockDim.x+threadIdx.x;if(p<pts){float d=0.0f;for(int j=0;j<dim;++j){float q=x[p*dim+j]-c[j];d+=q*q;}y[p]=d;}}
int main(int argc,char**argv){const int pts=4096,dim=8;const char*out=(argc>1)?argv[1]:"output/output.txt";float*hx=(float*)malloc((size_t)pts*dim*sizeof(float)),*hc=(float*)malloc((size_t)dim*sizeof(float)),*hy=(float*)malloc((size_t)pts*sizeof(float));for(int i=0;i<pts*dim;++i)hx[i]=hs(i,123);for(int i=0;i<dim;++i)hc[i]=hs(i,321);
float*dx,*dc,*dy;CK(cudaMalloc(&dx,(size_t)pts*dim*sizeof(float)));CK(cudaMalloc(&dc,(size_t)dim*sizeof(float)));CK(cudaMalloc(&dy,(size_t)pts*sizeof(float)));CK(cudaMemcpy(dx,hx,(size_t)pts*dim*sizeof(float),cudaMemcpyHostToDevice));CK(cudaMemcpy(dc,hc,(size_t)dim*sizeof(float),cudaMemcpyHostToDevice));int tpb=256,grid=(pts+tpb-1)/tpb;dist_center<<<grid,tpb>>>(dx,dc,dy,pts,dim);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hy,dy,(size_t)pts*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hy,pts);cudaFree(dx);cudaFree(dc);cudaFree(dy);free(hx);free(hc);free(hy);return 0;}
