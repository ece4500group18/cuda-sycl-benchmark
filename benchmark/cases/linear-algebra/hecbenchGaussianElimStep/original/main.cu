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

__global__ void elim(const float*A,float*B,int rows,int cols){int idx=blockIdx.x*blockDim.x+threadIdx.x,total=rows*cols;if(idx<total){int r=idx/cols,c=idx%cols;float pivot=A[c];float factor=A[(r+1)*cols];B[idx]=A[(r+1)*cols+c]-factor*pivot;}}
int main(int argc,char**argv){const int rows=256,cols=256,total=rows*cols;const char*out=(argc>1)?argv[1]:"output/output.txt";float*ha=(float*)malloc((size_t)(rows+1)*cols*sizeof(float)),*hb=(float*)malloc((size_t)total*sizeof(float));for(int i=0;i<(rows+1)*cols;++i)ha[i]=0.01f+0.1f*h01(i,123);
float*da,*db;CK(cudaMalloc(&da,(size_t)(rows+1)*cols*sizeof(float)));CK(cudaMalloc(&db,(size_t)total*sizeof(float)));CK(cudaMemcpy(da,ha,(size_t)(rows+1)*cols*sizeof(float),cudaMemcpyHostToDevice));int tpb=256,grid=(total+tpb-1)/tpb;elim<<<grid,tpb>>>(da,db,rows,cols);CK(cudaGetLastError());CK(cudaDeviceSynchronize());CK(cudaMemcpy(hb,db,(size_t)total*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out,hb,total);cudaFree(da);cudaFree(db);free(ha);free(hb);return 0;}
