#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA %s @%d\n", cudaGetErrorString(e), __LINE__); return 2; \
} } while (0)

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

__global__ void csr_spmv(const int *row, const int *col, const float *val, const float *x, float *y, int rows) {
  int r=blockIdx.x*blockDim.x+threadIdx.x;
  if(r<rows){
    float sum=0.0f;
    for(int p=row[r]; p<row[r+1]; ++p) sum += val[p] * x[col[p]];
    y[r]=sum;
  }
}

int main(int argc, char **argv) {
  const int rows=4096, nnz_per=7, nnz=rows*nnz_per; const char *out=(argc>1)?argv[1]:"output/output.txt";
  int *hr=(int*)malloc((size_t)(rows+1)*sizeof(int)), *hc=(int*)malloc((size_t)nnz*sizeof(int));
  float *hv=(float*)malloc((size_t)nnz*sizeof(float)), *hx=(float*)malloc((size_t)rows*sizeof(float)), *hy=(float*)malloc((size_t)rows*sizeof(float));
  for(int r=0;r<=rows;++r) hr[r]=r*nnz_per;
  for(int r=0;r<rows;++r){ for(int j=0;j<nnz_per;++j){ int p=r*nnz_per+j; hc[p]=(r+j*13+rows-39)%rows; hv[p]=0.1f+0.01f*(float)j; } hx[r]=hs(r,123); }
  int *dr,*dc; float *dv,*dx,*dy; CK(cudaMalloc(&dr,(size_t)(rows+1)*sizeof(int))); CK(cudaMalloc(&dc,(size_t)nnz*sizeof(int))); CK(cudaMalloc(&dv,(size_t)nnz*sizeof(float))); CK(cudaMalloc(&dx,(size_t)rows*sizeof(float))); CK(cudaMalloc(&dy,(size_t)rows*sizeof(float)));
  CK(cudaMemcpy(dr,hr,(size_t)(rows+1)*sizeof(int),cudaMemcpyHostToDevice)); CK(cudaMemcpy(dc,hc,(size_t)nnz*sizeof(int),cudaMemcpyHostToDevice)); CK(cudaMemcpy(dv,hv,(size_t)nnz*sizeof(float),cudaMemcpyHostToDevice)); CK(cudaMemcpy(dx,hx,(size_t)rows*sizeof(float),cudaMemcpyHostToDevice));
  int tpb=256, grid=(rows+tpb-1)/tpb; csr_spmv<<<grid,tpb>>>(dr,dc,dv,dx,dy,rows);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hy,dy,(size_t)rows*sizeof(float),cudaMemcpyDeviceToHost)); write_vec(out,hy,rows);
  cudaFree(dr); cudaFree(dc); cudaFree(dv); cudaFree(dx); cudaFree(dy); free(hr); free(hc); free(hv); free(hx); free(hy); return 0;
}
