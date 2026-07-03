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

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
int main(int argc,char**argv){const int n=65536;const char*out=(argc>1)?argv[1]:"output/output.txt";int*hk=(int*)malloc((size_t)n*sizeof(int));float*hv=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i){hk[i]=(i*17+13)%n;hv[i]=(float)i;}thrust::device_vector<int> k(hk,hk+n);thrust::device_vector<float> v(hv,hv+n);thrust::sort_by_key(k.begin(),k.end(),v.begin());thrust::copy(v.begin(),v.end(),hy);write_vec(out,hy,n);free(hk);free(hv);free(hy);return 0;}
