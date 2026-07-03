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
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
struct fma_zip { __host__ __device__ float operator()(const thrust::tuple<float,float>& t) const { return thrust::get<0>(t) * thrust::get<1>(t) + 0.125f; } };
int main(int argc,char**argv){const int n=262144;const char*out=(argc>1)?argv[1]:"output/output.txt";float*ha=(float*)malloc((size_t)n*sizeof(float)),*hb=(float*)malloc((size_t)n*sizeof(float)),*hy=(float*)malloc((size_t)n*sizeof(float));for(int i=0;i<n;++i){ha[i]=hs(i,123);hb[i]=hs(i,321);}thrust::device_vector<float>a(ha,ha+n),b(hb,hb+n),y(n);auto first=thrust::make_zip_iterator(thrust::make_tuple(a.begin(),b.begin()));auto last=thrust::make_zip_iterator(thrust::make_tuple(a.end(),b.end()));thrust::transform(first,last,y.begin(),fma_zip());thrust::copy(y.begin(),y.end(),hy);write_vec(out,hy,n);free(ha);free(hb);free(hy);return 0;}
