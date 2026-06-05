// curandUniform: generate n=1048576 uniform [0,1) floats with cuRAND.
// Verified statistically (mean ~ 0.5, all values in [0,1)) since the exact
// stream is RNG-implementation specific.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <curand.h>
#define CK(x){cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);return 2;}}

int main(int argc, char **argv) {
  const int n = 1048576;
  const char *out = (argc > 1) ? argv[1] : "output/output.txt";
  size_t bytes = (size_t)n * sizeof(float);
  float *h = (float *)malloc(bytes);
  float *d; CK(cudaMalloc(&d, bytes));
  curandGenerator_t gen;
  if (curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS) {
    fprintf(stderr, "curandCreateGenerator failed\n"); return 2;
  }
  curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
  curandGenerateUniform(gen, d, n);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost));
  curandDestroyGenerator(gen);
  FILE *f = fopen(out, "w"); if (!f) { fprintf(stderr, "open %s\n", out); return 2; }
  for (int i = 0; i < n; ++i) fprintf(f, "%.9g\n", h[i]); fclose(f);
  printf("curandUniform done: n=%d -> %s\n", n, out);
  cudaFree(d); free(h);
  return 0;
}
