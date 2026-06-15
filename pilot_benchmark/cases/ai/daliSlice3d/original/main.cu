#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) { cudaError_t e = (x); if (e) { fprintf(stderr, "CUDA %s @%d\n", cudaGetErrorString(e), __LINE__); return 2; } }

__host__ __device__ static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

__device__ static inline int clampi(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

static int write_output(const char *path, const std::vector<float> &out) {
  FILE *f = fopen(path, "w");
  if (!f) { fprintf(stderr, "open %s\n", path); return 2; }
  for (size_t i = 0; i < out.size(); ++i) fprintf(f, "%.9g\n", out[i]);
  fclose(f);
  return 0;
}


__global__ void slice3d(const float *in, float *out, int batch, int d, int h, int w, int od, int oh, int ow) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * od * oh * ow;
  if (idx >= total) return;
  int x = idx % ow;
  int y = (idx / ow) % oh;
  int z = (idx / (ow * oh)) % od;
  int n = idx / (ow * oh * od);
  int z0 = n == 0 ? 1 : 2;
  int y0 = n == 0 ? 2 : 1;
  int x0 = n == 0 ? 3 : 4;
  out[idx] = in[((n * d + z0 + z) * h + y0 + y) * w + x0 + x];
}

int main(int argc, char **argv) {
  const int batch = 2, d = 6, h = 8, w = 10, od = 3, oh = 4, ow = 5;
  const int in_total = batch * d * h * w, out_total = batch * od * oh * ow;
  const char *path = (argc > 1) ? argv[1] : "output/output.txt";
  std::vector<float> in(in_total), out(out_total);
  for (int i = 0; i < in_total; ++i) in[i] = h01(i, 123);
  float *din, *dout; CK(cudaMalloc(&din, in_total * sizeof(float))); CK(cudaMalloc(&dout, out_total * sizeof(float)));
  CK(cudaMemcpy(din, in.data(), in_total * sizeof(float), cudaMemcpyHostToDevice));
  int tpb = 128;
  slice3d<<<(out_total + tpb - 1) / tpb, tpb>>>(din, dout, batch, d, h, w, od, oh, ow);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(out.data(), dout, out_total * sizeof(float), cudaMemcpyDeviceToHost));
  cudaFree(din); cudaFree(dout);
  return write_output(path, out);
}
