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


__global__ void crop_hwc(const float *in, float *out, int batch, int h, int w, int c, int oh, int ow) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * oh * ow * c;
  if (idx >= total) return;
  int ch = idx % c;
  int x = (idx / c) % ow;
  int y = (idx / (c * ow)) % oh;
  int n = idx / (c * ow * oh);
  int y0 = (n == 0) ? 5 : 9;
  int x0 = (n == 0) ? 7 : 3;
  int in_idx = ((n * h + y0 + y) * w + x0 + x) * c + ch;
  out[idx] = in[in_idx];
}

int main(int argc, char **argv) {
  const int batch = 2, h = 32, w = 32, c = 3, oh = 16, ow = 16;
  const int in_total = batch * h * w * c, out_total = batch * oh * ow * c;
  const char *path = (argc > 1) ? argv[1] : "output/output.txt";
  std::vector<float> in(in_total), out(out_total);
  for (int i = 0; i < in_total; ++i) in[i] = h01(i, 123);
  float *din, *dout; CK(cudaMalloc(&din, in_total * sizeof(float))); CK(cudaMalloc(&dout, out_total * sizeof(float)));
  CK(cudaMemcpy(din, in.data(), in_total * sizeof(float), cudaMemcpyHostToDevice));
  int tpb = 128;
  crop_hwc<<<(out_total + tpb - 1) / tpb, tpb>>>(din, dout, batch, h, w, c, oh, ow);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(out.data(), dout, out_total * sizeof(float), cudaMemcpyDeviceToHost));
  cudaFree(din); cudaFree(dout);
  return write_output(path, out);
}
