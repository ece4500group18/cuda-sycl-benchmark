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


__device__ static inline float mean_c(int ch) { return ch == 0 ? 0.45f : (ch == 1 ? 0.50f : 0.55f); }
__device__ static inline float std_c(int ch) { return ch == 0 ? 0.20f : (ch == 1 ? 0.25f : 0.30f); }

__global__ void cmn(const float *in, float *out, int batch, int h, int w, int c, int oh, int ow) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * c * oh * ow;
  if (idx >= total) return;
  int x = idx % ow;
  int y = (idx / ow) % oh;
  int ch = (idx / (ow * oh)) % c;
  int n = idx / (ow * oh * c);
  int y0 = n == 0 ? 4 : 8;
  int x0 = n == 0 ? 6 : 5;
  int mirror = n == 1;
  int sx = mirror ? (x0 + ow - 1 - x) : (x0 + x);
  float v = in[((n * h + y0 + y) * w + sx) * c + ch];
  out[idx] = (v - mean_c(ch)) / std_c(ch);
}

int main(int argc, char **argv) {
  const int batch = 2, h = 32, w = 32, c = 3, oh = 16, ow = 16;
  const int in_total = batch * h * w * c, out_total = batch * c * oh * ow;
  const char *path = (argc > 1) ? argv[1] : "output/output.txt";
  std::vector<float> in(in_total), out(out_total);
  for (int i = 0; i < in_total; ++i) in[i] = h01(i, 123);
  float *din, *dout; CK(cudaMalloc(&din, in_total * sizeof(float))); CK(cudaMalloc(&dout, out_total * sizeof(float)));
  CK(cudaMemcpy(din, in.data(), in_total * sizeof(float), cudaMemcpyHostToDevice));
  int tpb = 128;
  cmn<<<(out_total + tpb - 1) / tpb, tpb>>>(din, dout, batch, h, w, c, oh, ow);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(out.data(), dout, out_total * sizeof(float), cudaMemcpyDeviceToHost));
  cudaFree(din); cudaFree(dout);
  return write_output(path, out);
}
