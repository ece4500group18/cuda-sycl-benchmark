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


__device__ static inline float sample2(const float *in, int h, int w, int c, float y, float x, int ch) {
  if (x < 0.0f || y < 0.0f || x > (float)(w - 1) || y > (float)(h - 1)) return 0.0f;
  int x0 = (int)floorf(x), y0 = (int)floorf(y);
  int x1 = clampi(x0 + 1, 0, w - 1), y1 = clampi(y0 + 1, 0, h - 1);
  float wx = x - x0, wy = y - y0;
  float v00 = in[(y0 * w + x0) * c + ch], v01 = in[(y0 * w + x1) * c + ch];
  float v10 = in[(y1 * w + x0) * c + ch], v11 = in[(y1 * w + x1) * c + ch];
  return (1.0f - wy) * ((1.0f - wx) * v00 + wx * v01) + wy * ((1.0f - wx) * v10 + wx * v11);
}

__global__ void warp_affine(const float *in, float *out, int h, int w, int c) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = h * w * c;
  if (idx >= total) return;
  int ch = idx % c, x = (idx / c) % w, y = idx / (c * w);
  float sx = 0.92f * x + 0.12f * y - 2.3f;
  float sy = -0.08f * x + 1.04f * y + 1.7f;
  out[idx] = sample2(in, h, w, c, sy, sx, ch);
}

int main(int argc, char **argv) {
  const int batch = 1, h = 24, w = 28, c = 3, total = batch * h * w * c;
  const char *path = (argc > 1) ? argv[1] : "output/output.txt";
  std::vector<float> in(total), out(total);
  for (int i = 0; i < total; ++i) in[i] = h01(i, 123);
  float *din, *dout; CK(cudaMalloc(&din, total * sizeof(float))); CK(cudaMalloc(&dout, total * sizeof(float)));
  CK(cudaMemcpy(din, in.data(), total * sizeof(float), cudaMemcpyHostToDevice));
  int tpb = 128;
  warp_affine<<<(total + tpb - 1) / tpb, tpb>>>(din, dout, h, w, c);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(out.data(), dout, total * sizeof(float), cudaMemcpyDeviceToHost));
  cudaFree(din); cudaFree(dout);
  return write_output(path, out);
}
