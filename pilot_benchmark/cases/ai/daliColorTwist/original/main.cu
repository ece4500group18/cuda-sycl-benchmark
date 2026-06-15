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


__global__ void color_twist(const float *in, float *out, int n) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n) return;
  int base = p * 3;
  float r = in[base + 0], g = in[base + 1], b = in[base + 2];
  float brightness = 0.08f, contrast = 1.15f, saturation = 0.85f, hue = 0.25f;
  r = (r - 0.5f) * contrast + 0.5f + brightness;
  g = (g - 0.5f) * contrast + 0.5f + brightness;
  b = (b - 0.5f) * contrast + 0.5f + brightness;
  float y = 0.299f * r + 0.587f * g + 0.114f * b;
  float u = r - y, v = b - y;
  float c = cosf(hue), s = sinf(hue);
  float ru = saturation * (c * u - s * v);
  float rv = saturation * (s * u + c * v);
  out[base + 0] = y + ru;
  out[base + 1] = y - 0.509f * ru - 0.194f * rv;
  out[base + 2] = y + rv;
}

int main(int argc, char **argv) {
  const int batch = 2, h = 16, w = 16, c = 3, total = batch * h * w * c;
  const char *path = (argc > 1) ? argv[1] : "output/output.txt";
  std::vector<float> in(total), out(total);
  for (int i = 0; i < total; ++i) in[i] = h01(i, 123);
  float *din, *dout; CK(cudaMalloc(&din, total * sizeof(float))); CK(cudaMalloc(&dout, total * sizeof(float)));
  CK(cudaMemcpy(din, in.data(), total * sizeof(float), cudaMemcpyHostToDevice));
  int pixels = batch * h * w, tpb = 128;
  color_twist<<<(pixels + tpb - 1) / tpb, tpb>>>(din, dout, pixels);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(out.data(), dout, total * sizeof(float), cudaMemcpyDeviceToHost));
  cudaFree(din); cudaFree(dout);
  return write_output(path, out);
}
