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


__global__ void resize_bilinear(const float *in, float *out, int batch, int ih, int iw, int c, int oh, int ow) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * oh * ow * c;
  if (idx >= total) return;
  int ch = idx % c, ox = (idx / c) % ow, oy = (idx / (c * ow)) % oh, n = idx / (c * ow * oh);
  float fy = ((float)oy + 0.5f) * ih / oh - 0.5f;
  float fx = ((float)ox + 0.5f) * iw / ow - 0.5f;
  int y0 = clampi((int)floorf(fy), 0, ih - 1), x0 = clampi((int)floorf(fx), 0, iw - 1);
  int y1 = clampi(y0 + 1, 0, ih - 1), x1 = clampi(x0 + 1, 0, iw - 1);
  float wy = fy - floorf(fy), wx = fx - floorf(fx);
  int base = n * ih * iw * c;
  float v00 = in[base + (y0 * iw + x0) * c + ch], v01 = in[base + (y0 * iw + x1) * c + ch];
  float v10 = in[base + (y1 * iw + x0) * c + ch], v11 = in[base + (y1 * iw + x1) * c + ch];
  out[idx] = (1.0f - wy) * ((1.0f - wx) * v00 + wx * v01) + wy * ((1.0f - wx) * v10 + wx * v11);
}

int main(int argc, char **argv) {
  const int batch = 2, ih = 18, iw = 20, c = 3, oh = 28, ow = 24;
  const int in_total = batch * ih * iw * c, out_total = batch * oh * ow * c;
  const char *path = (argc > 1) ? argv[1] : "output/output.txt";
  std::vector<float> in(in_total), out(out_total);
  for (int i = 0; i < in_total; ++i) in[i] = h01(i, 123);
  float *din, *dout; CK(cudaMalloc(&din, in_total * sizeof(float))); CK(cudaMalloc(&dout, out_total * sizeof(float)));
  CK(cudaMemcpy(din, in.data(), in_total * sizeof(float), cudaMemcpyHostToDevice));
  int tpb = 128;
  resize_bilinear<<<(out_total + tpb - 1) / tpb, tpb>>>(din, dout, batch, ih, iw, c, oh, ow);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(out.data(), dout, out_total * sizeof(float), cudaMemcpyDeviceToHost));
  cudaFree(din); cudaFree(dout);
  return write_output(path, out);
}
