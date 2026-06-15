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


struct Box { float x1, y1, x2, y2; };

__device__ static inline Box make_anchor(int i) {
  float cx = 0.1f + 0.8f * h01(i * 4 + 0, 123);
  float cy = 0.1f + 0.8f * h01(i * 4 + 1, 123);
  float ww = 0.08f + 0.22f * h01(i * 4 + 2, 123);
  float hh = 0.08f + 0.22f * h01(i * 4 + 3, 123);
  return {fmaxf(0.0f, cx - ww * 0.5f), fmaxf(0.0f, cy - hh * 0.5f), fminf(1.0f, cx + ww * 0.5f), fminf(1.0f, cy + hh * 0.5f)};
}

__device__ static inline Box make_gt(int i) {
  float cx = 0.15f + 0.7f * h01(i * 4 + 11, 321);
  float cy = 0.15f + 0.7f * h01(i * 4 + 12, 321);
  float ww = 0.18f + 0.25f * h01(i * 4 + 13, 321);
  float hh = 0.18f + 0.25f * h01(i * 4 + 14, 321);
  return {fmaxf(0.0f, cx - ww * 0.5f), fmaxf(0.0f, cy - hh * 0.5f), fminf(1.0f, cx + ww * 0.5f), fminf(1.0f, cy + hh * 0.5f)};
}

__device__ static inline float iou(Box a, Box b) {
  float ix1 = fmaxf(a.x1, b.x1), iy1 = fmaxf(a.y1, b.y1);
  float ix2 = fminf(a.x2, b.x2), iy2 = fminf(a.y2, b.y2);
  float iw = fmaxf(0.0f, ix2 - ix1), ih = fmaxf(0.0f, iy2 - iy1);
  float inter = iw * ih;
  float area_a = fmaxf(0.0f, a.x2 - a.x1) * fmaxf(0.0f, a.y2 - a.y1);
  float area_b = fmaxf(0.0f, b.x2 - b.x1) * fmaxf(0.0f, b.y2 - b.y1);
  return inter / (area_a + area_b - inter + 1e-6f);
}

__global__ void encode(float *out, int anchors, int gts) {
  int aidx = blockIdx.x * blockDim.x + threadIdx.x;
  if (aidx >= anchors) return;
  Box a = make_anchor(aidx);
  int best = 0; float best_iou = -1.0f;
  for (int g = 0; g < gts; ++g) {
    float v = iou(a, make_gt(g));
    if (v > best_iou) { best_iou = v; best = g; }
  }
  Box gt = make_gt(best);
  float aw = a.x2 - a.x1, ah = a.y2 - a.y1, acx = 0.5f * (a.x1 + a.x2), acy = 0.5f * (a.y1 + a.y2);
  float gw = gt.x2 - gt.x1, gh = gt.y2 - gt.y1, gcx = 0.5f * (gt.x1 + gt.x2), gcy = 0.5f * (gt.y1 + gt.y2);
  int o = aidx * 5;
  out[o + 0] = (gcx - acx) / aw;
  out[o + 1] = (gcy - acy) / ah;
  out[o + 2] = logf(gw / aw);
  out[o + 3] = logf(gh / ah);
  out[o + 4] = best_iou >= 0.35f ? (float)(best + 1) : 0.0f;
}

int main(int argc, char **argv) {
  const int anchors = 64, gts = 5, total = anchors * 5;
  const char *path = (argc > 1) ? argv[1] : "output/output.txt";
  std::vector<float> out(total);
  float *dout; CK(cudaMalloc(&dout, total * sizeof(float)));
  int tpb = 128;
  encode<<<(anchors + tpb - 1) / tpb, tpb>>>(dout, anchors, gts);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(out.data(), dout, total * sizeof(float), cudaMemcpyDeviceToHost));
  cudaFree(dout);
  return write_output(path, out);
}
