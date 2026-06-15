#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
  return 1; \
} } while (0)

#define OP 40
#define D0 1
#define D1 2
#define D2 8
#define D3 8
#define D4 16
#define D5 16
#define D6 1
#define D7 1
#define OUT_N 512

__host__ __device__ static inline float h01(unsigned i, unsigned s) {
  unsigned h = i * 2654435761u + s * 2246822519u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  return (float)(h & 0xFFFFFFu) / (float)0x1000000u;
}

__device__ static inline int clampi(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

__device__ static inline float img_hwc(int y, int x, int ch, int h, int w, int c, unsigned seed) {
  y = clampi(y, 0, h - 1); x = clampi(x, 0, w - 1);
  return h01((unsigned)((y * w + x) * c + ch), seed);
}

__device__ static inline float img_chw(int ch, int y, int x, int c, int h, int w, unsigned seed) {
  y = clampi(y, 0, h - 1); x = clampi(x, 0, w - 1);
  return h01((unsigned)((ch * h + y) * w + x), seed);
}

__device__ static inline float bilinear_hwc(float y, float x, int ch, int h, int w, int c, unsigned seed) {
  if (x < 0.0f || y < 0.0f || x > (float)(w - 1) || y > (float)(h - 1)) return 0.0f;
  int x0 = (int)floorf(x), y0 = (int)floorf(y);
  int x1 = clampi(x0 + 1, 0, w - 1), y1 = clampi(y0 + 1, 0, h - 1);
  float wx = x - x0, wy = y - y0;
  float v00 = img_hwc(y0, x0, ch, h, w, c, seed);
  float v01 = img_hwc(y0, x1, ch, h, w, c, seed);
  float v10 = img_hwc(y1, x0, ch, h, w, c, seed);
  float v11 = img_hwc(y1, x1, ch, h, w, c, seed);
  return (1.0f - wy) * ((1.0f - wx) * v00 + wx * v01) + wy * ((1.0f - wx) * v10 + wx * v11);
}

struct Box { float x1, y1, x2, y2, score; };

__device__ static inline Box make_box(int i, unsigned seed) {
  float cx = 0.12f + 0.76f * h01(i * 5u + 0u, seed);
  float cy = 0.12f + 0.76f * h01(i * 5u + 1u, seed);
  float ww = 0.10f + 0.25f * h01(i * 5u + 2u, seed);
  float hh = 0.10f + 0.25f * h01(i * 5u + 3u, seed);
  Box b;
  b.x1 = fmaxf(0.0f, cx - 0.5f * ww); b.y1 = fmaxf(0.0f, cy - 0.5f * hh);
  b.x2 = fminf(1.0f, cx + 0.5f * ww); b.y2 = fminf(1.0f, cy + 0.5f * hh);
  b.score = h01(i * 5u + 4u, seed);
  return b;
}

__device__ static inline float iou(Box a, Box b) {
  float ix1 = fmaxf(a.x1, b.x1), iy1 = fmaxf(a.y1, b.y1);
  float ix2 = fminf(a.x2, b.x2), iy2 = fminf(a.y2, b.y2);
  float iw = fmaxf(0.0f, ix2 - ix1), ih = fmaxf(0.0f, iy2 - iy1);
  float inter = iw * ih;
  float aa = fmaxf(0.0f, a.x2 - a.x1) * fmaxf(0.0f, a.y2 - a.y1);
  float ab = fmaxf(0.0f, b.x2 - b.x1) * fmaxf(0.0f, b.y2 - b.y1);
  return inter / (aa + ab - inter + 1e-6f);
}

__device__ static inline void point3(int i, unsigned seed, float &x, float &y, float &z) {
  x = h01(i * 3u + 0u, seed);
  y = h01(i * 3u + 1u, seed);
  z = h01(i * 3u + 2u, seed);
}

__device__ static inline float dist3(int a, unsigned sa, int b, unsigned sb) {
  float ax, ay, az, bx, by, bz;
  point3(a, sa, ax, ay, az); point3(b, sb, bx, by, bz);
  float dx = ax - bx, dy = ay - by, dz = az - bz;
  return sqrtf(dx * dx + dy * dy + dz * dz);
}

__device__ static inline void sort_small(float *v, int *idx, int n) {
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      if (v[j] < v[i]) {
        float tv = v[i]; v[i] = v[j]; v[j] = tv;
        int ti = idx[i]; idx[i] = idx[j]; idx[j] = ti;
      }
    }
  }
}

__global__ void compute(float *out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= OUT_N) return;
  float v = 0.0f;

  if (OP == 1) {
    int x = idx % D2, y = (idx / D2) % D1, ch = (idx / (D2 * D1)) % D3, n = idx / (D2 * D1 * D3);
    float means[3] = {0.45f, 0.50f, 0.55f};
    float stds[3] = {0.20f, 0.25f, 0.30f};
    v = (h01((unsigned)(((n * D1 + y) * D2 + x) * D3 + ch), 123) - means[ch]) / stds[ch];
  } else if (OP == 2) {
    int ch = idx % D3, ox = (idx / D3) % D5, oy = (idx / (D3 * D5)) % D4;
    int mx = D5 - 1 - ox;
    float sy = ((float)(oy + 2) + 0.5f) * D1 / (D4 + 4) - 0.5f;
    float sx = ((float)(mx + 3) + 0.5f) * D2 / (D5 + 6) - 0.5f;
    v = bilinear_hwc(sy, sx, ch, D1, D2, D3, 123);
  } else if (OP == 3) {
    int ch = idx % D3, ox = (idx / D3) % D5, oy = (idx / (D3 * D5)) % D4;
    float sy = 4.0f + ((float)oy + 0.5f) * 16.0f / D4 - 0.5f;
    float sx = 5.0f + ((float)ox + 0.5f) * 18.0f / D5 - 0.5f;
    v = bilinear_hwc(sy, sx, ch, D1, D2, D3, 123);
  } else if (OP == 4) {
    int ch = idx % D2, x = (idx / D2) % D1, y = idx / (D2 * D1);
    int y0 = 7, x0 = 5;
    v = h01((unsigned)((y * D1 + x) * D2 + ch), 123) * 0.25f;
    if (y >= y0 && y < y0 + D3 && x >= x0 && x < x0 + D4) {
      v = 0.75f * h01((unsigned)(((y - y0) * D4 + (x - x0)) * D2 + ch), 321) + 0.25f * v;
    }
  } else if (OP == 5) {
    int x = idx % D3, y = (idx / D3) % D2, c = (idx / (D3 * D2)) % D1, n = idx / (D3 * D2 * D1);
    v = h01((unsigned)(((n * D1 + c) * D2 + y) * 8 + x), 123);
  } else if (OP == 6) {
    int ch = idx % D3, x = (idx / D3) % D2, y = (idx / (D3 * D2)) % D1, n = idx / (D3 * D2 * D1);
    int sx = D2 - 1 - x, sy = (n == 1) ? (D1 - 1 - y) : y;
    v = h01((unsigned)(((n * D1 + sy) * D2 + sx) * D3 + ch), 123);
  } else if (OP == 7) {
    int coord = idx % 4, b = idx / 4;
    Box box = make_box(b, 123);
    if (coord == 0) v = 1.0f - box.x2;
    else if (coord == 1) v = box.y1;
    else if (coord == 2) v = 1.0f - box.x1;
    else v = box.y2;
  } else if (OP == 8) {
    int f = idx % D2, s = idx / D2;
    int pos = (2 * s + 1) % D1;
    v = h01((unsigned)((s * D1 + pos) * D2 + f), 123);
  } else if (OP == 9) {
    float x = h01(idx, 123) * 300.0f - 20.0f;
    int q = (int)floorf(x + 0.5f);
    q = clampi(q, 0, 255);
    v = (float)q;
  } else if (OP == 10) {
    int ch = idx % D3, x = (idx / D3) % D2, y = (idx / (D3 * D2)) % D1;
    float sx = x + 1.75f * sinf(0.35f * y);
    float sy = y + 1.25f * sinf(0.27f * x);
    v = bilinear_hwc(sy, sx, ch, D1, D2, D3, 123);
  } else if (OP == 11) {
    int bin = idx, count = 0;
    for (int i = 0; i < D0; ++i) {
      int b = clampi((int)floorf(h01(i, 123) * D1), 0, D1 - 1);
      if (b == bin) ++count;
    }
    v = (float)count;
  } else if (OP == 12) {
    int x = idx % D1, y = idx / D1, bins = D2, tile = 8;
    int y0 = (y / tile) * tile, x0 = (x / tile) * tile;
    int b = clampi((int)floorf(h01(y * D1 + x, 123) * bins), 0, bins - 1);
    int cdf = 0, total = 0;
    for (int yy = y0; yy < min(y0 + tile, D0); ++yy) for (int xx = x0; xx < min(x0 + tile, D1); ++xx) {
      int bb = clampi((int)floorf(h01(yy * D1 + xx, 123) * bins), 0, bins - 1);
      if (bb <= b) ++cdf; ++total;
    }
    v = (float)cdf / (float)total;
  } else if (OP == 13) {
    int x = idx % D1, y = idx / D1;
    float a[9]; int k = 0;
    for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) a[k++] = h01((unsigned)(clampi(y + dy, 0, D0 - 1) * D1 + clampi(x + dx, 0, D1 - 1)), 123);
    for (int i = 0; i < 9; ++i) for (int j = i + 1; j < 9; ++j) if (a[j] < a[i]) { float t = a[i]; a[i] = a[j]; a[j] = t; }
    v = a[4];
  } else if (OP == 14) {
    int x = idx % D1, y = idx / D1;
    float den = 0.0015f * x - 0.0010f * y + 1.0f;
    float sx = (0.94f * x + 0.11f * y - 1.7f) / den;
    float sy = (-0.07f * x + 1.03f * y + 1.2f) / den;
    v = bilinear_hwc(sy, sx, 0, D0, D1, 1, 123);
  } else if (OP == 15) {
    int x = idx % D1, y = idx / D1;
    float r = h01((unsigned)((y * D1 + x) * 3 + 0), 123), g = h01((unsigned)((y * D1 + x) * 3 + 1), 123), b = h01((unsigned)((y * D1 + x) * 3 + 2), 123);
    v = 0.299f * r + 0.587f * g + 0.114f * b;
  } else if (OP >= 16 && OP <= 19) {
    int x = idx % D1, y = idx / D1;
    float kw[3] = {0.25f, 0.5f, 0.25f};
    if (OP == 16) {
      float k2[9] = {0.0625f,0.125f,0.0625f,0.125f,0.25f,0.125f,0.0625f,0.125f,0.0625f};
      int p = 0; for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) v += k2[p++] * h01((unsigned)(clampi(y + dy, 0, D0 - 1) * D1 + clampi(x + dx, 0, D1 - 1)), 123);
    } else if (OP == 17) {
      for (int dx = -1; dx <= 1; ++dx) v += kw[dx + 1] * h01((unsigned)(y * D1 + clampi(x + dx, 0, D1 - 1)), 123);
    } else if (OP == 18) {
      for (int dy = -1; dy <= 1; ++dy) v += kw[dy + 1] * h01((unsigned)(clampi(y + dy, 0, D0 - 1) * D1 + x), 123);
    } else {
      for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) v += kw[dy + 1] * kw[dx + 1] * h01((unsigned)(clampi(y + dy, 0, D0 - 1) * D1 + clampi(x + dx, 0, D1 - 1)), 123);
    }
  } else if (OP == 20) {
    int x = idx % D1, y = idx / D1;
    v = 1.0f;
    for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) v = fminf(v, h01((unsigned)(clampi(y + dy, 0, D0 - 1) * D1 + clampi(x + dx, 0, D1 - 1)), 123));
  } else if (OP == 21) {
    int ox = idx % D3, oy = idx / D3;
    float acc = 0.0f;
    for (int yy = 0; yy < 2; ++yy) for (int xx = 0; xx < 2; ++xx) acc += h01((unsigned)((oy * 2 + yy) * D1 + (ox * 2 + xx)), 123);
    v = acc * 0.25f;
  } else if (OP == 22) {
    int y = idx; for (int x = 0; x < D1; ++x) v += h01((unsigned)(y * D1 + x), 123);
  } else if (OP == 23) {
    float mn = 1.0f, mx = 0.0f;
    for (int i = 0; i < D0; ++i) { float x = h01(i, 123); mn = fminf(mn, x); mx = fmaxf(mx, x); }
    v = (h01(idx, 123) - mn) / (mx - mn + 1e-6f);
  } else if (OP == 24) {
    v = 0.65f * h01(idx, 123) + 0.35f * h01(idx, 321) + 0.1f;
  } else if (OP == 25) {
    int a = (int)floorf(h01(idx, 123) * 256.0f), b = (int)floorf(h01(idx, 321) * 256.0f);
    v = (float)(a & b);
  } else if (OP == 26) {
    v = h01(idx, 123) > h01(idx, 321) ? 1.0f : 0.0f;
  } else if (OP == 27) {
    int b = idx % D1, a = idx / D1;
    v = iou(make_box(a, 123), make_box(b, 321));
  } else if (OP == 28) {
    Box bi = make_box(idx, 123);
    int keep = 1;
    for (int j = 0; j < D0; ++j) {
      Box bj = make_box(j, 123);
      if (bj.score > bi.score && iou(bi, bj) > 0.35f) keep = 0;
    }
    v = (float)keep;
  } else if (OP == 29 || OP == 30) {
    int pw = D6, ph = D5, rois = D4, c = D1, h = D2, w = D3;
    int px = idx % pw, py = (idx / pw) % ph, ch = (idx / (pw * ph)) % c, r = idx / (pw * ph * c);
    Box box = make_box(r, 555);
    float x1 = box.x1 * (w - 1), y1 = box.y1 * (h - 1), x2 = box.x2 * (w - 1), y2 = box.y2 * (h - 1);
    if (OP == 29) {
      float sy = y1 + ((float)py + 0.5f) * (y2 - y1) / ph;
      float sx = x1 + ((float)px + 0.5f) * (x2 - x1) / pw;
      v = bilinear_hwc(sy, sx, ch, h, w, c, 123);
    } else {
      int yy0 = (int)floorf(y1 + (float)py * (y2 - y1) / ph), yy1 = (int)ceilf(y1 + (float)(py + 1) * (y2 - y1) / ph);
      int xx0 = (int)floorf(x1 + (float)px * (x2 - x1) / pw), xx1 = (int)ceilf(x1 + (float)(px + 1) * (x2 - x1) / pw);
      v = -1.0f;
      for (int yy = yy0; yy <= yy1; ++yy) for (int xx = xx0; xx <= xx1; ++xx) v = fmaxf(v, img_hwc(yy, xx, ch, h, w, c, 123));
    }
  } else if (OP == 31 || OP == 32) {
    int k = (OP == 31) ? 3 : D2, q = idx / k, rank = idx % k;
    float ds[16]; int ids[16];
    for (int i = 0; i < D1; ++i) { ds[i] = dist3(q, 123, i, 321); ids[i] = i; }
    sort_small(ds, ids, D1);
    v = ds[rank];
  } else if (OP == 33) {
    int m = idx % D3, ch = (idx / D3) % D1, b = idx / (D3 * D1);
    int src = (m * 3 + b) % D2;
    v = h01((unsigned)((b * D1 + ch) * D2 + src), 123);
  } else if (OP == 34) {
    int k = idx % D3, m = (idx / D3) % D2, ch = idx / (D3 * D2);
    int src = (m * 2 + k * 3) % D1;
    v = h01((unsigned)(ch * D1 + src), 123);
  } else if (OP == 35) {
    int q = idx % D2, ch = idx / D2;
    int i0 = q % D1, i1 = (q * 3 + 1) % D1, i2 = (q * 5 + 2) % D1;
    v = 0.5f * h01((unsigned)(ch * D1 + i0), 123) + 0.3f * h01((unsigned)(ch * D1 + i1), 123) + 0.2f * h01((unsigned)(ch * D1 + i2), 123);
  } else if (OP == 36) {
    int selected[16]; selected[0] = 0;
    for (int s = 1; s <= idx; ++s) {
      float best = -1.0f; int best_i = 0;
      for (int p = 0; p < D0; ++p) {
        float nearest = 1e30f;
        for (int j = 0; j < s; ++j) nearest = fminf(nearest, dist3(p, 123, selected[j], 123));
        if (nearest > best) { best = nearest; best_i = p; }
      }
      selected[s] = best_i;
    }
    v = (float)selected[idx];
  } else if (OP == 37) {
    int k = idx % D2, q = idx / D2, found = 0, ans = -1;
    for (int p = 0; p < D1; ++p) {
      if (dist3(q, 123, p, 321) < 0.55f) {
        if (found == k) ans = p;
        ++found;
      }
    }
    v = (float)ans;
  } else if (OP == 38) {
    int b = idx % D1, p = idx / D1;
    float x, y, z; point3(p, 123, x, y, z);
    Box box = make_box(b, 321);
    v = (x >= box.x1 && x <= box.x2 && y >= box.y1 && y <= box.y2 && z >= 0.2f && z <= 0.8f) ? 1.0f : 0.0f;
  } else if (OP == 39) {
    int ow = D3 / 2, oh = D2 / 2;
    int ox = idx % ow, oy = (idx / ow) % oh, ch = (idx / (ow * oh)) % D1, n = idx / (ow * oh * D1);
    v = -1.0f;
    for (int yy = 0; yy < 2; ++yy) for (int xx = 0; xx < 2; ++xx) v = fmaxf(v, img_chw(ch, oy * 2 + yy, ox * 2 + xx, D1, D2, D3, 123 + n));
  } else if (OP == 40) {
    int ox = idx % D5, oy = (idx / D5) % D4, ch = (idx / (D5 * D4)) % D1, n = idx / (D5 * D4 * D1);
    int iy = clampi((int)floorf((float)oy * D2 / D4), 0, D2 - 1);
    int ix = clampi((int)floorf((float)ox * D3 / D5), 0, D3 - 1);
    v = img_chw(ch, iy, ix, D1, D2, D3, 123 + n);
  }
  out[idx] = v;
}

int main(int argc, char **argv) {
  const char *path = (argc > 1) ? argv[1] : "output/output.txt";
  std::vector<float> host(OUT_N);
  float *dout = nullptr;
  CK(cudaMalloc(&dout, OUT_N * sizeof(float)));
  int tpb = 128;
  compute<<<(OUT_N + tpb - 1) / tpb, tpb>>>(dout);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(host.data(), dout, OUT_N * sizeof(float), cudaMemcpyDeviceToHost));
  cudaFree(dout);
  FILE *f = fopen(path, "w");
  if (!f) { fprintf(stderr, "open %s\n", path); return 1; }
  for (int i = 0; i < OUT_N; ++i) fprintf(f, "%.9g\n", host[i]);
  fclose(f);
  return 0;
}
