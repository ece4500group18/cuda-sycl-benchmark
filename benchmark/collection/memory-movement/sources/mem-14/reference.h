// reference.h
//
// CPU reference implementation for the separableConvHaloTiling case
// (separable 2D convolution: horizontal pass then vertical pass, each a
// 1D convolution with a fixed radius-4, 9-tap kernel).
//
// The accumulation loop below is ported directly from the algorithm shape
// of the original sample's own host reference
// (NVIDIA/cuda-samples cpp/2_Concepts_and_Techniques/convolutionSeparable/
// convolutionSeparable_gold.cpp: convolutionRowCPU / convolutionColumnCPU),
// which is itself: `sum = 0; for (k = -R; k <= R; k++) { d = x + k; if (d in
// bounds) sum += Src[d] * Kernel[R - k]; }`. Both GPU kernels in main.cu
// (the shared-memory tiled kernel reproduced verbatim from that sample, and
// the naive global-memory kernel written for this repo) use that exact same
// `sum = 0; for k = -R..R: sum += Kernel[R-k] * pixel;` accumulation, in the
// same left-to-right order, with the same "out of image bounds -> treat the
// missing pixel as 0" boundary rule -- so this reference produces the same
// sequence of floating-point operations as both GPU paths.
#ifndef SEPARABLECONVHALOTILING_REFERENCE_H
#define SEPARABLECONVHALOTILING_REFERENCE_H

#define REF_KERNEL_RADIUS 4
#define REF_KERNEL_LENGTH (2 * REF_KERNEL_RADIUS + 1)

// Deterministic input generators:
//   image[y*imageW + x] = ((x + y) % 13) - 6              (integers in [-6, 6])
//   kernel[i]           = ((i % 5) - 2) * 0.25             (multiples of 0.25 in [-0.5, 0.5])
//
// Both are chosen so every partial product and partial sum that occurs
// anywhere in the row/column convolution is an exact dyadic value (small
// integer times a power-of-two fraction, with a running sum whose
// magnitude never exceeds a few dozen) that IEEE-754 single precision
// represents with zero rounding error. That is what lets this reference,
// the shared-memory tiled GPU kernel, and the naive global-memory GPU
// kernel agree exactly (max_abs_error == 0): with no rounding error at any
// step, fused-multiply-add vs. separate multiply-then-add and CPU vs. GPU
// arithmetic units all collapse to the same mathematical (and bit-exact
// IEEE-754) result. The original upstream sample uses a non-power-of-two
// step for its random kernel/image values and therefore only checks
// against an epsilon; using a dyadic-safe step here is what upgrades that
// to a documented exact match.
inline double gen_image(int x, int y) {
  return (double)(((x + y) % 13) - 6);
}

inline double gen_kernel(int i) {
  return (double)((i % 5) - 2) * 0.25;
}

// Horizontal (row) pass: out[y*imageW+x] = sum_{k=-R..R} kernel[R-k] * in[y*imageW + (x+k)],
// treating any (x+k) outside [0, imageW) as 0.
inline void reference_convolve_rows(const double *kernel, const double *src,
                                     double *dst, int imageW, int imageH) {
  for (int y = 0; y < imageH; ++y) {
    for (int x = 0; x < imageW; ++x) {
      double sum = 0.0;
      for (int k = -REF_KERNEL_RADIUS; k <= REF_KERNEL_RADIUS; ++k) {
        int d = x + k;
        if (d >= 0 && d < imageW) {
          sum += kernel[REF_KERNEL_RADIUS - k] * src[y * imageW + d];
        }
      }
      dst[y * imageW + x] = sum;
    }
  }
}

// Vertical (column) pass: out[y*imageW+x] = sum_{k=-R..R} kernel[R-k] * in[(y+k)*imageW + x],
// treating any (y+k) outside [0, imageH) as 0.
inline void reference_convolve_columns(const double *kernel, const double *src,
                                        double *dst, int imageW, int imageH) {
  for (int y = 0; y < imageH; ++y) {
    for (int x = 0; x < imageW; ++x) {
      double sum = 0.0;
      for (int k = -REF_KERNEL_RADIUS; k <= REF_KERNEL_RADIUS; ++k) {
        int d = y + k;
        if (d >= 0 && d < imageH) {
          sum += kernel[REF_KERNEL_RADIUS - k] * src[d * imageW + x];
        }
      }
      dst[y * imageW + x] = sum;
    }
  }
}

#endif  // SEPARABLECONVHALOTILING_REFERENCE_H
