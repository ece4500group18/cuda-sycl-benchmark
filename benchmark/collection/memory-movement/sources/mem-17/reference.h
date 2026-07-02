// reference.h
//
// CPU reference implementation for the textureInterpolationGather case
// (image rotation via bilinear-interpolated, wrap-addressed gather).
#ifndef TEXTUREINTERPOLATIONGATHER_REFERENCE_H
#define TEXTUREINTERPOLATIONGATHER_REFERENCE_H

#include <cmath>

// Deterministic input generator (width x height row-major image):
//   img[y*width+x] = ((x*3 + y) % 23) - 11
inline float gen_img(int x, int y) {
  return (float)(((x * 3 + y) % 23) - 11);
}

// Reference rotate + bilinear + wrap-addressing gather, computed entirely
// in double precision from the source image `img` (width*height floats,
// row-major). This mirrors, step for step, both the texture-object path
// (transformKernel's tex2D<float> fetch with normalizedCoords = true,
// filterMode = cudaFilterModeLinear, addressMode = cudaAddressModeWrap)
// and the plain-global-memory path (softwareGatherKernel) in main.cu --
// it is the common "ground truth" both GPU kernels are compared against.
inline void reference_rotate_bilinear_wrap(const float *img, double *out,
                                            int width, int height,
                                            double theta) {
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      double u = (double)x - (double)width / 2.0;
      double v = (double)y - (double)height / 2.0;
      double tu = u * std::cos(theta) - v * std::sin(theta);
      double tv = v * std::cos(theta) + u * std::sin(theta);

      tu /= (double)width;
      tv /= (double)height;

      double un = tu + 0.5;
      double vn = tv + 0.5;

      // cudaAddressModeWrap on normalized coordinates: wrap into [0, 1).
      un = un - std::floor(un);
      vn = vn - std::floor(vn);

      // Un-normalize to texel space (texel i's center sits at i + 0.5).
      double fx = un * (double)width - 0.5;
      double fy = vn * (double)height - 0.5;

      long x0 = (long)std::floor(fx);
      long y0 = (long)std::floor(fy);
      double ax = fx - (double)x0;
      double ay = fy - (double)y0;

      long x1 = x0 + 1;
      long y1 = y0 + 1;

      // Wrap texel indices into [0, width) / [0, height).
      x0 = ((x0 % width) + width) % width;
      x1 = ((x1 % width) + width) % width;
      y0 = ((y0 % height) + height) % height;
      y1 = ((y1 % height) + height) % height;

      double p00 = img[y0 * width + x0];
      double p10 = img[y0 * width + x1];
      double p01 = img[y1 * width + x0];
      double p11 = img[y1 * width + x1];

      double top = p00 + ax * (p10 - p00);
      double bot = p01 + ax * (p11 - p01);
      out[y * width + x] = top + ay * (bot - top);
    }
  }
}

#endif  // TEXTUREINTERPOLATIONGATHER_REFERENCE_H
