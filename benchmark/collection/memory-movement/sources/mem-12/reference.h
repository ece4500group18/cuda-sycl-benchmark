// reference.h
//
// CPU reference implementation for the pitchLinearVsCudaArrayTexture case
// (periodic/wrap circular shift of a WxH float grid, read via a
// point-filtered, wrap-addressed texture).
#ifndef PITCHLINEARVSCUDAARRAYTEXTURE_REFERENCE_H
#define PITCHLINEARVSCUDAARRAYTEXTURE_REFERENCE_H

// Deterministic input generator:
//   src[y*w + x] = ((x*3 + y*5) % 17) - 8
inline float gen_src(int x, int y) {
  return (float)(((x * 3 + y * 5) % 17) - 8);
}

// CPU reference: same periodic (wrap) shift the two GPU kernels compute via
// a texture fetch with cudaAddressModeWrap --
//   gold[j*w + i] = src[((j+shiftY) % h) * w + ((i+shiftX) % w)]
// This mirrors the original NVIDIA sample's own host-side gold computation
// exactly (see simplePitchLinearTexture.cu's runTest()), just generating
// `src` from gen_src() instead of reading it back from a device buffer.
inline void reference_shift(float *gold, int w, int h, int shiftX, int shiftY) {
  for (int j = 0; j < h; ++j) {
    int jshift = (j + shiftY) % h;
    for (int i = 0; i < w; ++i) {
      int ishift = (i + shiftX) % w;
      gold[j * w + i] = gen_src(ishift, jshift);
    }
  }
}

#endif  // PITCHLINEARVSCUDAARRAYTEXTURE_REFERENCE_H
