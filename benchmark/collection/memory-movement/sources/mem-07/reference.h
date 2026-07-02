// reference.h
//
// CPU reference implementation for the cubemapTextureGather case.
//
// Mirrors the original NVIDIA sample's own "expected output" computation
// (simpleCubemapTexture.cu builds h_data_ref on the host as
// `-h_data[layer*cubemap_size+i] + layer` before ever touching the GPU,
// with num_layers=1 so layer is always 0) -- we reuse that exact formula
// (with the `+layer` term dropped since this case has a single layer) as
// the correctness oracle.
#ifndef CUBEMAPTEXTUREGATHER_REFERENCE_H
#define CUBEMAPTEXTUREGATHER_REFERENCE_H

// Deterministic input generator: data value at flat, face-major index i
// (0 <= i < width*width*6, i.e. face*width*width + y*width + x), exactly
// like the original sample's `h_data[i] = (float)i` but bounded/signed so
// it exercises negative and positive values:
//   data[i] = (i % 29) - 14
inline float gen_data(int i) {
  return (float)((i % 29) - 14);
}

// Reference transform, matching transformKernel's
// `-texCubemap<float>(tex, cx, cy, cz)` exactly (since the cubemap texture
// is sampled with point/nearest filtering at exact face-texel centers, the
// fetch returns precisely gen_data(i) with zero interpolation error):
//   out[face*width*width + y*width + x] = -data[face*width*width + y*width + x]
inline void reference_cubemap_transform(float *out, int width) {
  const int cubemap_size = width * width * 6;
  for (int i = 0; i < cubemap_size; ++i) {
    out[i] = -gen_data(i);
  }
}

#endif  // CUBEMAPTEXTUREGATHER_REFERENCE_H
