// reference.h
//
// CPU reference implementation for the layeredTextureGather case.
//
// Mirrors the original NVIDIA sample's own "expected output" computation
// (simpleLayeredTexture.cu builds h_data_ref on the host as
// `-h_data[layer*width*height+i] + layer` before ever touching the GPU) --
// we reuse that exact formula as the correctness oracle.
#ifndef LAYEREDTEXTUREGATHER_REFERENCE_H
#define LAYEREDTEXTUREGATHER_REFERENCE_H

// Deterministic input generator: data value at flat in-layer index i
// (0 <= i < width*height), same formula replicated identically into every
// layer, exactly like the original sample's
// `h_data[layer*width*height+i] = (float)i` but bounded/signed so it
// exercises negative and positive values:
//   data[i] = (i % 19) - 9
inline float gen_data(int i) {
  return (float)((i % 19) - 9);
}

// Reference transform, matching transformKernel's
// `-tex2DLayered(tex,u,v,layer) + layer` exactly (since the texture is
// sampled with point/nearest filtering at exact texel centers, the texture
// fetch returns precisely gen_data(i) with zero interpolation error):
//   out[layer*width*height + y*width + x] = -data[y*width+x] + layer
inline void reference_layered_transform(float *out, int width, int height, int num_layers) {
  for (int layer = 0; layer < num_layers; ++layer) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        int i = y * width + x;
        out[layer * width * height + i] = -gen_data(i) + (float)layer;
      }
    }
  }
}

#endif  // LAYEREDTEXTUREGATHER_REFERENCE_H
