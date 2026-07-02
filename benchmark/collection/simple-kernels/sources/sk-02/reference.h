// reference.h
//
// CPU reference implementation for the backpropTreeReduction case
// (backprop layer-forward weighted partial sum, power-of-two-stride tree
// reduction over HEIGHT=16 input units per hidden unit).
#ifndef BACKPROPTREEREDUCTION_REFERENCE_H
#define BACKPROPTREEREDUCTION_REFERENCE_H

// Must match main.cu's #define WIDTH/HEIGHT (from Rodinia's backprop.h).
#ifndef WIDTH
#define WIDTH 16
#endif
#ifndef HEIGHT
#define HEIGHT 16
#endif

// Deterministic input generators:
//   input_node[i]       = ((i % 7) - 3) * 0.5,       i in [0, in]
//   weight_matrix[i][j] = ((i + j) % 5 - 2) * 0.1,   i in [0, in], j in [0, hid]
inline float gen_input(int i) {
  return (float)((i % 7) - 3) * 0.5f;
}

inline float gen_weight(int i, int j) {
  return (float)(((i + j) % 5) - 2) * 0.1f;
}

// CPU reference for bpnn_layerforward_CUDA's `hidden_partial_sum` output.
//
// For each input-block `by` (HEIGHT=16 input units) and each hidden unit
// `tx` (0 <= tx < hid, hid must equal WIDTH, exactly as the GPU kernel
// requires), this reproduces:
//   1. col[ty] = weight[row][tx+1] * input[row], row = HEIGHT*by + ty + 1,
//      for ty in [0, HEIGHT) -- identical to the kernel's
//      `weight_matrix[ty][tx] = input_hidden_cuda[index] * input_node[ty]`.
//   2. The *same* power-of-two-stride tree reduction the kernel performs
//      (stride = 2, 4, 8, ..., HEIGHT; at each stride, slot `ty` for which
//      `ty % stride == 0` adds slot `ty + stride/2`), all in `float`
//      arithmetic and in the same step order, so the sequence of additions
//      -- and therefore the final rounding -- is bit-for-bit identical to
//      the GPU kernel's. Reads at each stride only ever touch slots that
//      are *not* written during that same stride (since `ty + stride/2` is
//      never itself `== 0 (mod stride)` for `0 < stride/2 < stride`), so
//      processing `ty` sequentially within a stride (rather than "in
//      parallel" as the GPU does) does not change any value that step
//      reads -- the two are computationally equivalent, not just
//      numerically close.
//   3. col[0] after all strides is the reduced weighted partial sum for
//      (by, tx), written to hidden_partial_sum_ref[by*hid + tx] (matching
//      the kernel's `hidden_partial_sum[by * hid + ty] = weight_matrix[tx][ty]`
//      under the `tx == 0` guard, which is equivalent because HEIGHT ==
//      WIDTH here so the swapped tx/ty roles in that final store line up).
//
// Because every step is the exact same sequence of float multiplies and
// additions as the kernel, in the exact same order, this is an EXACT
// (bit-for-bit) match oracle: max_abs_error == 0 is expected, not a
// tolerance-bounded approximation.
inline void reference_layerforward(const float *input_cuda,
                                    const float *input_hidden_cuda,
                                    float *hidden_partial_sum_ref,
                                    int in, int hid) {
  const int num_blocks = in / HEIGHT;
  for (int by = 0; by < num_blocks; ++by) {
    for (int tx = 0; tx < hid; ++tx) {
      float col[HEIGHT];
      for (int ty = 0; ty < HEIGHT; ++ty) {
        int row = HEIGHT * by + ty + 1;
        float input_node = input_cuda[row];
        float w = input_hidden_cuda[row * (hid + 1) + (tx + 1)];
        col[ty] = w * input_node;
      }
      // Tree reduction: stride = 2, 4, 8, ..., HEIGHT (matches the
      // kernel's __log2f(HEIGHT)/__powf(2,i) strides, which are exact
      // powers of two for HEIGHT == 16).
      for (int stride = 2; stride <= HEIGHT; stride *= 2) {
        for (int ty = 0; ty < HEIGHT; ++ty) {
          if (ty % stride == 0) {
            col[ty] = col[ty] + col[ty + stride / 2];
          }
        }
      }
      hidden_partial_sum_ref[by * hid + tx] = col[0];
    }
  }
}

#endif  // BACKPROPTREEREDUCTION_REFERENCE_H
