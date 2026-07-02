// reference.h
//
// CPU reference implementation for the templatedSharedMemIdiom case.
//
// Deterministic input generator, matching the original sample's own
// `h_idata[i] = (T)i;` (NVIDIA/cuda-samples
// cpp/0_Introduction/simpleTemplates/simpleTemplates.cu, runTest<T>()):
//   g_idata[i] = (T)i
#ifndef TEMPLATEDSHAREDMEMIDIOM_REFERENCE_H
#define TEMPLATEDSHAREDMEMIDIOM_REFERENCE_H

template <typename T>
inline T gen_idata(unsigned int i) {
  return (T)i;
}

// Ported directly from the original sample's own computeGold<T>()
// (simpleTemplates.cu):
//
//   template <class T> void computeGold(T *reference, T *idata, const
//   unsigned int len) {
//     const T T_len = static_cast<T>(len);
//     for (unsigned int i = 0; i < len; ++i) {
//       reference[i] = idata[i] * T_len;
//     }
//   }
//
// This is exactly what testKernel<T> computes in main.cu: each output
// element is produced independently as `idata[i] * (T)len` (`len` here
// is `num_threads` = `blockDim.x`, the same value on host and device).
// There is no cross-thread accumulation/reduction, so the result is
// exact for T = int (plain integer multiply, no overflow at N = 256:
// max value is 255 * 256 = 65280) and exact for T = float as well
// (multiplying by 256.0f, an exact power of two, is exact in IEEE-754
// binary32 for any value whose exponent has headroom -- true here since
// both idata[i] and the product easily fit in the 24-bit mantissa).
template <typename T>
inline void reference_transform(const T *idata, T *ref_out, unsigned int len) {
  const T T_len = (T)len;
  for (unsigned int i = 0; i < len; ++i) {
    ref_out[i] = idata[i] * T_len;
  }
}

#endif  // TEMPLATEDSHAREDMEMIDIOM_REFERENCE_H
