# daliColorSpaceConversion

- Category: ai
- Operation: DALI RGB to YCbCr conversion
- Source project: NVIDIA/DALI
- Source URL: https://github.com/NVIDIA/DALI/blob/main/dali/operators/color_space/color_space_conversion.cu
- License: Apache-2.0
- Correctness method: CPU reference in `tests/verify.py`
- Tolerance: 1e-05
- CUDA features: 2D grid, channel reorder, color conversion
- Libraries: none
- Notes: Standalone RGB-to-YCbCr color-space conversion using DALI-style per-pixel channel math.
