# daliWarpAffine

- Category: ai
- Operation: DALI affine image warp
- Source project: NVIDIA/DALI
- Source URL: https://github.com/NVIDIA/DALI/blob/main/dali/operators/displacement/warpaffine.cu
- License: Apache-2.0
- Correctness method: CPU reference in `tests/verify.py`
- Tolerance: 0.0001
- CUDA features: 2D grid, affine transform, bilinear interpolation
- Libraries: none
- Notes: Standalone inverse affine image warp using DALI-style coordinate transform and border behavior.
