# daliColorTwist

- Category: ai
- Operation: DALI ColorTwist RGB transform
- Source project: NVIDIA/DALI
- Source URL: https://github.com/NVIDIA/DALI/blob/main/dali/operators/color/color_twist.cu
- License: Apache-2.0
- Correctness method: CPU reference in `tests/verify.py`
- Tolerance: 1e-05
- CUDA features: 2D grid, per-channel arithmetic, color transform
- Libraries: none
- Notes: Standalone RGB brightness/contrast/saturation/hue transform adapted from the DALI ColorTwist operator shape.
