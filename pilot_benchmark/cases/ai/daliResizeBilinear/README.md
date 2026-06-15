# daliResizeBilinear

- Category: ai
- Operation: DALI bilinear resize
- Source project: NVIDIA/DALI
- Source URL: https://github.com/NVIDIA/DALI/blob/main/dali/operators/resize/resize.cu
- License: Apache-2.0
- Correctness method: CPU reference in `tests/verify.py`
- Tolerance: 0.0001
- CUDA features: 2D grid, bilinear interpolation, batched image indexing
- Libraries: none
- Notes: Standalone half-pixel bilinear resize benchmark adapted from DALI Resize's image resampling workload.
