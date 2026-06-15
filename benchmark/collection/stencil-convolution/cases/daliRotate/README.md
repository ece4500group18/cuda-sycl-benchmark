# daliRotate

- Category: ai
- Operation: DALI bilinear image rotation
- Source project: NVIDIA/DALI
- Source URL: https://github.com/NVIDIA/DALI/blob/main/dali/operators/displacement/rotate.cu
- License: Apache-2.0
- Correctness method: CPU reference in `tests/verify.py`
- Tolerance: 0.0001
- CUDA features: 2D grid, bilinear interpolation, border handling
- Libraries: none
- Notes: Standalone center rotation with bilinear sampling, adapted from DALI displacement/rotate coordinate mapping.
