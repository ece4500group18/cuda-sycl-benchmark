# daliCropMirrorNormalize

- Category: ai
- Operation: DALI crop mirror normalize
- Source project: NVIDIA/DALI
- Source URL: https://github.com/NVIDIA/DALI/blob/main/dali/operators/fused/crop_mirror_normalize.cu
- License: Apache-2.0
- Correctness method: CPU reference in `tests/verify.py`
- Tolerance: 1e-05
- CUDA features: fused preprocessing, layout conversion, branching
- Libraries: none
- Notes: Standalone fused crop, optional mirror, per-channel normalize, and HWC-to-CHW layout conversion.
