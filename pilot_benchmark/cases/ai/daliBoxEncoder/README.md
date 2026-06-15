# daliBoxEncoder

- Category: ai
- Operation: DALI detection box encoder
- Source project: NVIDIA/DALI
- Source URL: https://github.com/NVIDIA/DALI/blob/main/dali/operators/detection/box_encoder.cu
- License: Apache-2.0
- Correctness method: CPU reference in `tests/verify.py`
- Tolerance: 0.0001
- CUDA features: bbox IoU, branching, per-anchor reduction
- Libraries: none
- Notes: Standalone anchor-to-ground-truth matcher and box encoder modeled after DALI BoxEncoder.
