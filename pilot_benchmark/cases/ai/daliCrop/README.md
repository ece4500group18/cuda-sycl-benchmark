# daliCrop

- Category: ai
- Operation: DALI batched HWC crop
- Source project: NVIDIA/DALI
- Source URL: https://github.com/NVIDIA/DALI/blob/main/dali/operators/crop/crop.cu
- License: Apache-2.0
- Correctness method: CPU reference in `tests/verify.py`
- Tolerance: 0.0
- CUDA features: 2D grid, batched indexing, strided image access
- Libraries: none
- Notes: Standalone batched crop with per-sample anchors, adapted from the indexing pattern of DALI Crop.
