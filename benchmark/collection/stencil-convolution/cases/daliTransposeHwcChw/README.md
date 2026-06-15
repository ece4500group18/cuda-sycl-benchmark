# daliTransposeHwcChw

- Category: ai
- Operation: DALI HWC to CHW transpose
- Source project: NVIDIA/DALI
- Source URL: https://github.com/NVIDIA/DALI/blob/main/dali/operators/transpose/transpose.cu
- License: Apache-2.0
- Correctness method: CPU reference in `tests/verify.py`
- Tolerance: 0.0
- CUDA features: layout transform, strided memory access, batched tensor indexing
- Libraries: none
- Notes: Standalone HWC-to-CHW batched layout transform adapted from DALI Transpose.
