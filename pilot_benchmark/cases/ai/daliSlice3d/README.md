# daliSlice3d

- Category: ai
- Operation: DALI batched 3D slice
- Source project: NVIDIA/DALI
- Source URL: https://github.com/NVIDIA/DALI/blob/main/dali/operators/crop/slice.cu
- License: Apache-2.0
- Correctness method: CPU reference in `tests/verify.py`
- Tolerance: 0.0
- CUDA features: 3D indexing, batched slice, strided tensor access
- Libraries: none
- Notes: Standalone 3D tensor slicing benchmark adapted from DALI Slice's generic N-D indexing workload.
