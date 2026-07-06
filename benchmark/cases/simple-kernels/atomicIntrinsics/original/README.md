# Original CUDA

Standalone Stage 1 CUDA case for `atomicIntrinsics`.

The `testKernel` is reproduced verbatim from NVIDIA/cuda-samples
`simpleAtomicIntrinsics_kernel.cuh` (BSD-3-Clause). The host driver is a
minimal deterministic launcher that writes the 11 final slot values to
`output/cuda_output.txt`; correctness is checked by `../tests/verify.py`.
