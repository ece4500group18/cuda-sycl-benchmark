# cuTENSOR elementwise permutation (NHWC->NCHW)

Tensor layout permutation C_cwhn = A_whcn via the cuTENSOR plan-based API.

Source project: NVIDIA/CUDALibrarySamples

Source URL: https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuTENSOR/elementwise_permute.cu

License: BSD-3-Clause

Extraction fidelity: extracted

Extraction notes: cuTENSOR 2.x plan API verbatim with CreatePermutation and cutensorPermute (no workspace); extents shrunk, hash inputs, single run. Snapshot lib-05.
