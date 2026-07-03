# cuTENSOR partial tensor reduction

Partial reduction C_mv = alpha*sum_hk A_mhkv via the cuTENSOR plan-based API.

Source project: NVIDIA/CUDALibrarySamples

Source URL: https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuTENSOR/reduction.cu

License: BSD-3-Clause

Extraction fidelity: extracted

Extraction notes: cuTENSOR 2.x plan API verbatim with CreateReduction/CUTENSOR_OP_ADD and cutensorReduce; extents shrunk, hash inputs, single run. Snapshot lib-04.
