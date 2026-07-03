# cuTENSOR 4D tensor contraction

General tensor contraction C_munv = alpha*A_mhkn*B_ukvh via the cuTENSOR plan-based API.

Source project: NVIDIA/CUDALibrarySamples

Source URL: https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuTENSOR/contraction.cu

License: BSD-3-Clause

Extraction fidelity: extracted

Extraction notes: cuTENSOR 2.x plan API verbatim (descriptors, CreateContraction, scalar-type query, plan preference, workspace estimate, plan, Contract); extents shrunk, hash inputs, single run. Snapshot lib-03.
