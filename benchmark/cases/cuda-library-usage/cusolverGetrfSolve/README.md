# cuSOLVER dense LU solve (getrf/getrs)

Dense LU factorization with pivoting and linear solve via cusolverDn.

Source project: NVIDIA/CUDALibrarySamples

Source URL: https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSOLVER/getrf/cusolver_getrf_example.cu

License: Apache-2.0

Extraction fidelity: extracted

Extraction notes: cusolverDn API sequence verbatim (handle+stream, getrf_bufferSize, Dgetrf with pivoting, Dgetrs); 3x3 example scaled to deterministic 64x64 system. Snapshot lib-02.
