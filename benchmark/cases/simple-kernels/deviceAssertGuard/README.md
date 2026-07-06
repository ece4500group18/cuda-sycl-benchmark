# Device-side assert guard

testKernel calls assert(gtid < N). With 1024 threads and N=1000, gtid in [1000,1024) violate the predicate and trip the device assert, which surfaces asynchronously as cudaErrorAssert -- a GPU-only, non-arithmetic control-flow technique with no numeric answer. The paired testKernelFlag records (gtid < N) ? 1 : 0 per thread, giving a bit-exact CPU-checkable oracle for the predicate.

Source project: NVIDIA/cuda-samples

Source URL: https://github.com/NVIDIA/cuda-samples/blob/master/cpp/0_Introduction/simpleAssert/simpleAssert.cu

Snapshot: benchmark/collection/simple-kernels/sources/sk-06

License: BSD-3-Clause

Extraction fidelity: extracted

Extraction notes: testKernel reproduced verbatim from NVIDIA/cuda-samples simpleAssert.cu. testKernelFlag (numeric-oracle counterpart), host driver, CMakeLists, and Python oracle are new. Snapshot sk-06.
