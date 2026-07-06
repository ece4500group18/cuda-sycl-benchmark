# Original CUDA

Standalone Stage 1 CUDA case for `deviceAssertGuard`.

testKernel (the device assert) is upstream cuda-samples code verbatim; testKernelFlag, host driver, and CMakeLists are new. Writes 1024 per-thread 0/1 predicate flags to output/cuda_output.txt; checked by ../tests/verify.py. Part 2 intentionally trips the device assert (expected 'Assertion failed' stderr lines; cleared before exit).
