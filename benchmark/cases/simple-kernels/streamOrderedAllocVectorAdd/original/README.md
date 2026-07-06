# Original CUDA

Standalone Stage 1 CUDA case for `streamOrderedAllocVectorAdd`.

vectorAddGPU is upstream cuda-samples code verbatim; the host driver (classic + stream-ordered allocation paths), gen_a/gen_b, and CMakeLists are new. a[i]=(i%23)-11, b[i]=(i%19)-9. Writes the stream-ordered path's c[] (1048576 floats) to output/cuda_output.txt; checked by ../tests/verify.py. Requires CUDA 11.2+ for cudaMallocAsync.
