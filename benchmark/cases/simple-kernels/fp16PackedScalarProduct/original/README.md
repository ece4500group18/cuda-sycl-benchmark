# Original CUDA

Standalone Stage 1 CUDA case for `fp16PackedScalarProduct`.

Both scalar-product kernels and both reduce helpers are upstream cuda-samples code verbatim; host driver, gen_* input, and CMakeLists are new. Inputs a[i]=i%4, b[i]=i%2 keep every fp16 sum in the exact-integer range. Writes 128 per-block partials (intrinsics kernel) to output/cuda_output.txt; checked by ../tests/verify.py.
