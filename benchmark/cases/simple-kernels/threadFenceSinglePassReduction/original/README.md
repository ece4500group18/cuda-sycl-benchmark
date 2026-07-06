# Original CUDA

Standalone Stage 1 CUDA case for `threadFenceSinglePassReduction`.

All reduction kernels/device functions are upstream cuda-samples code verbatim; the host driver, gen_input, and CMakeLists are new. input[i]=((i%29)-14)*0.25, N=131072. Writes two lines (single-pass sum, two-launch sum) to output/cuda_output.txt; checked by ../tests/verify.py.
