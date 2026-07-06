# Original CUDA

Standalone Stage 1 CUDA case for `dynamicSharedMinReduction`.

timedReduction is upstream cuda-samples clock.cu code with only the timing instrumentation removed; naiveMinReduction, host driver, gen_input, and CMakeLists are new. Writes 8 per-block minima to output/cuda_output.txt; checked by ../tests/verify.py.
