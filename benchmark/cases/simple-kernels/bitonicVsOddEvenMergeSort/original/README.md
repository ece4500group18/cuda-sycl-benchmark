# Original CUDA

Standalone Stage 1 CUDA case for `bitonicVsOddEvenMergeSort`.

Comparator + both sort-network kernels are upstream cuda-samples code verbatim; host driver, gen_* input generators, and CMakeLists are new. Keys use k[i]=(i*40503)%65536 (all distinct). Writes the bitonic kernel's sorted 'key val' pairs to output/cuda_output.txt; checked by ../tests/verify.py.
