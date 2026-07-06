# Original CUDA

Standalone Stage 1 CUDA case for `backpropTreeReduction`.

bpnn_layerforward_CUDA is upstream Rodinia code verbatim; the host driver, gen_* input generators, and CMakeLists are new. Writes 8192 hidden_partial_sum floats to output/cuda_output.txt; checked by ../tests/verify.py.
