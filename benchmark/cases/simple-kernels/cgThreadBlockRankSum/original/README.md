# Original CUDA

Standalone Stage 1 CUDA case for `cgThreadBlockRankSum`.

sumReduction is upstream cuda-samples code verbatim; cgkernel is adapted to write per-block results, and the host driver + CMakeLists are new. Writes 16 per-block sums to output/cuda_output.txt; checked by ../tests/verify.py.
