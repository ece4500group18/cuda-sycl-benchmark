# Original CUDA

Standalone Stage 1 CUDA case for `binaryPartitionOddEvenReduce`.

oddEvenCountAndSumCG is upstream cuda-samples code verbatim; oddEvenCountAndSumNaive, the host driver, gen() input, and CMakeLists are new. Writes the 3 CG results (numOfOdds, sumOfOdds, sumOfEvens) to output/cuda_output.txt; checked by ../tests/verify.py.
