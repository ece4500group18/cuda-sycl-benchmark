# Original CUDA

Standalone Stage 1 CUDA case for `gridSyncCGReduction`.

reduceBlock is upstream cuda-samples code verbatim; reduceSinglePassMultiBlockCG has only a float->double change; the two-launch counterpart kernels, host driver, gen_val, and CMakeLists are new. input[i]=((i%23)-11)*0.5 keeps every partial sum exact. Writes the cooperative kernel's scalar sum to output/cuda_output.txt; checked by ../tests/verify.py. Requires a GPU with cooperativeLaunch support.
