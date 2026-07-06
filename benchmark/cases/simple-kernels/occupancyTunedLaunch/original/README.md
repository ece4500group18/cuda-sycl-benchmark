# Original CUDA

Standalone Stage 1 CUDA case for `occupancyTunedLaunch`.

The square kernel is upstream cuda-samples code verbatim; the host driver (manual + occupancy-tuned launches), gen_array, and CMakeLists are new. array[i]=i%1000. Writes the occupancy-tuned launch's squared array (1048576 uints) to output/cuda_output.txt; checked by ../tests/verify.py.
