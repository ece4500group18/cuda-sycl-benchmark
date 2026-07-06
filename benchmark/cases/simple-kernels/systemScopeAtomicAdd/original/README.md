# Original CUDA

Standalone Stage 1 CUDA case for `systemScopeAtomicAdd`.

atomicKernel and atomicKernel_CPU are upstream cuda-samples code verbatim; atomicKernel_device, the host driver, and CMakeLists are new. len=2*64*256=32768, LOOP_NUM=50; array all 0 except slots 7/9 = 0xff. Writes the 10 final system-array slots to output/cuda_output.txt; checked by ../tests/verify.py. Requires unified memory + system-scope atomics (sm_60+).
