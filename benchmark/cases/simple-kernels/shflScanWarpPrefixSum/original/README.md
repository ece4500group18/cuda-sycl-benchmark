# Original CUDA

Standalone Stage 1 CUDA case for `shflScanWarpPrefixSum`.

shfl_scan_test is upstream cuda-samples code verbatim; the host driver (both block-size launches), gen_in, and CMakeLists are new. in[i]=(i%9)+1, no cross-block carry. Writes the 256-wide (multi-warp) scan of 262144 ints to output/cuda_output.txt; checked by ../tests/verify.py.
