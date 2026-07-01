# Pilot Benchmark Performance

Metric: end-to-end process runtime in seconds, including program startup and output writing.

## Summary

| variant | pass | fail | skipped | unknown |
| --- | --- | --- | --- | --- |
| cuda | 50 | 0 | 0 | 0 |
| sycl | 0 | 0 | 50 | 0 |

## Per-case Detail

| case_id | category | variant | status | metric | median_s | mean_s | min_s | max_s | stdev_s | repeat | warmup | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| affine | easy | cuda | pass | end_to_end_process_runtime | 0.210890 | 0.210451 | 0.208792 | 0.211670 | 0.001488 | 3 | 1 |  |
| affine | easy | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| gridStride | easy | cuda | pass | end_to_end_process_runtime | 0.221923 | 0.227369 | 0.220641 | 0.239542 | 0.010562 | 3 | 1 |  |
| gridStride | easy | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| hadamard | easy | cuda | pass | end_to_end_process_runtime | 0.192933 | 0.194703 | 0.191952 | 0.199223 | 0.003945 | 3 | 1 |  |
| hadamard | easy | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| matrixMul | easy | cuda | pass | end_to_end_process_runtime | 0.184593 | 0.183266 | 0.173606 | 0.191599 | 0.009070 | 3 | 1 |  |
| matrixMul | easy | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| reverseArray | easy | cuda | pass | end_to_end_process_runtime | 0.201200 | 0.202922 | 0.200448 | 0.207119 | 0.003654 | 3 | 1 |  |
| reverseArray | easy | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| saxpy | easy | cuda | pass | end_to_end_process_runtime | 0.203662 | 0.203624 | 0.201560 | 0.205652 | 0.002046 | 3 | 1 |  |
| saxpy | easy | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| simpleTemplates | easy | cuda | pass | end_to_end_process_runtime | 0.194392 | 0.194850 | 0.187281 | 0.202878 | 0.007809 | 3 | 1 |  |
| simpleTemplates | easy | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| square | easy | cuda | pass | end_to_end_process_runtime | 0.203342 | 0.201846 | 0.197742 | 0.204453 | 0.003597 | 3 | 1 |  |
| square | easy | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| transpose | easy | cuda | pass | end_to_end_process_runtime | 0.195697 | 0.195747 | 0.191528 | 0.200016 | 0.004244 | 3 | 1 |  |
| transpose | easy | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| vectorAdd | easy | cuda | pass | end_to_end_process_runtime | 0.221140 | 0.219230 | 0.214620 | 0.221931 | 0.004012 | 3 | 1 |  |
| vectorAdd | easy | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| bitonicSort | medium | cuda | pass | end_to_end_process_runtime | 0.175915 | 0.177724 | 0.170214 | 0.187042 | 0.008558 | 3 | 1 |  |
| bitonicSort | medium | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| conv1dShared | medium | cuda | pass | end_to_end_process_runtime | 0.213433 | 0.212188 | 0.204935 | 0.218196 | 0.006717 | 3 | 1 |  |
| conv1dShared | medium | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| dotProduct | medium | cuda | pass | end_to_end_process_runtime | 0.178052 | 0.183250 | 0.176583 | 0.195117 | 0.010303 | 3 | 1 |  |
| dotProduct | medium | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| histogram | medium | cuda | pass | end_to_end_process_runtime | 0.174867 | 0.180095 | 0.172315 | 0.193104 | 0.011338 | 3 | 1 |  |
| histogram | medium | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| nbodyTiled | medium | cuda | pass | end_to_end_process_runtime | 0.186562 | 0.186029 | 0.174354 | 0.197172 | 0.011419 | 3 | 1 |  |
| nbodyTiled | medium | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| reduceMax | medium | cuda | pass | end_to_end_process_runtime | 0.181613 | 0.180008 | 0.173391 | 0.185019 | 0.005978 | 3 | 1 |  |
| reduceMax | medium | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| reduceSum | medium | cuda | pass | end_to_end_process_runtime | 0.172031 | 0.172546 | 0.171519 | 0.174089 | 0.001360 | 3 | 1 |  |
| reduceSum | medium | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| scanBlock | medium | cuda | pass | end_to_end_process_runtime | 0.181181 | 0.182849 | 0.175534 | 0.191832 | 0.008276 | 3 | 1 |  |
| scanBlock | medium | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| tiledMatmul | medium | cuda | pass | end_to_end_process_runtime | 0.208064 | 0.209232 | 0.207691 | 0.211941 | 0.002354 | 3 | 1 |  |
| tiledMatmul | medium | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| transposeShared | medium | cuda | pass | end_to_end_process_runtime | 0.197630 | 0.200662 | 0.193666 | 0.210690 | 0.008908 | 3 | 1 |  |
| transposeShared | medium | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| bfs | hpc | cuda | pass | end_to_end_process_runtime | 0.176762 | 0.179117 | 0.172813 | 0.187777 | 0.007755 | 3 | 1 |  |
| bfs | hpc | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| conjugateGradient | hpc | cuda | pass | end_to_end_process_runtime | 0.189501 | 0.188089 | 0.180045 | 0.194722 | 0.007440 | 3 | 1 |  |
| conjugateGradient | hpc | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| finiteDiff1d | hpc | cuda | pass | end_to_end_process_runtime | 0.210508 | 0.211249 | 0.207731 | 0.215508 | 0.003941 | 3 | 1 |  |
| finiteDiff1d | hpc | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| heat2d | hpc | cuda | pass | end_to_end_process_runtime | 0.186276 | 0.186769 | 0.177710 | 0.196321 | 0.009315 | 3 | 1 |  |
| heat2d | hpc | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| jacobi2d | hpc | cuda | pass | end_to_end_process_runtime | 0.176758 | 0.182655 | 0.175015 | 0.196192 | 0.011756 | 3 | 1 |  |
| jacobi2d | hpc | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| monteCarloPi | hpc | cuda | pass | end_to_end_process_runtime | 0.173999 | 0.174519 | 0.170302 | 0.179256 | 0.004499 | 3 | 1 |  |
| monteCarloPi | hpc | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| spmv | hpc | cuda | pass | end_to_end_process_runtime | 0.214104 | 0.213431 | 0.207801 | 0.218388 | 0.005326 | 3 | 1 |  |
| spmv | hpc | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| stencil1d | hpc | cuda | pass | end_to_end_process_runtime | 0.210736 | 0.211939 | 0.207806 | 0.217274 | 0.004847 | 3 | 1 |  |
| stencil1d | hpc | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| stencil2d | hpc | cuda | pass | end_to_end_process_runtime | 0.198785 | 0.200078 | 0.194781 | 0.206667 | 0.006047 | 3 | 1 |  |
| stencil2d | hpc | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| stencil3d | hpc | cuda | pass | end_to_end_process_runtime | 0.258673 | 0.259277 | 0.257698 | 0.261459 | 0.001951 | 3 | 1 |  |
| stencil3d | hpc | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| attention | ai | cuda | pass | end_to_end_process_runtime | 0.181379 | 0.184069 | 0.181094 | 0.189735 | 0.004909 | 3 | 1 |  |
| attention | ai | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| batchedGemm | ai | cuda | pass | end_to_end_process_runtime | 0.201944 | 0.198543 | 0.191327 | 0.202359 | 0.006253 | 3 | 1 |  |
| batchedGemm | ai | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| embedding | ai | cuda | pass | end_to_end_process_runtime | 0.369749 | 0.369811 | 0.348216 | 0.391466 | 0.021625 | 3 | 1 |  |
| embedding | ai | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| gelu | ai | cuda | pass | end_to_end_process_runtime | 0.571223 | 0.568147 | 0.560730 | 0.572488 | 0.006454 | 3 | 1 |  |
| gelu | ai | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| gemm | ai | cuda | pass | end_to_end_process_runtime | 0.205857 | 0.206395 | 0.204481 | 0.208847 | 0.002232 | 3 | 1 |  |
| gemm | ai | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| layernorm | ai | cuda | pass | end_to_end_process_runtime | 0.366067 | 0.367987 | 0.365150 | 0.372744 | 0.004145 | 3 | 1 |  |
| layernorm | ai | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| rmsnorm | ai | cuda | pass | end_to_end_process_runtime | 0.374233 | 0.375207 | 0.361687 | 0.389702 | 0.014033 | 3 | 1 |  |
| rmsnorm | ai | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| rope | ai | cuda | pass | end_to_end_process_runtime | 0.200200 | 0.196387 | 0.187923 | 0.201037 | 0.007342 | 3 | 1 |  |
| rope | ai | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| softmax | ai | cuda | pass | end_to_end_process_runtime | 0.292448 | 0.288469 | 0.269654 | 0.303306 | 0.017175 | 3 | 1 |  |
| softmax | ai | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| topk | ai | cuda | pass | end_to_end_process_runtime | 0.197887 | 0.193886 | 0.181367 | 0.202402 | 0.011074 | 3 | 1 |  |
| topk | ai | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| cublasAxpy | library_api | cuda | pass | end_to_end_process_runtime | 0.694164 | 0.683367 | 0.657187 | 0.698752 | 0.022789 | 3 | 1 |  |
| cublasAxpy | library_api | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| cublasGemm | library_api | cuda | pass | end_to_end_process_runtime | 0.338175 | 0.338778 | 0.337632 | 0.340528 | 0.001539 | 3 | 1 |  |
| cublasGemm | library_api | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| cudaEventTiming | library_api | cuda | pass | end_to_end_process_runtime | 0.495517 | 0.489940 | 0.463558 | 0.510743 | 0.024082 | 3 | 1 |  |
| cudaEventTiming | library_api | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| cudaGraph | library_api | cuda | pass | end_to_end_process_runtime | 0.461828 | 0.465221 | 0.455426 | 0.478411 | 0.011862 | 3 | 1 |  |
| cudaGraph | library_api | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| cudaMemcpyAsyncPinned | library_api | cuda | pass | end_to_end_process_runtime | 0.484541 | 0.497632 | 0.476309 | 0.532048 | 0.030087 | 3 | 1 |  |
| cudaMemcpyAsyncPinned | library_api | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| cudaStream | library_api | cuda | pass | end_to_end_process_runtime | 0.520813 | 0.521430 | 0.507565 | 0.535914 | 0.014185 | 3 | 1 |  |
| cudaStream | library_api | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| cufftC2C | library_api | cuda | pass | end_to_end_process_runtime | 0.213772 | 0.212459 | 0.198537 | 0.225068 | 0.013314 | 3 | 1 |  |
| cufftC2C | library_api | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| curandUniform | library_api | cuda | pass | end_to_end_process_runtime | 0.605716 | 0.605306 | 0.584641 | 0.625562 | 0.020464 | 3 | 1 |  |
| curandUniform | library_api | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| thrustReduce | library_api | cuda | pass | end_to_end_process_runtime | 0.225458 | 0.217143 | 0.195760 | 0.230212 | 0.018671 | 3 | 1 |  |
| thrustReduce | library_api | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
| thrustSort | library_api | cuda | pass | end_to_end_process_runtime | 0.564282 | 0.557871 | 0.539262 | 0.570070 | 0.016374 | 3 | 1 |  |
| thrustSort | library_api | sycl | skipped_not_built |  |  |  |  |  |  |  |  | sycl_compile is skipped_no_sycl_compiler |
