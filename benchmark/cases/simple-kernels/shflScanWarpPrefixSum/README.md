# Warp-shuffle prefix-sum scan

An inclusive prefix sum (scan) via warp-shuffle __shfl_up_sync: each warp scans its lanes in log2 steps with pure register communication, then multi-warp blocks broadcast each warp's total through shared memory and uniformly add the cross-warp prefix. Launched at 32 threads (single warp; cross-warp stage is a no-op) and 256 threads (8 warps; broadcast fully exercised). Integer addition is associative, so a segmented CPU running sum (segment = block size) is an exact oracle.

Source project: NVIDIA/cuda-samples

Source URL: https://github.com/NVIDIA/cuda-samples/blob/master/cpp/2_Concepts_and_Techniques/shfl_scan/shfl_scan.cu

Snapshot: benchmark/collection/simple-kernels/sources/sk-11

License: BSD-3-Clause

Extraction fidelity: extracted

Extraction notes: shfl_scan_test reproduced verbatim from NVIDIA/cuda-samples shfl_scan.cu, launched at two block sizes (32 and 256). Host driver, gen_in, CMakeLists, and Python oracle are new. Snapshot sk-11.
