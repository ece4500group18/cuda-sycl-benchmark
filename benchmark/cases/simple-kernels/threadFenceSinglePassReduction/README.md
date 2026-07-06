# Thread-fence single-pass reduction

reduceSinglePass reduces each block's slice to a partial sum, then __threadfence() + atomicInc on a retirement counter lets the last-arriving block detect it is last and combine all partials -- all in one launch. The classic variant does the same with two ordinary kernel launches (launch ordering standing in for the fence). Inputs are multiples of 0.25 with bounded partial sums, so every float32 addition is exact and both trees produce the identical exact total.

Source project: NVIDIA/cuda-samples

Source URL: https://github.com/NVIDIA/cuda-samples/blob/master/cpp/2_Concepts_and_Techniques/threadFenceReduction/threadFenceReduction_kernel.cuh

Snapshot: benchmark/collection/simple-kernels/sources/sk-15

License: BSD-3-Clause

Extraction fidelity: extracted

Extraction notes: reduceBlock/reduceBlocks/reduceMultiPass/reduceSinglePass/retirementCount/setRetirementCount reproduced verbatim from NVIDIA/cuda-samples threadFenceReduction_kernel.cuh. Host driver, gen_input, CMakeLists, and Python oracle are new. Snapshot sk-15.
