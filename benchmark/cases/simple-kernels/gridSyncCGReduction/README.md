# Grid-sync cooperative-groups reduction

A grid-wide sum reduction done in one cooperative kernel launch: each block reduces its grid-stride share (warp-shuffle tree + leader loop) to a partial, a cg::sync(grid) barrier synchronizes all co-resident blocks, then grid thread 0 combines the partials. A two-plain-launch variant performs the identical arithmetic with kernel-launch ordering standing in for the grid barrier. Inputs are exact multiples of 0.5 with bounded partial sums, so both paths equal the CPU sum exactly.

Source project: NVIDIA/cuda-samples

Source URL: https://github.com/NVIDIA/cuda-samples/blob/master/cpp/2_Concepts_and_Techniques/reductionMultiBlockCG/reductionMultiBlockCG.cu

Snapshot: benchmark/collection/simple-kernels/sources/sk-09

License: BSD-3-Clause

Extraction fidelity: extracted

Extraction notes: reduceBlock reproduced verbatim and reduceSinglePassMultiBlockCG reproduced with a mechanical float->double type change, from NVIDIA/cuda-samples reductionMultiBlockCG.cu. reduceBlockPartial/reduceCombinePartials (two-launch counterpart), host driver, gen_val, CMakeLists, and Python oracle are new. Snapshot sk-09.
