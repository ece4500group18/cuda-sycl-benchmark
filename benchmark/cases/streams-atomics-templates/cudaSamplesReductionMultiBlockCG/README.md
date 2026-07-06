# Single-pass grid-wide reduction with cooperative groups

Grid-synchronized single-kernel sum reduction: warp-tile cg::reduce, block reduction, grid.sync(), final cross-block accumulation.

Source project: NVIDIA/cuda-samples

Source URL: https://github.com/NVIDIA/cuda-samples/blob/master/Samples/2_Concepts_and_Techniques/reductionMultiBlockCG/reductionMultiBlockCG.cu

Upstream commit: b7c5481c

License: BSD-3-Clause + CUDA EULA note

Extraction fidelity: extracted

Extraction notes: reduceBlock, reduceSinglePassMultiBlockCG and the cudaLaunchCooperativeKernel wrapper verbatim; occupancy-sized grid like upstream; deterministic hash input.
