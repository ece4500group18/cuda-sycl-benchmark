# Backprop layer-forward tree reduction

The input-to-hidden layer-forward pass of a backpropagation network. Each thread block covers 16 input units x 16 hidden units; for its hidden unit it computes a 16-wide dot product via the classic shared-memory power-of-two-stride tree reduction (stride 2,4,8,16 gated by __syncthreads). Fixed HEIGHT=WIDTH=16 make every stride an exact power of two, so the reduction order is deterministic and the CPU reference matches it step for step.

Source project: yuhc/gpu-rodinia

Source URL: https://github.com/yuhc/gpu-rodinia/blob/master/cuda/backprop/backprop_cuda_kernel.cu

Snapshot: benchmark/collection/simple-kernels/sources/sk-02

License: BSD-3-Clause

Extraction fidelity: extracted

Extraction notes: bpnn_layerforward_CUDA reproduced verbatim from Rodinia backprop_cuda_kernel.cu. The host driver, CMakeLists, and Python oracle are new; deterministic gen_input/gen_weight are inlined into the host. Snapshot sk-02.
