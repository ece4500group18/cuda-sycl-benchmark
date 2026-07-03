# Rodinia backprop training step (forward + weight adjust)

NN training step: shared-memory layer-forward with in-block tree reduction, host sigmoid squash over block partial sums, then momentum weight adjustment.

Source project: ORNL/HeCBench

Source URL: https://github.com/zjin-lcf/HeCBench/blob/master/src/backprop-cuda/main.cu

Upstream commit: 01f58fc5

License: BSD-3-Clause

Extraction fidelity: extracted

Extraction notes: kernel_layerforward and kernel_adjust_weights verbatim (incl. the in-block tree reduction's initial self-doubling quirk, replicated exactly in the oracle); upstream launch geometry and host squash loop kept; deterministic hash data replaces random init and the CPU error chain (documented). From the HeCBench raw clone, registered as mkp-04.
