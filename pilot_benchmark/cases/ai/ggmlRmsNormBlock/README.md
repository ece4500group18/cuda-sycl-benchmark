# ggml RMSNorm block reduction

Per-row RMS normalization with learned weight, adapted from ggml CUDA normalization kernels.

Source project: ggml-org/llama.cpp

Source URL: https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/norm.cu

License: MIT

Extraction notes: Standalone simplification of llama.cpp/ggml norm.cu using deterministic row-major tensors and a CPU reference.
