# ggml interleaved RoPE

Interleaved rotary position embedding over token rows, adapted from ggml CUDA RoPE kernels.

Source project: ggml-org/llama.cpp

Source URL: https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/rope.cu

License: MIT

Extraction notes: Standalone simplification of llama.cpp/ggml rope.cu; host code fixes deterministic positions and tensor dimensions.
