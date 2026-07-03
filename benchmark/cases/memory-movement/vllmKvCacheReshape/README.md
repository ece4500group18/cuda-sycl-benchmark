# vLLM KV cache reshape

Token-major to head-major KV cache layout transform, adapted from vLLM cache kernels.

Source project: vllm-project/vllm

Source URL: https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/cache_kernels.cu

License: Apache-2.0

Extraction notes: Standalone KV cache reshape inspired by vLLM cache_kernels.cu.
