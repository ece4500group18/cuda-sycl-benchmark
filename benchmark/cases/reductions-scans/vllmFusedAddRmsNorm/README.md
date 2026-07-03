# vLLM fused add RMSNorm

Fused residual add and RMSNorm, adapted from vLLM layernorm CUDA kernels.

Source project: vllm-project/vllm

Source URL: https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/layernorm_kernels.cu

License: Apache-2.0

Extraction notes: Standalone simplification of vLLM fused_add_rms_norm-style behavior.
