# vLLM top-1 MoE gate

Select the top-1 expert index per token from routing logits.

Source project: vllm-project/vllm

Source URL: https://github.com/vllm-project/vllm/blob/main/csrc/moe/dynamic_4bit_int_moe_cpu.cpp

License: Apache-2.0

Extraction fidelity: inspired_by

Extraction notes: CUDA standalone inspired by vLLM MoE routing logic; source file is CPU-side but belongs to the vLLM MoE component.
