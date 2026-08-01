# Knowledge Base Index

Route to the smallest shard that answers your question. Load one shard at a time.

| If you need… | Load |
|--------------|------|
| CUDA→SYCL 2020 API mapping (execution model, memory, warp/sub-group, atomics, math, libraries) | [sycl-kernel-patterns.md](sycl-kernel-patterns.md) |
| Canonical SYCL kernel patterns (element-wise, reduction, SLM, GEMM/XMX) | [sycl-kernel-patterns.md](sycl-kernel-patterns.md) |
| **Required** migration recipes: parallel **reduction** (row-per-work-group; never a serial loop) and **GEMM** (oneMKL/oneDNN library call; never a naive triple loop) | [sycl-kernel-patterns.md](sycl-kernel-patterns.md) |
| Triton→SYCL 2020 mapping (program/tile model, block pointers + masks, `tl.dot`, reductions, atomics, autotune) | [triton-patterns.md](triton-patterns.md) |
| Whether/how to target **sycl-tla** (CUTLASS-style tiled GEMM/attention) vs plain SYCL | [sycl-tla-patterns.md](sycl-tla-patterns.md) |
| Xe2/Xe3 hardware facts: sub-group width, SLM, XMX, cache, int32 vs int64 | [intel-gpu-hardware.md](intel-gpu-hardware.md) |
| Target platform catalog (B60/B70/CRI): peaks, AOT token, how to switch targets | [intel-gpu-hardware.md](intel-gpu-hardware.md) |
| Device envelope (peak FP/bandwidth, ridge point) + how to verify it | [intel-gpu-hardware.md](intel-gpu-hardware.md) |
| Roofline method (classify memory- vs compute-bound) | [intel-gpu-hardware.md](intel-gpu-hardware.md) |
| icpx/AOT build & compile flags: JIT vs AOT, device tokens, oneMKL/oneDNN/sycl-tla linking, common errors | [sycl-build-guide.md](sycl-build-guide.md) |
| Intel GPU profiling **tools**: `unitrace` modes (timing/metric/stall), scoped collection, IGC asm dump, VTune/Advisor, backend selector | [intel-gpu-profiling-tools.md](intel-gpu-profiling-tools.md) |
| Optimization lookup table (profiling signal → strategy, priority, conflicts/synergies) | [sycl-optimization-catalog.md](sycl-optimization-catalog.md) |
| Proven anti-patterns (what regressed and why) | [sycl-optimization-catalog.md](sycl-optimization-catalog.md) |
| `unitrace` metric-group + counter catalog, derived ratios, thresholds, stall-name crosswalk — **BMG (Xe2)** | [intel-gpu-hardware-metrics-bmg.md](intel-gpu-hardware-metrics-bmg.md) |
| `unitrace` metric-group + counter catalog, derived ratios, thresholds — **CRI (Xe3P)** deltas (4 ALU pipes, `L1Profile`, FLOP/lane/GRF, copy engine, XMX types) | [intel-gpu-hardware-metrics-cri.md](intel-gpu-hardware-metrics-cri.md) |
| How to actually apply one strategy (before→after code, correctness invariants, how to verify) — one card per strategy ID | [sycl-optimization-strategies/](sycl-optimization-strategies/) (e.g. [increase-ilp.md](sycl-optimization-strategies/increase-ilp.md)) |
| How real Intel-GPU SYCL code is written (oneDNN, sycl-tla, PyTorch XPU, vLLM XPU, Triton, IGC, …) | [intel-gpu-software-repos.md](intel-gpu-software-repos.md) |
| Run-specific measured facts / lessons | `.sycl/state/lessons.md` (project-local, provenance-tagged) |

## Adding a shard
Keep each shard focused on one topic and under a few hundred lines. If a shard grows too large, split
it and update this table. The goal is that the agent loads only what a given step needs.
