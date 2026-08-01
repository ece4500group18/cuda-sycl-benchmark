# Intel GPU Hardware — Xe2 (Battlemage) & Xe3P (Crescent Island)

Target-hardware reference for setting per-kernel expectations (roofline / ridge point) and choosing
work-group sizes, sub-group sizes, SLM usage, and the AOT device token. The agent supports a
**catalog of target platforms**; the active one is `target.platform` in `.sycl/config.json`.

> **Provenance:** rows marked `[VERIFY]` MUST be confirmed on the running device and the official
> spec sheet, then recorded (with source + date) in `.sycl/state/lessons.md`. `[ARCH]` facts are
> stable for a given architecture.

## Supported target platforms `[SPEC]`/`[VERIFY]`
Pick one with `target.platform`. `aot_device` is the `-Xs "-device <token>"` value for AOT builds.

| platform key | GPU | Arch | Die | aot_device | VRAM | Mem BW (GB/s) | FP32 (TFLOPS) | FP16/BF16 XMX (TFLOPS) | INT8/FP8 XMX (TOPS/TFLOPS) | MxFP4 | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `b60` | Intel Arc Pro B60 | Xe2 (Battlemage) | BMG-G21 | `bmg-g21` | 24 GB GDDR6 | 456 `[SPEC]` | 12.28 `[SPEC]` | 98.5 `[SPEC]` | 197 `[SPEC]` | — | Shipping (Q3 2025) |
| `b70` | Intel Arc Pro B70 | Xe2 (Battlemage) | BMG-G31 | `bmg-g31` | 32 GB GDDR6 | 608 `[SPEC]` | 22.94 `[SPEC]` | 183.5 `[SPEC]` | 367 `[SPEC]` | — | Launch Q1 2026 |
| `cri` | Intel Crescent Island | Xe3P (datacenter) | — | `cri` | 160 GB LPDDR5X | 1370 `[VERIFY]` | 20.5 `[VERIFY]` | 300 `[VERIFY]` | 600 `[VERIFY]` | 1200 `[VERIFY]` | Sampling 2H 2026 |

Compute-unit counts (for occupancy math): B60 = 20 Xe2-cores / 160 XMX; B70 = 32 Xe2-cores / 256 XMX;
CRI = 32 Xe3P-cores / 256 XMX. `[VERIFY]` the live device with `sycl-ls --verbose` / `zeinfo`.

## Official sources `[SPEC]`
- Arc Pro B70: https://www.intel.com/content/www/us/en/products/sku/245797/intel-arc-pro-b70-graphics/specifications.html
- Arc Pro B60: https://www.intel.com/content/www/us/en/products/sku/243534/intel-arc-pro-b60-graphics/specifications.html
- Crescent Island (Xe3P): https://newsroom.intel.com/artificial-intelligence/intel-unveils-new-gpu-built-for-ai-inference
- Query the live device: `sycl-ls --verbose`, `clinfo`, Level Zero `zeinfo` for EU count, max
  work-group size, SLM size, sub-group sizes.

## Architecture facts `[ARCH]` (stable per arch)
Common to **Xe2** and **Xe3P** unless noted:
- Compute organized into **Xe-cores**, each containing Xe Vector Engines (XVE) and **XMX** (matrix)
  engines. Xe2 groups cores as 4 sub-slices(Xe-cores)/slice; **Xe3P (CRI)** uses 8 sub-slices(Xe-cores)/slice — same 256 XMX total on CRI vs B70 but a different slice topology, so re-verify
  occupancy limits on Xe3P.
- **SIMD / sub-group width**: native **SIMD16**; supported sub-group sizes typically **{16, 32}**.
  Prefer 16 unless a collective/XMX path benefits from 32. Do **not** force a sub-group size on
  simple element-wise kernels.
- **XMX (matrix engines)**: the DPAS units that accelerate dense GEMM/conv in FP16/BF16/INT8 (and
  lower). Route dense matmul/conv through **oneDNN** (production GEMM/conv library) or **sycl-tla**
  (CUTLASS-style tiled tensor algebra, for custom/fused kernels) — both actively maintained with
  verified Intel-GPU benchmarks. Do **not** hand-roll XMX via `joint_matrix` or FMA loops.
- **Shared Local Memory (SLM)**: per work-group scratchpad, commonly **up to 128 KB per Xe-core** on
  Xe2 (verify via `local_mem_size`); **Xe3P (CRI)** reports a larger 384 KB local memory + 512 KB L1
  `[VERIFY]`. Use `sycl::local_accessor`. Note: Intel L1 already caches read-only global data well —
  manual SLM caching of read-only vectors is often a regression.
- **Registers / GRF**: large register file with optional "large GRF" mode; high register pressure
  spills to memory and kills occupancy. Watch generated code for spills.
- **Cache hierarchy**: per-Xe-core L1 + shared L2. Read-only data benefits from L1 automatically.
- **Integer math**: 32-bit integer ops are far cheaper than 64-bit on the GPU. Use `int32` for
  indexing/strides whenever dimensions fit; avoid `size_t`/64-bit division in hot paths.

## Device envelope (confirm on the live device — `[VERIFY]`)
The peaks the roofline uses live in `target.*` of `.sycl/config.json` (a snapshot of the chosen
platform row above). Confirm them against the running device + spec sheet and record in
`.sycl/state/lessons.md`.

| Metric | Arc Pro B60 | Arc Pro B70 | Crescent Island (CRI) | How to get it |
|--------|-------------|-------------|------------------------|----------------|
| Xe-cores / XMX | 20 / 160 | 32 / 256 | 32 / 256 (Xe3P) | spec sheet / `zeinfo` |
| GPU clock (boost) | ~2.4 GHz | ~2.8 GHz | ~2.5 GHz | spec sheet / sysfs |
| Peak FP32 (TFLOPS) | 12.28 | 22.94 | 20.5 | `target.peak_fp32_tflops` |
| Peak FP16/BF16 (XMX) | 98.5 | 183.5 | 300 | `target.peak_fp16_bf16_xmx_tflops` |
| Peak INT8 (XMX, TOPS) | 197 | 367 | 600 | `target.peak_int8_xmx_tops` |
| VRAM | 24 GB GDDR6 | 32 GB GDDR6 | 160 GB LPDDR5X | `target.vram_gb` |
| Memory bandwidth (GB/s) | 456 | 608 | 1370 | `target.peak_bandwidth_gbps` |
| Max work-group size | VERIFY (commonly 1024) | VERIFY | VERIFY | `device::max_work_group_size` |
| SLM per work-group | VERIFY (≤128 KB) | VERIFY (≤128 KB) | VERIFY (≤384 KB) | `device::local_mem_size` |
| AOT device token | `bmg-g21` | `bmg-g31` | `cri` `[VERIFY]` | `ocloc compile --help` / `target.aot_device` |

## Choosing / switching the target platform
1. Set `target.platform` (and the matching `gpu`/`arch`/`aot_device`/peaks) in `.sycl/config.json`.
   The `sycl-agent` detect phase probes the live device and compares it to this target.
2. On a **mismatch**, `target.on_mismatch` decides behavior: `halt` (default — stop and ask),
   `warn` (proceed on the configured target, flag it), or `adopt` (retarget config to the detected
   device). See workflow.md → *Phase: detect*.
3. Xe2 B60 uses `bmg-g21`, B70 uses `bmg-g31` for AOT token; **Xe3P (CRI)** needs a different token `cri` — leave
   AOT off (JIT) until the token is confirmed with `ocloc`. Re-verify all `[VERIFY]` peaks after switching.


## Roofline method `[SPEC]`
1. **Arithmetic intensity** `AI = FLOPs / bytes_moved`.
2. **Ridge point** `= peak_FLOPS / peak_bandwidth`.
   - `AI < ridge` → **memory-bound**: optimize for bandwidth (coalesced access, vectorized
     loads/stores, fewer passes, fuse).
   - `AI > ridge` → **compute-bound**: optimize ALU/XMX utilization, native math, reduce redundant work.
3. **Expectation per kernel**: achieved bandwidth should approach a healthy fraction of peak for
   memory-bound kernels; achieved FLOPS for compute-bound. A ≥60–70% of relevant peak is a reasonable
   first target; tune with profiling evidence.

> **Performance is judged vs the SYCL baseline and this hardware roofline**
