# sycl-tla — CUTLASS-style Tiled Tensor Algebra for Intel GPUs (GEMM / Attention)

`sycl-tla` (*SYCL Templates for Linear Algebra*, Intel's SYCL port of NVIDIA CUTLASS/CuTe, formerly
"CUTLASS-SYCL" — `https://github.com/intel/sycl-tla`) is a **header-only C++ template library** for
building near-peak dense linear algebra on Intel Xe GPUs: templated GEMM, epilogue fusion, FP8 / narrow
-int quantized GEMM, grouped/batched (MoE) GEMM, Stream-K, and Flash Attention. It targets Intel Xe
(Data Center GPU Max/Flex, PVC), **Arc B580 / Battlemage (BMG)**, and Xe3/CRI, expressed in code as
`cutlass::arch::IntelXe` + `cutlass::arch::OpClassTensorOp` (XMX/DPAS). Projects consume it via CMake
FetchContent and compile with `icpx`.

Some kernels should **not** be migrated to plain hand-written SYCL: tiled GEMM, fused GEMM+epilogue,
and attention need XMX/DPAS tiling, register/SLM staging, and K-loop pipelining to reach a useful
fraction of peak. `sycl-tla` (SYCL Tile Algebra — the CUTLASS-style tile library for Intel GPUs)
provides those abstractions so you compose a high-performance kernel instead of hand-rolling
`joint_matrix`. This shard says **when** to target sycl-tla and **how** to structure the port.

`[REF: sycl-tla]` This is the *target* library. Its local checkout path is in
`.sycl/config.json` → `reference_repos` (`sycl-tla`) and, when enabled, `targets.sycl_tla`. Treat its
examples as pattern references (`[REF]`), not copy sources — see `intel-gpu-software-repos.md` rules.

## Decision rule — plain SYCL vs sycl-tla
Default is **plain-sycl**. Choose **sycl-tla** only when *all* of these hold:

1. The kernel is fundamentally **tiled tensor algebra**: dense/batched GEMM, GEMM + epilogue (bias,
   activation, residual), convolution-as-GEMM, or attention (QKᵀ → softmax → ·V).
2. It is **compute-bound on XMX** at the target's roofline (see `intel-gpu-hardware.md`) — i.e. peak
   depends on DPAS utilization, K-loop pipelining, and tile staging, which plain SYCL won't reach.
3. A **comparable example exists** in the sycl-tla checkout to adapt the tiling from (grep
   `examples/**`, `include/**`).
4. `targets.sycl_tla.enabled` is `true` in `.sycl/config.json` and its include path resolves.

If any fails, migrate to **plain SYCL** — for a standalone standard GEMM call **oneDNN/oneMKL**
(actively maintained, benchmark-verified); do **not** hand-roll `joint_matrix`. Small/one-shot
`tl.dot`, elementwise, reductions, norms, and anything memory-bound → **plain SYCL**. Record the
chosen `target_style` and the reason in the kernel detail.

> A pragmatic path: migrate to a **correct plain-SYCL** version first (fast to verify against the CPU
> oracle), then, if it is XMX-bound and short of roofline, re-target that kernel to sycl-tla during
> `sycl-optimization` as a single tracked trial. Correctness is always gated by the same CPU oracle.

## Programming model (CUTLASS-style, on Intel) `[REF: sycl-tla]`
sycl-tla mirrors CUTLASS's layered decomposition; the Intel port maps the collective/atom layers to
Xe sub-groups + XMX. You compose a kernel from these layers (names/namespaces may differ by version —
confirm against the local checkout, don't hardcode):

- **Tile shapes** — threadblock/work-group tile (M×N×K), warp/sub-group tile, and the DPAS instruction
  shape. Pick from a matching example, then tune in `sycl-optimization`.
- **Collective mainloop** — the staged K-loop that streams A/B tiles through SLM/registers into DPAS
  MMA with pipelining. This is what you get "for free" instead of hand-writing.
- **Collective epilogue** — fuses bias/activation/residual/scaling on the accumulator before store.
- **Device/kernel launcher** — builds the grid over output tiles and launches on a `sycl::queue`.
- **Concrete Intel types** — express the arch as `cutlass::arch::IntelXe` with
  `cutlass::arch::OpClassTensorOp` (XMX/DPAS); **`ClusterShape` is always `<1,1,1>` on IntelXe**;
  accumulator is fp32 for fp16/bf16 inputs. Prefer the **CollectiveBuilder** entry point (it picks the
  MMA/copy atoms, pipeline stages, and schedule from the dtypes + `TileShape`); drop to explicit
  MMA/copy atoms only when tuning. `TileShape` (M,N,K) is the main perf lever — tune it in
  `sycl-optimization`, not migration. Confirm all type/namespace names against the local checkout.

Structure a migrated GEMM as: choose tile shapes → instantiate the collective mainloop for the
A/B/C dtypes (e.g. bf16 in, fp32 acc) → attach the epilogue that matches the source's fused ops →
launch over the problem shape. Recover M/N/K, dtypes, transpositions, and the epilogue from the
`cuda-analysis`/`triton-analysis` detail (a CUDA tiled-GEMM, cuBLAS/cutlass call, or a Triton
`tl.dot` K-loop).

## Example — a BF16 GEMM via CollectiveBuilder `[REF: sycl-tla]`
Recommended entry point is the **CollectiveBuilder**: it selects the MMA, copy atoms, pipeline stages,
and schedule given the dtypes, layouts, and a `TileShape`. Compose a mainloop collective + an epilogue
collective, wrap them in `GemmUniversal`, and drive with `GemmUniversalAdapter`. Confirm exact
type/namespace names against the local checkout — versions differ; treat this as illustrative.

```cpp
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
using namespace cute;

// --- element / layout types ---
using ElementInputA  = bfloat16_t;
using ElementInputB  = bfloat16_t;
using ElementAcc     = float;              // fp32 accumulator for bf16/fp16 inputs
using ElementOutput  = float;
using ElementCompute = float;              // epilogue compute
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;
constexpr int AlignA = sizeof(ElementInputA);
constexpr int AlignB = sizeof(ElementInputB);
constexpr int AlignC = sizeof(ElementAcc);
constexpr int AlignD = sizeof(ElementOutput);

// --- work-group tile (main perf lever); ClusterShape is always <1,1,1> on IntelXe ---
using TileShape = Shape<_256, _256, _32>;   // strong BMG bf16 starting point

// --- mainloop: builder picks MMA/copy atoms, stages, schedule ---
using CollectiveMainloop = cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::IntelXe, cutlass::arch::OpClassTensorOp,
    ElementInputA, LayoutA, AlignA,
    ElementInputB, LayoutB, AlignB,
    ElementAcc,
    TileShape, Shape<_1,_1,_1>,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

// --- epilogue: fused Linear-Combination + activation (EVT) ---
using EpilogueOp = cutlass::epilogue::fusion::LinCombEltAct<
    cutlass::epilogue::thread::ReLu,        // swap for Identity / GELU / etc.
    ElementOutput, ElementCompute, ElementAcc, ElementAcc,
    cutlass::FloatRoundStyle::round_to_nearest>;

using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::IntelXe, cutlass::arch::OpClassTensorOp,
    TileShape, Shape<_1,_1,_1>,
    cutlass::epilogue::collective::EpilogueTileAuto, ElementCompute, ElementAcc,
    ElementAcc, LayoutC, AlignC,
    ElementOutput, LayoutD, AlignD,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    EpilogueOp>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
```

Knobs: `TileShape` (M,N,K) is the main perf lever (tune in `sycl-optimization`, not migration);
`ClusterShape` is always `<1,1,1>`; `StageCountAuto`/`KernelScheduleAuto` let the builder choose depth
and schedule; accumulator is fp32 for fp16/bf16 inputs; swap the `EpilogueOp` activation or use a full
Epilogue Visitor Tree (EVT) for multi-op fusion (bias + scale + activation). The lower-level
`00_bmg_gemm` example writes the same GEMM with **explicit** MMA/copy atoms + DispatchPolicy — use it
only when the builder's choices need overriding.

### Example map (`sycl-tla/examples/`) `[REF: sycl-tla]`
Start from `01`; drop to `00` only when tuning. Copy the kernel *structure*, not the source.

| Example | Demonstrates |
|---------|--------------|
| `00_bmg_gemm` | Baseline BMG GEMM with explicit MMA/copy atoms & DispatchPolicy |
| `01_bmg_gemm_with_collective_builder` | CollectiveBuilder + fused ReLU epilogue (**start here**) |
| `02_bmg_gemm_mixed_dtype` | Mixed-precision GEMM incl. dequantization |
| `03_bmg_gemm_streamk` | Stream-K scheduler for load balancing on skewed shapes |
| `04_bmg_grouped_gemm` | Batched GEMMs with distinct per-group problem sizes (MoE) |
| `05_bmg_gemm_with_epilogues` | Epilogue Visitor Tree (EVT) fusion patterns |
| `06_bmg_flash_attention` | Flash Attention V2 on BMG/PVC |
| `07_bmg_dual_gemm` | Two GEMMs sharing an A matrix fused into one kernel |
| `08_bmg_gemm_f8` | FP8 (→FP32) GEMM |
| `09_bmg_grouped_gemm_f8` | FP8 grouped GEMM |
| `10_bmg_grouped_gemm_mixed_dtype` | Mixed-precision grouped GEMM |
| `11_xe20_cutlass_library` | Profiler / library instantiation layer |
| `12_xe20_moe_gemm_cute_interface` | MoE GEMM via the CuTe interface |
| `13_bmg_gemm_bias` | GEMM with bias epilogue |

## Build & include wiring `[SPEC]`
- sycl-tla is **header-only, CUTLASS-style**. Add its include dir to the SYCL build; no separate link.
  Resolve the path from `.sycl/config.json` → `targets.sycl_tla.include` (falls back to the
  `reference_repos` `sycl-tla` path). Do **not** build the sycl-tla repo itself.
- Compile with the same `icpx -fsycl` toolchain the plain-SYCL build uses (AOT device token from
  `target.aot_device`); sycl-tla templates resolve to the Xe DPAS ops for that target.
- In the scaffolded CMake (`sycl-migration` templates), add the include dir behind an option so
  sycl-tla targets and plain-SYCL targets share one build tree.
- All builds/tests still run through `.sycl/scripts/run.sh` on the runner — never invoke `icpx`
  directly. sycl-tla needs a real Xe device to run (XMX), so its tests run on the runner GPU.
- **CMake switches** (confirm exact names against the checkout — they vary by version): the SYCL/Intel
  backend is enabled with `CUTLASS_ENABLE_SYCL=ON` and `SYCL_INTEL_TARGET=ON`;
  `CUTLASS_ENABLE_HEADERS_ONLY=ON` keeps it header-only; `DPCPP_SYCL_TARGET` is the AOT device token
  (`intel_gpu_bmg_g21` / `intel_gpu_cri`). Typical compile defines:
  `-DCUTLASS_ENABLE_SYCL -DCUTLASS_ENABLE_HEADERS_ONLY -DSYCL_INTEL_TARGET`, plus
  `-ftemplate-backtrace-limit=0` to tame template error spew. Header-only means long template compiles
  and verbose errors — build one config at a time and pin the checkout revision for reproducibility.
- **Study source**: the sycl-tla `examples/**` are the pattern reference (baseline GEMM,
  collective-builder + fused epilogue, mixed-dtype/dequant, Stream-K, grouped/MoE GEMM, EVT epilogues,
  flash-attention, FP8). Each ships a CMake target with a built-in `--verify=1` correctness check and an
  `--iterations` timing loop — reuse `--verify` for correctness; do your own perf measurement with
  pinned GPU frequency. Adapt the tiling pattern; never copy example source verbatim.

### Build script — CMake FetchContent `[SPEC]`
Pull sycl-tla header-only (prefer a local checkout via `SYCL_TLA_SRC_DIR` for fast iteration; fall back
to a pinned Git tag), wire the include dirs, and compile the kernel with the icpx `-fsycl` toolchain and
the AOT device token for the target GPU.

```cmake
include(FetchContent)
set(CUTLASS_ENABLE_HEADERS_ONLY ON  CACHE BOOL "")
set(CUTLASS_ENABLE_SYCL         ON  CACHE BOOL "")
set(DPCPP_SYCL_TARGET "intel_gpu_bmg_g21" CACHE STRING "")   # or intel_gpu_cri (Xe3)

# Local checkout (fastest for iteration): export SYCL_TLA_SRC_DIR=/path/to/sycl-tla
if(DEFINED ENV{SYCL_TLA_SRC_DIR})
  FetchContent_Declare(sycl-tla SOURCE_DIR $ENV{SYCL_TLA_SRC_DIR})
else()
  FetchContent_Declare(sycl-tla
    GIT_REPOSITORY https://github.com/intel/sycl-tla.git
    GIT_TAG v0.9.1 GIT_SHALLOW TRUE)
endif()
FetchContent_MakeAvailable(sycl-tla)

set(CUTLASS_INCLUDE_DIR        ${sycl-tla_SOURCE_DIR}/include)
set(CUTLASS_TOOLS_UTIL_INCLUDE ${sycl-tla_SOURCE_DIR}/tools/util/include)

add_executable(bmg_gemm bmg_gemm.cpp)
target_include_directories(bmg_gemm PRIVATE
  ${CUTLASS_INCLUDE_DIR} ${CUTLASS_TOOLS_UTIL_INCLUDE})
target_compile_options(bmg_gemm PRIVATE
  -fsycl -fsycl-targets=${DPCPP_SYCL_TARGET}
  -DCUTLASS_ENABLE_SYCL -DCUTLASS_ENABLE_HEADERS_ONLY -DSYCL_INTEL_TARGET
  -ftemplate-backtrace-limit=0)
target_link_options(bmg_gemm PRIVATE -fsycl -fsycl-targets=${DPCPP_SYCL_TARGET})
```

Equivalent one-off compile (source the oneAPI env first — `source /opt/intel/oneapi/setvars.sh`):

```bash
icpx -fsycl -fsycl-targets=intel_gpu_bmg_g21 \
  -DCUTLASS_ENABLE_SYCL -DCUTLASS_ENABLE_HEADERS_ONLY -DSYCL_INTEL_TARGET \
  -ftemplate-backtrace-limit=0 \
  -I"$SYCL_TLA_SRC_DIR/include" -I"$SYCL_TLA_SRC_DIR/tools/util/include" \
  bmg_gemm.cpp -o bmg_gemm
```

In the scaffolded build, gate sycl-tla targets behind an option so they share one build tree with the
plain-SYCL targets, and always drive builds/tests through `.sycl/scripts/run.sh` on the runner GPU —
never invoke `icpx` directly.

## Correctness (same CPU oracle) `[SPEC]`
A sycl-tla kernel is gated exactly like a plain-SYCL one: compared **live** against the CPU-reference
oracle within a dtype-appropriate tolerance. Mixed-precision tiles (bf16/fp16 A·B, fp32 accumulate)
need a looser `rtol` than fp32 — set it from the accumulate precision, and reuse the source project's
reference/tolerance when one exists (see `sycl-migration`).

## Optimization hooks (deferred to `sycl-optimization`) `[ARCH]`
Once correct, the tuning knobs are: work-group/sub-group **tile shapes**, K-loop **pipeline depth**
(stages), **SLM vs register** staging, epilogue fusion, and DPAS instruction shape. Each is one
tracked trial (keep-if-wins) against the target roofline. Cross-reference `oneDNN` (production Intel
GEMM/conv blocking) and `sycl-tla` `examples/**` for proven tile choices; log findings as `[REF: …]`.

## Do / don't `[ARCH]`
- DO reserve sycl-tla for real tiled tensor algebra that is XMX-bound; default everything else to
  plain SYCL.
- DO confirm the actual sycl-tla API/namespaces against the local checkout — versions differ; treat
  names here as illustrative.
- DO recover dtypes + fused epilogue from the source so the collective epilogue matches semantics.
- DON'T copy sycl-tla example source verbatim — adapt the tiling pattern into the project's own code.
- DON'T build the sycl-tla repo; consume it header-only.
- DON'T tune tile shapes/pipeline depth during migration — get it correct against the CPU oracle
  first, then hand it to `sycl-optimization`.
