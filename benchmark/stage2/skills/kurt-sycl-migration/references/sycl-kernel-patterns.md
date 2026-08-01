# SYCL 2020 Kernel Patterns (CUDA → SYCL migration)

Hand-write SYCL from the algorithm. **Do not** use `dpct`/`SYCLomatic`. Follow the SYCL 2020
specification. `[SPEC]` https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html

## Execution model mapping `[SPEC]`
| CUDA | SYCL 2020 |
|------|-----------|
| `__global__ void k(...)` + `k<<<grid, block>>>(...)` | `q.parallel_for(nd_range<N>(global, local), [=](nd_item<N> it){...})` |
| `threadIdx.x` | `it.get_local_id(0)` |
| `blockIdx.x` | `it.get_group(0)` |
| `blockDim.x` | `it.get_local_range(0)` |
| `gridDim.x` | `it.get_group_range(0)` |
| global index | `it.get_global_id(0)` |
| `__syncthreads()` | `sycl::group_barrier(it.get_group())` |
| warp (32) | sub-group (Intel native **16**, also 32) — `it.get_sub_group()` |
| `cudaMalloc` / `cudaMemcpy` | `sycl::malloc_device` / `q.memcpy(...)` (USM) |
| `cudaMallocManaged` | `sycl::malloc_shared` |
| stream | `sycl::queue` (in-order or out-of-order) |
| `cudaStreamSynchronize` | `q.wait()` / event-based deps |

## Memory `[SPEC]`
| CUDA | SYCL 2020 |
|------|-----------|
| `__shared__ T buf[N]` | `sycl::local_accessor<T,1> buf({N}, cgh)` |
| `__constant__` | USM device buffer (read-only) — L1 caches it; constant-like via kernel args |
| `__restrict__` | pass raw USM pointers; mark `__restrict__` where supported |
| texture/`tex1Dfetch` | plain device memory + manual interpolation, or `sycl::image` if needed |

## Warp/sub-group primitives `[SPEC]`
| CUDA | SYCL 2020 |
|------|-----------|
| `__shfl_sync` | `sycl::shift_group_left/right`, `select_from_group`, `permute_group_by_xor` |
| `__shfl_down_sync` reduction | `sycl::reduce_over_group(sg, val, sycl::plus<>())` |
| `__ballot_sync` | `sycl::group_ballot` (ext) / reductions over bool |
| `__any_sync` / `__all_sync` | `sycl::any_of_group` / `sycl::all_of_group` |
| block-wide reduction | sub-group reduce → SLM → work-group reduce, or `reduce_over_group(group, …)` |

## Atomics `[SPEC]`
| CUDA | SYCL 2020 |
|------|-----------|
| `atomicAdd(p, v)` | `sycl::atomic_ref<T, memory_order::relaxed, memory_scope::device, access::address_space::global_space>(*p) += v` |
| `atomicCAS` | `atomic_ref(...).compare_exchange_strong(...)` |

## Math `[SPEC]`
| CUDA | SYCL 2020 |
|------|-----------|
| `expf`, `sinf`, `__expf` | `sycl::exp`, `sycl::native::exp` (faster, lower precision) |
| `rsqrtf` | `sycl::rsqrt` / `sycl::native::rsqrt` |
| `fmaf` | `sycl::fma` |

## Libraries `[SPEC]`
| CUDA | Intel / SYCL |
|------|--------------|
| cuBLAS | **oneMKL** (`oneapi::mkl::blas`) |
| cuDNN | **oneDNN** (`dnnl`, SYCL interop) |
| Thrust | oneDPL (`oneapi::dpl`) |
| CUB block/warp ops | SYCL group/sub-group algorithms |
| cuFFT | oneMKL DFT |

## Canonical kernel patterns

### Minimal kernel skeleton `[SPEC]`
Every kernel below follows this shape: a `sycl::queue` on the GPU, raw **USM** pointers as kernel
arguments, and an `nd_range` `parallel_for`. **Standardize on USM, not buffers/accessors** — it maps
directly to the CUDA `cudaMalloc`/pointer model, keeps kernel signatures pointer-based (matching the
source), and avoids implicit buffer-migration surprises. Use `malloc_device` for compute data and
`q.memcpy(...)` for transfers; reach for `malloc_shared` only for prototyping.
```cpp
#include <sycl/sycl.hpp>

sycl::queue q{sycl::gpu_selector_v};            // pick the Intel GPU
float* d = sycl::malloc_device<float>(n, q);    // device USM (CUDA cudaMalloc)
q.memcpy(d, host, n * sizeof(float)).wait();    // H2D copy

constexpr int wg = 256;                          // work-group size (multiple of sub-group 16/32)
const int global = ((n + wg - 1) / wg) * wg;     // round up to a whole number of work-groups
q.parallel_for(sycl::nd_range<1>(global, wg), [=](sycl::nd_item<1> it) {
    const int i = it.get_global_id(0);
    if (i < n) d[i] = /* ... */ d[i];            // guard the tail: global ≥ n
}).wait();

q.memcpy(host, d, n * sizeof(float)).wait();     // D2H copy
sycl::free(d, q);
```

### Element-wise
```cpp
void sycl_scale(sycl::queue &q, const float* src, float* dst, float a, int n) {
    constexpr int wg = 256;
    const int global = ((n + wg - 1) / wg) * wg;
    q.parallel_for(sycl::nd_range<1>(global, wg), [=](sycl::nd_item<1> it) {
        const int i = it.get_global_id(0);
        if (i < n) dst[i] = a * src[i];
    });
}
```

### Reduction — REQUIRED recipe (do NOT serialize)
A CUDA reduction (warp-shuffle + shared-memory tree, `cub::BlockReduce`, cooperative groups, or a
`reduce`-style kernel) migrates to the **group-collective** form below. **Never** replace it with a
single work-item looping over the whole reduced axis — a serial per-row loop is *not* a faithful
migration of a parallel reduction; it is a correctness-only stub that throws away the parallelism the
source expressed. Use these two building blocks.

**(a) Two-level primitive** — combine per-work-item partials within one `nd_item`:
```cpp
// each work-item holds `val`; get the work-group total
float sg = sycl::reduce_over_group(it.get_sub_group(), val, sycl::plus<float>());   // within sub-group
float wg = sycl::reduce_over_group(it.get_group(),     val, sycl::plus<float>());   // whole work-group
// reduce_over_group(group, …) already spans the work-group; use the sub-group form only when you
// need the per-sub-group partial (e.g. to stage through SLM for very large work-groups).
```

**(b) One-row-per-work-group kernel** — the canonical layernorm / softmax / RMSNorm / mean-var
shape: assign **one work-group per row** and split the C-length axis across its work-items, so the
reduction is parallel. This is the pattern the naive "one work-item per row, serial loop over C"
migration gets wrong.
```cpp
void layernorm_forward(sycl::queue& q, float* out, float* mean, float* rstd,
                       const float* inp, const float* weight, const float* bias,
                       int N, int C, float eps) {
    constexpr int WG = 256;                         // work-group = one row's reduction team
    sycl::nd_range<1> ndr(sycl::range<1>(size_t(N) * WG), sycl::range<1>(WG));
    q.parallel_for(ndr, [=](sycl::nd_item<1> it) {
        const int row = it.get_group(0);            // one work-group owns row `row`
        const int lid = it.get_local_id(0);
        const float* x = inp + size_t(row) * C;

        // strided partial sums across the work-group, then a group reduction
        float s = 0.f;
        for (int i = lid; i < C; i += WG) s += x[i];
        const float m = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>()) / C;

        float v = 0.f;
        for (int i = lid; i < C; i += WG) { float d = x[i] - m; v += d * d; }
        const float var = sycl::reduce_over_group(it.get_group(), v, sycl::plus<float>()) / C;
        const float rs  = sycl::rsqrt(var + eps);

        if (lid == 0) { if (mean) mean[row] = m; if (rstd) rstd[row] = rs; }
        float* o = out + size_t(row) * C;
        for (int i = lid; i < C; i += WG) o[i] = rs * (x[i] - m) * weight[i] + bias[i];
    });
}
```
Notes: accumulate reductions in `float` (or `double` if the source does); `reduce_over_group` handles
the sub-group→work-group tree for you (no manual SLM tree needed for a single scalar). For **online
softmax** (max + sum in one pass) reduce a max with `sycl::maximum<>()` then the shifted-exp sum.
For a **grid-stride global reduction** (one scalar over a whole tensor), do the group reduction above
then `sycl::atomic_ref<...>(out[0]) += wg_partial` from `lid == 0`. When a mapping is non-obvious,
consult `intel-gpu-software-repos.md` → `pytorch` (`aten/src/ATen/native/xpu`, grep `reduce`) for idiomatic Xe
reductions. **Migrate reductions with this recipe, not a serial loop — this is a migration-time
requirement, not an optimization.**

### SLM with barrier
```cpp
q.submit([&](sycl::handler& h){
    sycl::local_accessor<float,1> smem(sycl::range<1>(256), h);
    h.parallel_for(ndr, [=](sycl::nd_item<1> it){
        smem[it.get_local_id(0)] = value;
        sycl::group_barrier(it.get_group());
        // safe to read smem
    });
});
```

### GEMM / matmul — REQUIRED recipe (use a library, do NOT hand-roll a triple loop)
A CUDA matmul that calls **cuBLAS / cuBLASLt / cuDNN / CUTLASS** migrates to a **library GEMM**, not a
naive `for (k) c += a[k]*b[k]` kernel. A one-work-item-per-output triple loop is memory-bound, ignores
XMX/DPAS, and is *not* a faithful migration of a cuBLAS call — it is a correctness-only stub. Choose,
in order:

> **oneMKL vs oneDNN — pick by fusion, not by speed.** Both dispatch the same XMX/DPAS-optimized
> kernels, so raw GEMM throughput is essentially identical. The difference is the programming model and
> what you can fuse. Default to **oneMKL** for a *naked* GEMM (BLAS one-liner, direct cuBLAS mapping,
> plain USM pointers). Switch to **oneDNN** the moment the source *fuses* an epilogue into the matmul
> (bias, GELU/ReLU, residual/sum, scales, zero-points) — oneDNN folds all of that into a **single**
> primitive via `post_ops`, so you keep the fusion in one launch instead of GEMM + a separate
> elementwise pass. Only drop to **sycl-tla** when neither library fits the data format/inner loop.

1. **oneMKL** — standard dense GEMM; near-peak on Intel GPUs. This is the direct cuBLAS replacement:
   ```cpp
   #include <oneapi/mkl.hpp>
   // C = alpha·op(A)·op(B) + beta·C, row-major (matches C/C++ storage)
   oneapi::mkl::blas::row_major::gemm(
       q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans,   // e.g. inp · weightᵀ
       M, N, K, alpha, A, lda, B, ldb, beta, C, ldc).wait();
   // Fused bias/activation: run the GEMM, then a cheap elementwise epilogue kernel — or use oneDNN.
   ```
2. **oneDNN** (`dnnl`, SYCL interop) — when you need a **fused** GEMM (bias + GELU/ReLU + residual,
   scales, or a cuDNN-style primitive) in one call. Build the engine/stream from the *same* `sycl::queue`
   so oneDNN runs on your device/stream, wrap USM pointers as `memory` objects, and attach the epilogue
   as `post_ops` — one launch, no epilogue round-trip:
   ```cpp
   #include <oneapi/dnnl/dnnl.hpp>
   #include <oneapi/dnnl/dnnl_sycl.hpp>
   using namespace dnnl;
   // Reuse the SYCL queue → oneDNN shares the device, context, and stream with your kernels
   engine eng  = sycl_interop::make_engine(q.get_device(), q.get_context());
   stream strm = sycl_interop::make_stream(eng, q);

   // Row-major operands: A[M,K], Bᵀ[N,K] → C[M,N]; dt = f32/f16/bf16 (in), f32 accumulate internally
   memory::desc a_md({M, K}, memory::data_type::f16, memory::format_tag::ab);
   memory::desc b_md({K, N}, memory::data_type::f16, memory::format_tag::ab);
   memory::desc c_md({M, N}, memory::data_type::f16, memory::format_tag::ab);

   // Fuse bias + GELU into the GEMM via post-ops (cuDNN-style single primitive)
   post_ops po; po.append_eltwise(algorithm::eltwise_gelu_erf, 0.f, 0.f);
   primitive_attr attr; attr.set_post_ops(po);
   matmul::primitive_desc pd(eng, a_md, b_md, /*bias*/ memory::desc(), c_md, attr);

   // Wrap USM device pointers (no copy); reuse `pd` across calls to amortize planning
   auto A = memory(a_md, eng, dA), B = memory(b_md, eng, dB), C = memory(c_md, eng, dC);
   matmul(pd).execute(strm, {{DNNL_ARG_SRC, A}, {DNNL_ARG_WEIGHTS, B}, {DNNL_ARG_DST, C}});
   strm.wait();
   ```
   See `intel-gpu-software-repos.md` → `oneDNN` (`src/gpu/intel`) for bias/scale args, batched matmul, and the
   full descriptor/primitive-cache setup.
3. **sycl-tla** — CUTLASS-style **tiled** tensor algebra (GEMM+epilogue, attention) for XMX-bound
   custom or fused kernels a library call doesn't fit, per the decision rule in `sycl-tla-patterns.md`.
   Actively maintained with verified Intel-GPU benchmarks.

Do **not** hand-roll XMX via `joint_matrix`: oneMKL/oneDNN and sycl-tla are actively maintained and
benchmark-verified, whereas a hand-written matrix kernel is unverified and near-impossible to keep at
peak across toolchain/arch changes. If no library fits a custom data format, target **sycl-tla**
rather than raw `joint_matrix`.

Recover M/N/K, the transposes, dtypes (e.g. bf16 in / fp32 accumulate), and any fused epilogue from
the `cuda-analysis`/`triton-analysis` detail. **Pick a library GEMM at migration time; leave only
tile-shape/pipeline tuning to `sycl-optimization`.**

### Custom XMX kernels — use sycl-tla, not hand-written `joint_matrix` `[SPEC]`
When no library GEMM fits (custom data format, fused inner loop), target **sycl-tla** (CUTLASS-style
tiled tensor algebra) rather than hand-rolling the Intel `joint_matrix` extension. sycl-tla and oneDNN
are actively maintained and their Intel-GPU performance is benchmark-verified; a hand-written
`joint_matrix` kernel is unverified, extremely sensitive to per-arch DPAS fragment shapes/dtypes, and
hard to keep near peak across toolchain releases. See `sycl-tla-patterns.md` for the decision rule and
skeleton, and `intel-gpu-software-repos.md` → `oneDNN`/`sycl-tla` for proven tile choices.

## Correctness validation (tolerances) `[SPEC]`
Compare the SYCL output against the CPU-reference oracle with the standard criterion
`|expected − actual| ≤ atol + rtol·|expected|` (the same test `torch.testing.assert_close` applies).
Starting tolerances by dtype (only for a *from-scratch* oracle — always reuse an inherited project
tolerance when one exists, per `sycl-migration` step 1):

| dtype | rtol | atol | note |
|---|---|---|---|
| fp32 | 1e-5 | 1e-8 | element-wise default |
| fp16 | 1e-3 | 1e-4 | ~3 significant digits |
| bf16 | 1e-2 | 1e-3 | ~2 significant digits |
| fp64 | 1e-12 | 1e-15 | |
| int8 / int32 | 0 | 0 | integer ops must match exactly |

- **Accumulated ops (GEMM, reductions, conv)** grow rounding error with the reduced length K: scale the
  absolute tolerance as `atol_K ≈ atol · sqrt(K)` (e.g. bf16 GEMM, K=4096 → `atol ≈ 1e-3·64 ≈ 0.06`).
  If mismatches appear only at large K, verify a few elements by hand *before* loosening — and never
  loosen an inherited tolerance.
- **Edge cases to cover**: `N=0`, `N=1`, non-power-of-2 (1000 / 1023 / 1025), `> 1M` (index overflow),
  and special values (NaN / Inf / denormal / ±0) where the math admits them.
- **GPU-side timing** (for `sycl-optimization`, not correctness) uses a profiling queue:
  `sycl::queue q{gpu_selector_v, sycl::property::queue::enable_profiling{}}`, then read
  `event.get_profiling_info<info::event_profiling::command_start>()` / `…command_end>()` (ns) after a
  warm-up run.

## Migration do/don't (Xe2/Xe3) `[ARCH]`
- DO use `int32` index math; DON'T use 64-bit division/modulo in hot loops.
- DON'T force `[[intel::reqd_sub_group_size(16)]]` on simple kernels — only for collectives / XMX / hand-tuned reductions.
- DO map naturally-multidimensional problems to `nd_range<2/3>` to avoid index decomposition.
- DO keep the CUDA reference accessible to compare intent.
- Optimization is the `sycl-optimization` skill's job, not migration's — produce a *correct* kernel first.
