# Triton → SYCL 2020 Migration Patterns

Hand-write SYCL from the **algorithm** a Triton kernel expresses — not from its Python syntax. A
Triton `@triton.jit` kernel is already a *tiled, block-programming* description of the computation:
one program instance (`tl.program_id`) owns a tile, addresses it with block pointers + masks, and
uses `tl.*` collectives. Recover that tiling intent, then express it in SYCL 2020.
`[SPEC]` Triton language: https://triton-lang.org/main/python-api/triton.language.html

> Companion shards: use `sycl-kernel-patterns.md` for the underlying SYCL API surface, and
> `sycl-tla-patterns.md` when the kernel is tiled tensor algebra (GEMM/attention) that should target
> sycl-tla instead of plain SYCL. For how Triton itself lowers to Xe, the `intel-xpu-backend-for-triton`
> reference repo is the authority (see `intel-gpu-software-repos.md`).

## Mental-model mapping `[SPEC]`
A Triton program instance ≈ a **SYCL work-group** (it owns a whole tile and loops over sub-tiles),
*not* a single work-item. The per-instance vector lanes (`BLOCK_SIZE` elements) map to the
work-group's work-items (and, for collectives/`tl.dot`, its sub-groups).

| Triton | SYCL 2020 |
|--------|-----------|
| `@triton.jit def k(...)` + `k[grid](...)` | `q.parallel_for(nd_range<N>(global, local), [=](nd_item<N> it){...})` |
| `pid = tl.program_id(0)` | `it.get_group(0)` (one work-group == one Triton program) |
| `tl.num_programs(0)` | `it.get_group_range(0)` |
| grid lambda `grid = lambda META: (cdiv(n, META['BLOCK']),)` | host-side `global = num_groups * local`; compute `num_groups` the same way |
| `BLOCK_SIZE: tl.constexpr` | compile-time constant (template param / `constexpr`); pick work-group size from it |
| `offs = tl.arange(0, BLOCK)` | `it.get_local_id(0)` (+ tile base) — the lane index within the tile |
| `num_warps` / `num_stages` (autotune) | work-group size / sub-group count + manual pipelining (see below) |

## Block pointers, loads/stores, masking `[SPEC]`
Triton hides addressing behind block pointers and masks; in SYCL you compute the addresses yourself.

| Triton | SYCL 2020 |
|--------|-----------|
| `offs = pid*BLOCK + tl.arange(0, BLOCK)` | `i = group*BLOCK + it.get_local_id(0)` |
| `mask = offs < N` | `if (i < N) { ... }` bounds guard |
| `x = tl.load(X + offs, mask=mask, other=0.0)` | `float x = (i < N) ? X[i] : 0.0f;` |
| `tl.store(Y + offs, y, mask=mask)` | `if (i < N) Y[i] = y;` |
| `tl.make_block_ptr(base, shape, strides, offsets, block_shape, order)` | explicit row/col index math over a USM pointer with the same strides; 2-D tiles → `nd_range<2>` |
| `tl.load(block_ptr, boundary_check=(0,1))` | explicit bounds guards on both tile dims |
| `tl.advance(block_ptr, (0, BLOCK_K))` | increment the running column/row base each K-step of the loop |

- Triton's implicit **vectorization** over `BLOCK` becomes either one element per work-item, or an
  explicit short loop / `sycl::vec` per work-item when a lane handles several elements. Match Triton's
  effective element-per-thread ratio; don't blindly make `BLOCK` work-items.
- `other=` fill on masked loads → the `? :` default value (usually `0`).

## Reductions & scans `[SPEC]`
Triton reduces *within a program's tile* (across its lanes). That is a **work-group reduction** in SYCL.

| Triton | SYCL 2020 |
|--------|-----------|
| `tl.sum(x, axis=0)` (over the block) | `sycl::reduce_over_group(it.get_group(), x, sycl::plus<>())` |
| `tl.max/min(x, axis=0)` | `reduce_over_group(group, x, sycl::maximum<>()/minimum<>())` |
| `tl.cumsum(x, axis=0)` | `sycl::inclusive_scan_over_group(group, x, sycl::plus<>())` |
| axis reduction over the *other* (non-tiled) dim | plain sequential loop inside the work-item |
| softmax-style `x - tl.max(x); tl.exp; / tl.sum` | two work-group reductions (max, then sum) with a barrier between, exactly like the SLM reduction pattern in `sycl-kernel-patterns.md` |

For big tiles, do sub-group reduce → SLM → work-group reduce (see the two-level reduction pattern).

## Matrix multiply — `tl.dot` `[ARCH]`
`tl.dot(a, b)` is the crux of most performance-relevant Triton kernels (GEMM, attention). On Intel it
maps to **XMX/DPAS**. Two routes (both actively maintained and benchmark-verified — never hand-roll
`joint_matrix`):
- **Library call:** **oneMKL** `gemm` for a standalone standard GEMM, or **oneDNN** for a fused
  GEMM+epilogue. Fine for a single `tl.dot` with no custom tiling.
- **sycl-tla (preferred for tiled GEMM/attention):** when the kernel is a K-loop of `tl.dot` with an
  epilogue (bias, activation, softmax) — i.e. a real tiled matmul/attention — target sycl-tla's
  collective tiled MMA. See `sycl-tla-patterns.md` for the decision rule and skeleton.

| Triton | Intel / SYCL |
|--------|--------------|
| `acc = tl.dot(a, b, acc)` in a K-loop | a **sycl-tla** collective MMA, or an oneMKL/oneDNN GEMM call |
| `tl.dot(..., out_dtype=tl.float32)` | fp32 accumulate — match accumulate precision |
| fp16/bf16 inputs, fp32 acc | bf16/fp16 inputs, fp32 accumulate — the native Xe XMX shape |

## Atomics `[SPEC]`
| Triton | SYCL 2020 |
|--------|-----------|
| `tl.atomic_add(P + offs, v, mask)` | `sycl::atomic_ref<T, memory_order::relaxed, memory_scope::device, access::address_space::global_space>(P[i]) += v;` (guarded by the mask) |
| `tl.atomic_max/min/cas` | `atomic_ref(...).fetch_max/fetch_min / compare_exchange_strong` |

## Math & dtypes `[SPEC]`
| Triton | SYCL 2020 |
|--------|-----------|
| `tl.exp`, `tl.log`, `tl.sqrt`, `tl.sigmoid` | `sycl::exp/log/sqrt`; sigmoid = `1/(1+sycl::exp(-x))` |
| `tl.math.*` fast variants | `sycl::native::*` (lower precision — only where Triton used the fast path) |
| `tl.float16` / `tl.bfloat16` | `sycl::half` / `sycl::ext::oneapi::bfloat16` |
| `tl.constexpr` meta-params | template parameters or `constexpr` — resolve at compile time, don't pass as runtime args |
| `tl.dot` accumulator dtype | keep the SYCL accumulator at the same (usually fp32) precision |

## Autotuning & launch config `[ARCH]`
- `@triton.autotune(configs=[triton.Config({'BLOCK_M':..,'BLOCK_N':..}, num_warps=w, num_stages=s)])`
  is Triton's search over tile sizes + warp count + pipeline depth. **Do not** port the autotuner;
  instead pick a sensible Xe default (work-group size, sub-group count, tile shape) from
  `intel-gpu-hardware.md` / `sycl-optimization-catalog.md`, migrate for **correctness**, then let the
  `sycl-optimization` skill search tile sizes on the real device.
- `num_warps` → number of sub-groups per work-group (`work-group size / sub-group width`). Intel
  sub-group is 16 (also 32) — don't assume 32.
- `num_stages` → software pipelining / double-buffering of SLM loads; a *manual* optimization in SYCL,
  deferred to `sycl-optimization` (don't attempt during migration).

## Recovering the reference (CPU oracle) for a Triton kernel `[SPEC]`
The accuracy model is identical to CUDA (see `sycl-migration`): the **CPU reference is the oracle**,
SYCL is compared to it live. Triton specifics:
- **Reuse the project's PyTorch reference (preferred).** Triton kernels almost always have a
  companion eager-PyTorch implementation and a `torch.testing.assert_close(triton_out, torch_out)`
  test. That torch path *is* a validated reference — point the SYCL test at the same math, seed, and
  tolerance.
- **One-time cross-check:** if a Triton/CUDA-capable host is reachable, run the original Triton kernel
  once against the CPU reference on seeded inputs and record `reference.cuda_crosscheck`-equivalent
  (`crosscheck: pass`) in the kernel detail. If no such host, the CPU/torch reference stands alone.
- Match Triton's dtype/accumulation when setting tolerance (bf16/fp16 tiles + fp32 acc → looser rtol).

## Do / don't `[ARCH]`
- DO map one Triton program → one SYCL work-group; recover the tile shape from `BLOCK_*` constexprs.
- DO translate masks to bounds guards; never read/write out of range.
- DON'T port the autotuner or `num_stages` pipelining during migration — get it correct first, then
  hand it to `sycl-optimization`.
- DON'T assume warp size 32 — Intel sub-group is 16 (also 32).
- DON'T hand-roll a big tiled `tl.dot` GEMM/attention in plain SYCL if it's really tiled tensor
  algebra — evaluate sycl-tla first (`sycl-tla-patterns.md`).
- DO keep the Triton source (and its torch reference) open to preserve intent.
