# SYCL Optimization Catalog (Intel Xe2/Xe3 — Battlemage Arc Pro B-series + Crescent Island)

Optimization knowledge for the `sycl-optimization` skill. Apply **one strategy per trial**, gate on
correctness, keep only measured wins, and record every trial
(see `sycl-optimization/references/search-strategy.md`).

This document has **two independent parts** — keep them separate:

- **Part 1 — General optimization checklist:** a human-readable, profiling-agnostic list of the
  highest-yield strategies, plus the profiling playbook, proven anti-patterns, and advanced options.
  Read this to learn the space.
- **Part 2 — Optimization strategy lookup table:** the structured decision layer the search loop
  consumes — profiling-signal → strategy rows with priority, impact, and conflict/synergy edges. Feed
  this to the loop.

Both parts share the same **strategy IDs** (e.g. `increase-ilp`, `vectorize-vec4`); the concrete
before→after code for each ID lives in a one-card-per-strategy folder,
`sycl-optimization-strategies/<strategy-id>.md`.

## Official references `[SPEC]`
- oneAPI GPU Optimization Guide: https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/current/overview.html
- SYCL 2020 spec: https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html
- oneMKL / oneDNN documentation for library-backed primitives.

---

# Part 1 — General optimization checklist

A profiling-agnostic tour of the space: the highest-yield strategies, how to profile, what never to try,
and the advanced options. This part is for a human (or the agent) to *understand* the strategies; Part 2
is what the search loop actually consumes.

## Candidate strategies (roughly in order)

The numbers are `[EMPIRICAL]` gains observed on Intel GPUs and are indicative, not guaranteed —
always measure on the target device.

1. **Work-group size tuning** — 256 is a common sweet spot; 128→256 can yield 2.0–2.3× on
   memory-bound ops. Use adaptive sizing for small dimensions.
2. **Native math** — replace `sycl::exp/sin/cos/pow/log` with `sycl::native::*` on compute-bound
   kernels (~1.3×) when precision allows.
3. **Compile-time specialization** — template on compile-time dims + switch dispatch → bounds-check
   elimination and full unrolling (up to ~3× on specialized paths).
4. **Vec4 vectorization** — process 4 elements/work-item via `sycl::float4` loads/stores (~1.5–2× on
   bandwidth-bound element-wise).
5. **Loop unrolling** — `#pragma unroll 4` on hot inner loops only when trip count ≥ 8.
6. **Sub-group reductions** — `sycl::reduce_over_group()` instead of SLM reductions — but profile vs
   manual shuffle (see anti-patterns).
7. **Float2 accumulation** — compute sum and sum-of-squares together in normalization kernels to cut
   passes.
8. **Fast integer division** — replace int64 div/mod with multiply-shift for non-power-of-2 divisors.
9. **Adaptive block sizing** — sub-group size (16) for small dims, 256+ for large.
10. **Multi-column / multi-output per work-group** — raise per-work-item compute, cut launch overhead.
11. **XMX offload** — route dense FP16/BF16/INT8 matmul/conv through **oneDNN** or **sycl-tla**
    (actively maintained, benchmark-verified); do **not** hand-roll XMX via `joint_matrix`.

## Profiling playbook `[SPEC]`
- Capture timeline / occupancy / memory traffic and stalls with `unitrace` (see `sycl-profiler` §4);
  roofline with Intel Advisor (GPU).
- Check for: low occupancy (register spills, oversized WG), uncoalesced memory, excessive 64-bit math,
  idle XMX on matmul, host-device sync bubbles.
- Pin the backend: `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`.

## Proven anti-patterns `[EMPIRICAL: Intel GPU]` (do NOT repeat)

These regressed on Intel Xe and are excluded from candidate ranking. If a trigger tempts one of these,
prefer the mapped alternative; if you retry one to confirm, **log the measured result**.

| Anti-pattern | Result | Root cause | Prefer instead |
|---|---|---|---|
| Explicit SLM caching of read-only vectors | neutral to −43% | Intel L1 already caches read-only access | rely on L1/L3; only stage genuinely reused tiles |
| Manual vectorized 64-bit loads | −8.6% avg | compiler already unrolls; manual adds register pressure | let the compiler; measure `vectorize-vec4` |
| `reduce_over_group` replacing butterfly shuffle | −2.9% avg | manual XOR shuffle can be faster — profile both | benchmark `subgroup-reduction` vs manual shuffle |
| 1D nd_range with index decomposition | up to −48% | index math overhead | use `nd_range<2/3>` direct dimension mapping |
| Hardcoded WG=256 for all inputs | up to −522% | small dims waste work-items | `adaptive-block-size` |
| `size_t` (64-bit) stride arithmetic | −35% | 64-bit int math is slow on Xe | use int32 when dims fit |
| O(n log n) sort for small top-k | −6525% | overhead dominates at small k | O(n·k) insertion sort for small k |
| Replacing `float4`+`sycl::dot()` with scalar | regression | float4+dot generates better SIMD | keep `float4`+`dot` |
| Unrolling loops with small trip count | −11 to −32% | unroll overhead > benefit when count < 8 | only `unroll-hot-loop` when trip count ≥ 8 |
| Hand-rolled `joint_matrix` where a library fits | varies / brittle | reinvents oneDNN/sycl-tla tuning | `offload-xmx-library` |

## Advanced `[SPEC]`
- **oneDNN** for standard GEMM/conv (near-peak); **sycl-tla** (CUTLASS-style tiled) for custom/fused XMX
  a library cannot express. Both are actively maintained and benchmark-verified — prefer them over
  hand-written `joint_matrix` (see `sycl-tla-patterns.md`).
- **SYCL graphs** to amortize launch overhead for repeated kernel sequences (requires
  `ext_oneapi_limited_graph`; incompatible with mid-graph `wait()`).
- **Multi-GPU** via row-wise tensor split; note SYCL graphs are single-device only.

> Record durable Intel Xe2/Xe3 findings (kept/reverted, with measured numbers + environment) in
> `.sycl/state/lessons.md`; promote broadly-true ones into the anti-pattern table above via a PR.

---

# Part 2 — Optimization strategy lookup table

The structured decision layer for the `sycl-optimization` search loop. Each row maps a **diagnosis**
(bottleneck class + dominant `unitrace` stall/counter, in this framework's vocabulary — see
`sycl-profiler` §4) to a concrete **candidate transformation**, with the priority, expected impact,
architecture applicability, and the conflict/synergy edges the search needs to plan combinations.

## How the search loop consumes this catalog

The columns line up with the phases in `search-strategy.md`:

| Search-loop step | Catalog column(s) it reads |
|---|---|
| ANALYZE — rank candidates from the DIAGNOSE root cause | **Bottleneck class** + **Trigger signal** → filter rows; **Priority** → order them |
| APPLY — make one change | **Strategy** → open its card `sycl-optimization-strategies/<strategy-id>.md` for the before→after transform |
| Phase 2 — combination trials | **Synergizes-with** (pairs to stack) + **Conflicts-with** (pairs to avoid) |
| Phase 3 / exploratory — escape local optima | **Enabling transformations** table |
| DECIDE — keep/revert | **Expected impact** as a sanity check against the measured delta |

## Profiling signals used in triggers `[SPEC]`

Triggers are written in the metric/stall vocabulary this framework already captures (`sycl-profiler`
§4, `unitrace --metric-query` groups + `--stall-sampling`). For the concrete counters behind each
signal — their units, derived ratios, alarm thresholds, and the raw `XVE_STALL_*` names behind the
short stall aliases — see `intel-gpu-hardware-metrics-bmg.md` (Xe2) and `intel-gpu-hardware-metrics-cri.md`
(Xe3P deltas: 4 ALU pipes, `L1Profile` in place of `MemoryProfile`, direct FLOP/lane/GRF counters).

- **Bottleneck class** — from the roofline (`intel-gpu-hardware.md`): memory-bound vs compute-bound vs
  occupancy-bound, refined into the classes below.
- **Dominant stall** — the largest `*Stall` in the `VectorEngineStalls` mix: `SendStall` (memory msg),
  `SbidStall` (scoreboard/async dep), `PipeStall`/`DistStall` (ALU dep / register distance),
  `SyncStall` (barrier), `ControlStall` (branch), `InstrFetchStall` (icache).
- **Occupancy** — `XVE`/thread occupancy % from `ComputeBasic`; register **spills** from the IGC asm
  dump.
- **Cache** — L1/L2/L3 hit rates from `ComputeBasic` / `DeviceCacheProfile` / `L3`.
- **Bandwidth** — achieved read/write bandwidth from `ComputeBasic` / `MemoryProfile`, compared to the
  device peak.
- **XMX idle** — no DPAS instructions in the asm dump on a matmul/conv-shaped loop.

Code for every strategy lives in its card under `sycl-optimization-strategies/<strategy-id>.md`
(before→after transform, correctness invariants, how to verify it landed) — this catalog is the
*index and decision layer*, not a code cookbook.

## Master lookup table

`Priority` = how strongly to prefer the row when its trigger fires (Critical → try first). `Impact` is
`[EMPIRICAL]` on Intel Xe2/Xe3 — **indicative, always measure on the target device**. `Arch`: BMG =
Battlemage/Xe2, CRI = Crescent Island/Xe3P.

### Memory-latency (hide the latency you cannot remove)
| Strategy | Trigger signal | Priority | Impact | Arch | Conflicts | Synergizes |
|---|---|---|---|---|---|---|
| `increase-ilp` | `SbidStall` dominant (or `SendStall` >40% with spare registers) | Critical | 15–40% | Both | `reduce-register-pressure`, `switch-grf-mode:small` | `software-pipeline`, `vectorize-vec4` |
| `software-pipeline` | `SbidStall` dominant, few in-flight async ops | High | 10–30% | Both | `reduce-register-pressure` | `increase-ilp`, `prefetch-to-slm` |
| `prefetch-to-slm` | L2/L3 hit rate < 0.60 **and** a tile loop with predictable reuse | High | 20–50% | Both | `reduce-slm-allocation`, `tune-work-group-size` | `slm-cache-reuse`, `tile-data-access` |
| `slm-cache-reuse` | L2/L3 hit rate < 0.60 **and** same neighborhood reused across a work-group | High | 20–60% | Both | `reduce-slm-allocation`, `pad-slm-arrays` | `prefetch-to-slm`, `tile-data-access` |
| `prefetch-global-to-l3` | `SbidStall` dominant with **indirect / pointer-chasing / gather** global loads (no tile reuse) | High | 10–30% | Both | `prefetch-to-slm`, `slm-cache-reuse` (if the set is already staged) | `increase-ilp`, `software-pipeline` |

### Memory-bandwidth (move fewer / fatter bytes)
| Strategy | Trigger signal | Priority | Impact | Arch | Conflicts | Synergizes |
|---|---|---|---|---|---|---|
| `vectorize-vec4` | `SendStall` dominant, scalar element-wise loads/stores | Critical | 1.5–4× BW | Both | `reduce-register-pressure`, `switch-grf-mode:small` | `coalesce-memory-access`, `increase-ilp` |
| `coalesce-memory-access` | high traffic + low effective BW; neighboring lanes hit strided addresses | High | 2–8× on bad strides | Both | `uniform-control-flow` (if bucketing kills locality) | `vectorize-vec4`, `optimize-access-pattern` |
| `vectorize-low-bitwidth-stores` | scalar stores of sub-32-bit types (bf16/fp16/int8/fp8/int4); asm shows `d16u32`/`d8u32` | Critical | 3–5× bf16, 4–17× int8 | Both | none (orthogonal) | `reduce-data-type-width` |
| `reduce-data-type-width` | bandwidth-bound (≥70% peak) and FP32 used where FP16/BF16 is numerically OK | Medium | up to 2× BW | Both | unsupported XMX input type only | `offload-xmx-library`, `vectorize-vec4` |
| `fuse-passes` | memory-bound, multiple kernels re-read the same arrays | High | 1.5–3× | Both | none | `float2-accumulation` |
| `float2-accumulation` | normalization/reduction making two passes (sum + sum-of-squares) | Medium | 1.2–1.5× | Both | none | `fuse-passes`, `subgroup-reduction` |
| `assume-aligned` | `vectorize-vec4` / low-bitwidth widening didn't emit wide loads; asm still shows narrow messages despite aligned data | Medium | enables 1.5–4× | Both | none | `vectorize-vec4`, `vectorize-low-bitwidth-stores`, `coalesce-memory-access` |

### Compute-bound (do less ALU / use the right engine)
| Strategy | Trigger signal | Priority | Impact | Arch | Conflicts | Synergizes |
|---|---|---|---|---|---|---|
| `offload-xmx-library` | matmul/conv shape **and** XMX idle (no DPAS in asm) — route to **oneDNN / sycl-tla / oneMKL**, do **not** hand-roll `joint_matrix` | Critical | 5–10× over XVE | Both | `balance-alu-pipes` on the same loop | `reduce-data-type-width` |
| `native-math` | compute-bound with `sycl::exp/sin/cos/pow/log` in a hot loop, precision permitting | High | ~1.3× | Both | none | `unroll-hot-loop` |
| `unroll-hot-loop` | `PipeStall`/`DistStall` dominant, hot inner loop with trip count ≥ 8 | High | 10–25% | Both | `reduce-register-pressure`, `InstrFetchStall` risk | `increase-ilp`, `native-math` |
| `compile-time-specialization` | dims known at launch; runtime bounds checks / dynamic loop bounds in hot path | High | up to 3× on specialized paths | Both | none | `unroll-hot-loop` |
| `fast-integer-division` | int64 div/mod on non-power-of-2 divisors in index math | Medium | 5–20% | Both | none | `balance-alu-pipes` |
| `balance-alu-pipes` | one ALU-pipe utilization metric ≫ the others | Medium | 10–25% | Both (3 pipes BMG / 4 CRI) | `offload-xmx-library` on GEMM loops | `increase-ilp`, `eliminate-branches` |
| `restrict-kernel-args` | pointer-heavy kernel; IGC asm shows redundant reloads / conservative vectorization from assumed pointer aliasing | High | 5–20% | Both | none | `vectorize-vec4`, `increase-ilp`, `unroll-hot-loop` |

> **Library-first for XMX.** This framework does **not** hand-roll XMX via `joint_matrix`. Route dense
> FP16/BF16/INT8 GEMM/conv through **oneDNN** or **sycl-tla** (actively maintained, benchmark-verified);
> see `sycl-tla-patterns.md`. Only if a library genuinely cannot express a fused/custom shape do you
> reach for `joint_matrix`, and then follow the K-loop double-buffering rules in `sycl-tla-patterns.md`.

### Occupancy-bound (more resident threads)
| Strategy | Trigger signal | Priority | Impact | Arch | Conflicts | Synergizes |
|---|---|---|---|---|---|---|
| `reduce-register-pressure` | thread occupancy < 30% **and** spills in asm dump | Critical | 20–50% | Both | `increase-ilp`, `vectorize-vec4`, `unroll-hot-loop` | `tune-work-group-size`, `switch-grf-mode` |
| `switch-grf-mode` | occupancy < 50% with high register pressure (large↔small GRF) | High | 10–30% | Both | `increase-ilp`, `vectorize-vec4` in `small` | `reduce-register-pressure`, `tune-work-group-size` |
| `tune-work-group-size` | occupancy < 50% or dispatch-queue resource stalls; 256 is the common sweet spot | Medium | 5–20% | Both | `xmx-block-size`, `reduce-slm-allocation` | `reduce-register-pressure`, `adaptive-block-size` |
| `adaptive-block-size` | one hardcoded WG size across very different shapes; small dims waste lanes | Medium | up to 2× on small dims | Both | none | `tune-work-group-size` |
| `reduce-slm-allocation` | SLM/WG > 32 KB and only one WG fits per Xe-core | Medium | 10–40% when occupancy-limited | Both | `prefetch-to-slm`, `slm-cache-reuse`, `double-buffer-slm`, `pad-slm-arrays` | `tune-work-group-size` |

### Synchronization (`SyncStall`)
| Strategy | Trigger signal | Priority | Impact | Arch | Conflicts | Synergizes |
|---|---|---|---|---|---|---|
| `double-buffer-slm` | `SyncStall` dominant in a tiled load→compute loop | Critical | 20–40% | Both | `reduce-slm-allocation`, `switch-grf-mode:small` | `prefetch-to-slm` |
| `reduce-barriers` | `SyncStall` > 20% of stall time, or barrier count high for the algorithm | High | 10–30% | Both | none | `subgroup-reduction`, `lock-free-atomics` |
| `subgroup-reduction` | work-group reduction where scope is actually sub-group local | Medium | 10–30% (profile vs shuffle) | Both | see anti-patterns (`reduce_over_group` can lose to manual shuffle) | `reduce-barriers` |
| `lock-free-atomics` | `SyncStall` high with global atomics / elevated L3 atomic traffic | Medium | 5–20× on reduction-heavy | Both | none | `reduce-barriers`, `coalesce-memory-access` |

### Cache (raise hit rate / fix layout)
| Strategy | Trigger signal | Priority | Impact | Arch | Conflicts | Synergizes |
|---|---|---|---|---|---|---|
| `tile-data-access` | L2/L3 hit rate < 0.60 traversing a regular 2D/3D neighborhood | Critical | 20–60% | Both (esp. BMG GDDR6) | none | `prefetch-to-slm`, `slm-cache-reuse` |
| `pad-slm-arrays` | SLM bank-conflict rate high (`MemoryProfile` on BMG; heuristic on CRI) | High | 10–25% | BMG / heuristic CRI | `reduce-slm-allocation` | `slm-cache-reuse` |
| `optimize-access-pattern` | good hit rate but low effective BW due to AoS / GPU-unfriendly layout | Medium | 10–40% | Both | none | `coalesce-memory-access`, `tile-data-access` |
| `cache-control-hints` | streaming write-once output polluting cache, or read-once data evicting reused data; L2/L3 hit rate hurt by streaming traffic | Medium | 10–30% | Both | `prefetch-to-slm` / `slm-cache-reuse` (do not mark reused data uncached) | `optimize-access-pattern`, `tile-data-access` |

### Control-flow (`ControlStall` / divergence)
| Strategy | Trigger signal | Priority | Impact | Arch | Conflicts | Synergizes |
|---|---|---|---|---|---|---|
| `eliminate-branches` | `ControlStall` high or control-instruction ratio > 0.15 | High | 10–25% | Both | none | `uniform-control-flow`, `balance-alu-pipes` |
| `uniform-control-flow` | SIMD efficiency < 0.70 / clear divergence symptoms | Medium | 10–30% | Both | `coalesce-memory-access` if re-bucketing hurts locality | `eliminate-branches` |

### Launch / dispatch overhead
| Strategy | Trigger signal | Priority | Impact | Arch | Conflicts | Synergizes |
|---|---|---|---|---|---|---|
| `multi-output-per-item` | tiny kernels, launch/dispatch overhead visible vs kernel time | Medium | 1.2–2× | Both | `reduce-register-pressure` | `increase-ilp` |
| `sycl-graphs` | repeated sequence of small kernels dominated by submit overhead | Medium | overhead-bound only | Both | mid-graph `wait()`; single-device only | none |

### Host / dispatch overlap (host-side)
| Strategy | Trigger signal | Priority | Impact | Arch | Conflicts | Synergizes |
|---|---|---|---|---|---|---|
| `overlap-copy-compute` | host–device-bound: timeline shows H2D/D2H serialized with kernels, GPU idle during copies | High | up to 2× when transfer-bound | Both | none (host-side) | `sycl-graphs` |

## Recommended ordering (classifier default) `[EMPIRICAL]`

When several rows fire, resolve conflicts in this order (mirrors the reference harness's
`classify_bottleneck` + combination order):

0. **Take the near-free codegen wins first** — `restrict-kernel-args` and `assume-aligned` are cheap,
   correctness-preserving hints that often unlock the vectorize/ILP rows below; apply them before
   restructuring anything.
1. **Fix occupancy limiters first** — `reduce-register-pressure`, `switch-grf-mode`,
   `tune-work-group-size`, `reduce-slm-allocation`. Everything downstream needs threads resident.
2. **Kill the dominant stall** — `increase-ilp`/`software-pipeline` (`SbidStall`),
   `vectorize-vec4`/`coalesce-memory-access` (`SendStall`), `double-buffer-slm`/`reduce-barriers`
   (`SyncStall`), `unroll-hot-loop`/`native-math` (`PipeStall`).
3. **Improve reuse and layout** — `tile-data-access`, `prefetch-to-slm`, `slm-cache-reuse`,
   `optimize-access-pattern`.
4. **For matmul/conv, go to the XMX library path** — `offload-xmx-library` (oneDNN/sycl-tla/oneMKL)
   instead of ALU tuning on the same loop.
5. **Control-flow cleanup last** unless divergence is clearly the root cause.

## Combination trials (Phase 2) — synergy pairs

Stack these only after each is an individual win (or is a known enabling transform), from the best
commit, marked `combination: true`. Interactions are usually **sub-additive** — do not expect the
product of the speedups.

- `increase-ilp` + `vectorize-vec4` / `software-pipeline` (more independent in-flight work)
- `coalesce-memory-access` + `vectorize-vec4` / `optimize-access-pattern` (fatter, aligned transactions)
- `prefetch-to-slm` + `slm-cache-reuse` + `tile-data-access` (staged on-chip reuse)
- `reduce-register-pressure` + `tune-work-group-size` + `switch-grf-mode` (co-tune the occupancy knobs)
- `double-buffer-slm` + `reduce-barriers` (cut both barrier count and barrier idle)
- `reduce-data-type-width` + `offload-xmx-library` / `vectorize-vec4` (halved bytes → more per vector)

**Do not** combine a memory-bound fix with an unrelated compute-bound fix unless the kernel is genuinely
balanced (arithmetic intensity near the ridge point). Never stack two rows that list each other under
**Conflicts** without re-profiling between them.

## Enabling transformations (exploratory acceptance) — Phase 3

Some strategies restructure code so *follow-ups* become effective, and may **regress or be neutral
alone**. Allow a bounded regression (simplicity/exploratory criterion in `search-strategy.md`) only to
reach the follow-up; revert if the chain does not pay off.

| Enabling strategy | Unlocks | Why it can regress alone |
|---|---|---|
| `optimize-access-pattern` (AoS→SoA) | `coalesce-memory-access`, `vectorize-vec4` | layout conversion overhead until a downstream strategy exploits it |
| `reduce-data-type-width` (FP32→FP16/BF16) | `offload-xmx-library`, `vectorize-vec4` (2× elems/vector) | precision handling (scaling/accumulator) adds work first |
| `tile-data-access` | `prefetch-to-slm`, `slm-cache-reuse`, `pad-slm-arrays` | tiling alone adds barriers + SLM before reuse pays off |
| `switch-grf-mode` (large→small) | `tune-work-group-size` at higher occupancy | more spills temporarily; only helps once WG size is re-tuned |
| `assume-aligned` | `vectorize-vec4`, `vectorize-low-bitwidth-stores` | neutral alone — only pays once a widening strategy emits the wide loads it permits |
