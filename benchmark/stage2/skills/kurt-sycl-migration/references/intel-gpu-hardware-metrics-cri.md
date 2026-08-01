# Intel Crescent Island (CRI / Xe3P) Hardware Metrics Reference

Canonical catalog of the Intel `unitrace` hardware performance metric groups and counters available on
**Crescent Island (CRI / Xe3P)** GPUs, with the deltas versus Battlemage (BMG). This is the
*counter-level* companion to the group names used in `sycl-profiler` §4 and the trigger vocabulary in
`sycl-optimization-catalog.md`; see `intel-gpu-hardware-metrics-bmg.md` for the BMG baseline and the
shared framework stall-name crosswalk.

> **CRI is not just "Battlemage plus a few counters".** It adds a **4th ALU pipe**, expands **XMX data
> types**, introduces direct **FLOP / lane / GRF occupancy** counters, adds **page-size-aware TLB**
> metrics, exposes **copy-engine** activity, and uses **`L1Profile` instead of BMG `MemoryProfile`**.
> These deltas are why the catalog marks some rows "3 pipes BMG / 4 CRI" and "`MemoryProfile` on BMG;
> heuristic on CRI".

## Provenance
- `[ARCH]` — architectural facts (4 ALU pipes, XMX types, copy engine). Stable across CRI boards.
- `[SPEC]` — counter definitions and metric-group membership from Intel Metrics Discovery / `unitrace`.
- `[VERIFY]` — the exact counters exposed depend on the installed `unitrace`, Metrics Discovery
  package, driver, and firmware; `L1Profile` in particular varies by system. Confirm with
  `unitrace --metric-list` and record confirmed values in `.sycl/state/lessons.md`.
- Thresholds are `[EMPIRICAL]` starting heuristics — always re-measure on the target device.

## Framework stall-name crosswalk `[SPEC]`
CRI uses the same `VectorEngineStalls` counter names as BMG, so the alias mapping in
`intel-gpu-hardware-metrics-bmg.md` → "Framework stall-name crosswalk" applies unchanged
(`SendStall`↔`XVE_STALL_SENDWR`, `SbidStall`↔`XVE_STALL_SBID`, `SyncStall`↔`XVE_STALL_BARRIER`, …).

## Quick Start `[SPEC]`

- Discover the exact metric groups on your installed system with:

  ```bash
  unitrace --metric-list
  ```

- Start with **`ComputeBasic`** for first-pass diagnosis.
- Add **`VectorEngineProfile`** for instruction mix / XMX analysis.
- Add **`VectorEngineStalls`** for stall root-cause breakdown.
- Use **`L1Profile`** for CRI-specific L1 / memory-path deep dives (replaces BMG `MemoryProfile`).

## CRI vs BMG at a Glance `[ARCH]`

| Feature | BMG (Xe2) | CRI (Xe3P) |
|---------|-----------|------------|
| ALU Pipes | 3 (ALU0-2) | 4 (ALU0-3) |
| XMX INT2 | ✓ | ✗ |
| XMX FP4/FP8 | ✗ | ✓ |
| XMX FP64 | ✗ | ✓ |
| BF16 ALU counting | ✗ | ✓ |
| FLOP counters | ✗ | ✓ (FP32, FP64) |
| Lane utilization | ✗ | ✓ |
| GRF occupancy | ✗ | ✓ |
| TLB page sizes | ✗ | ✓ |
| L3 byte counters | Only in VEProfile | In ComputeBasic |
| Copy engine metrics | ✗ | ✓ |
| Memory profile set | MemoryProfile | L1Profile |

## Architecture Notes `[ARCH]`

### What is new on CRI

- **4 ALU pipes** per XVE: `ALU0`, `ALU1`, `ALU2`, `ALU3`
- **Broader XMX datatype coverage**: `INT4`, `INT8`, `FP4`, `FP8`, `FP16`, `BF16`, `FP64`
- **Direct FLOP counters**: `XVE_OP_FP32`, `XVE_OP_FP64`
- **SIMD efficiency counters**: `XVE_INST_LANE_ENABLED`, `XVE_INST_LANE_TOTAL`
- **Register-pressure visibility**: `XVE_GRFBLOCK_OCCUPANCY_ALL`
- **TLB page-size breakdown**: `1G`, `2M`, `64K`, `4K`
- **Copy engine telemetry** in `ComputeBasic`
- **Xe3P server/HPC orientation**

### Implications for analysis

- A single ALU pipe now contributes **25%** of peak pipe issue bandwidth, not **33%** as on BMG — read
  the catalog's `balance-alu-pipes` trigger with a 4-pipe mental model.
- `XVE_MULTIPLE_PIPE_ACTIVE` is more valuable on CRI because there are **4** schedulable ALU pipes.
- FLOP estimation can rely on **direct counters** instead of instruction-count heuristics.
- SIMD under-utilization can be measured directly using lane counters (backs the catalog's
  `uniform-control-flow` trigger — SIMD efficiency < 0.70).
- Register pressure can be diagnosed directly through GRF block occupancy rather than inferred from
  spills in the asm dump alone.

## Available Metric Groups `[SPEC]`

| Metric group | Use it for | Notes |
|------------|------------|-------|
| `ComputeBasic` | First-pass overview | Best default set; includes many CRI-only counters |
| `VectorEngineProfile` | Instruction mix, ALU/XMX datatype mix | Includes new BF16 / FP4 / FP8 / FP64 XMX visibility |
| `VectorEngineStalls` | Stall root-cause analysis | Same core categories as BMG plus extra thread-dispatch metrics |
| `L1Profile` | CRI-specific L1 / memory-path investigation | Replaces BMG `MemoryProfile`; exact metric list can vary by system `[VERIFY]` |

## Which metric group when `[SPEC]`

Same flow as BMG (open with `VectorEngineStalls` + `ComputeBasic`, then pull one follow-up group), but
CRI folds several signals that need a separate group or the IGC asm dump on BMG **directly into
counters**, so some rows resolve without a second run.

| You already know (roofline + first pass) | Query next | Read these counters / ratios | Then see strategy |
|---|---|---|---|
| Nothing yet — first pass | `VectorEngineStalls` + `ComputeBasic` | Active % vs each `*Stall` %; EU active %, thread occupancy, L3 bytes + copy-engine activity (both in CRI `ComputeBasic`) | branch to a row below |
| Memory-bound roofline / dominant `SendStall` | **`L1Profile`** (replaces BMG `MemoryProfile`) | L1/L2/L3 hit rates, DRAM bytes, TLB page-size mix (`4K`/`64K`/`2M`/`1G`) | `coalesce-memory-access`, `vectorize-vec4`, `tile-data-access` |
| Low cache hit / L3 attribution | `DeviceCacheProfile` (+ `L3`); L3 bytes also in `ComputeBasic` | L3 hit rate, per-client L3 traffic | `prefetch-to-slm`, `tile-data-access` |
| Compute-bound / ALU-pipe imbalance / XMX idle | `VectorEngineProfile` | instruction mix, `ALU0`-`ALU3` balance (4 pipes → 25% each), XMX datatype mix incl. FP4/FP8/FP64; direct `XVE_OP_FP32`/`XVE_OP_FP64` FLOPs | `balance-alu-pipes`, `offload-xmx-library` (route matmul to `joint_matrix`/oneMKL/oneDNN) |
| SIMD under-utilization / divergence | `ComputeBasic` (lane counters) | `XVE_INST_LANE_ENABLED` / `XVE_INST_LANE_TOTAL` → SIMD efficiency (< 0.70 alarms) | `uniform-control-flow` |
| Suspect register pressure / low occupancy | `ComputeBasic` (GRF occupancy) | `XVE_GRFBLOCK_OCCUPANCY_ALL`, thread occupancy % (no asm dump needed) | `reduce-register-pressure` |
| Low ILP — dominant `PipeStall`/`DistStall` or `SbidStall` | `VectorEngineStalls` (already) + IGC asm | stalling IPs + reason; dependency distance | `increase-ilp`, `software-pipeline` |
| Over-synchronization — dominant `SyncStall` | `VectorEngineStalls` (already) | barrier stall % | `double-buffer-slm`, fewer/rebalanced barriers |

> **CRI specifics:** the memory deep-dive group is **`L1Profile`**, not `MemoryProfile`, and its exact
> metric list varies by system (`unitrace --metric-list` to confirm) `[VERIFY]`. SLM bank-conflict rate
> is **not** a first-class counter as on BMG — the catalog's `pad-slm-arrays` trigger is heuristic on
> CRI. SIMD efficiency (lane counters), register pressure (`XVE_GRFBLOCK_OCCUPANCY_ALL`), and FLOPs
> (`XVE_OP_FP32`/`XVE_OP_FP64`) are **direct counters** in `ComputeBasic`/`VectorEngineProfile`, so
> those rows need no IGC asm dump.

---

## `ComputeBasic` `[SPEC]`

The primary overview set for CRI, and where many **CRI-vs-BMG differences** first appear.

### GPU Overview

| Metric | Unit | Meaning |
|--------|------|---------|
| `GpuTime` | ns | GPU time accumulated by the sampled workload |
| `GpuCoreClocks` | cycles | Core clock cycles during the measured interval |
| `AvgGpuCoreFrequencyMHz` | MHz | Average core frequency during execution |

- Useful derived: `AvgGpuCoreFrequencyMHz ≈ GpuCoreClocks / GpuTime * 1e3`

### Front End

| Metric | Unit | Meaning | CRI note |
|--------|------|---------|----------|
| `GPU_BUSY` | % | Overall GPU busy time | Same general meaning as BMG |
| `COMMAND_PARSER_COMPUTE_ENGINE_BUSY` | % | Compute command parser busy | Useful for compute-focused workloads |
| `COMMAND_PARSER_COPY_ENGINE_BUSY` | % | Copy engine parser busy | Important for transfer-heavy workloads |
| `COMMAND_PARSER_FLUSH_COUNT` | events | Front-end flush events | High counts may indicate sync / submission overhead |

> **Difference from BMG:** `COMMAND_PARSER_RENDER_ENGINE_BUSY` is **not** present in CRI `ComputeBasic`.

### Thread Dispatcher

| Metric | Unit | Meaning | CRI note |
|--------|------|---------|----------|
| `ASYNC_GPGPU_THREADGROUP_COUNT` | events | Asynchronously dispatched thread groups | CRI exposes the async variant |
| `ASYNC_GPGPU_THREAD_EXIT_COUNT` | events | Thread exits / retirements | Helps relate dispatch to completion |
| `GPGPU_DISPATCH` | % | Dispatcher activity for GPGPU work | Higher values indicate steady front-end feed |

> **Difference from BMG:** `GPGPU_THREADGROUP_COUNT` is absent; CRI uses
> **`ASYNC_GPGPU_THREADGROUP_COUNT`** instead.

### Vector Engine (XVE)

> **Most important CRI change:** CRI has **4 ALU pipes** (`ALU0`-`ALU3`) per XVE instead of 3 on BMG.

#### Core activity / occupancy

| Metric | Unit | Meaning |
|--------|------|---------|
| `XVE_ACTIVE` | % | XVE active time |
| `XVE_STALL` | % | XVE stalled time |
| `XVE_THREADS_OCCUPANCY_ALL` | % | Thread-slot occupancy across XVEs |
| `XVE_SHARED_FUNCTION_ACCESS_HOLD` | % | Shared-function contention / hold time |

#### ALU / issue / send execution

| Metric | Unit | Meaning | CRI note |
|--------|------|---------|----------|
| `XVE_INST_EXECUTED_ALU0_ALL` | events | Instructions executed on ALU0 | Same family as BMG |
| `XVE_INST_EXECUTED_ALU1_ALL` | events | Instructions executed on ALU1 | Same family as BMG |
| `XVE_INST_EXECUTED_ALU2_ALL` | events | Instructions executed on ALU2 | Same family as BMG |
| `XVE_INST_EXECUTED_ALU3_ALL` | events | Instructions executed on ALU3 | New 4th pipe |
| `XVE_INST_EXECUTED_SEND_ALL` | events | SEND instructions executed | Memory / message traffic proxy |
| `XVE_INST_ISSUED_ALL` | events | Total issued instructions | Useful normalization denominator |
| `XVE_INST_EXECUTED_BARRIER` | events | `sync.bar` / `sync.host` execution | In `ComputeBasic` on CRI |

#### Pipe-level utilization / ILP

| Metric | Unit | Meaning |
|--------|------|---------|
| `XVE_MULTIPLE_PIPE_ACTIVE` | % | 2+ ALU pipes active simultaneously among ALU0-3 |
| `XVE_PIPE_ALU0_AND_ALU1_ACTIVE` | % | ALU0 and ALU1 co-active |
| `XVE_PIPE_ALU0_AND_ALU2_ACTIVE` | % | ALU0 and ALU2 co-active |
| `XVE_INST_EXECUTED_ALU0_ALL_UTILIZATION` | % | ALU0 utilization |
| `XVE_INST_EXECUTED_ALU1_ALL_UTILIZATION` | % | ALU1 utilization |
| `XVE_INST_EXECUTED_ALU2_ALL_UTILIZATION` | % | ALU2 utilization |

> **Interpretation note:** read per-pipe utilization with a **4-pipe mental model**. Over-focus on
> ALU0/1 with low ALU2/3 activity often suggests ILP or compiler scheduling limitations.

#### Instruction fetch / front-end quality

| Metric | Unit | Meaning |
|--------|------|---------|
| `ICACHE_HIT` | events | Instruction-cache hits |
| `ICACHE_MISS` | events | Instruction-cache misses |

#### CRI-only XVE counters

| Metric | Unit | Meaning | Why it matters |
|--------|------|---------|----------------|
| `XVE_GRFBLOCK_OCCUPANCY_ALL` | events | GRF register-block occupancy | Direct register-pressure visibility |
| `XVE_INST_LANE_ENABLED` | events | Active SIMD lanes enabled | Numerator for SIMD efficiency |
| `XVE_INST_LANE_TOTAL` | events | Total available lanes after shootdown | Denominator for SIMD efficiency |
| `XVE_OP_FP32` | events | Effective FP32 ops on ALU0+ALU1 | Direct FP32 FLOP counting |
| `XVE_OP_FP64` | events | Effective FP64 ops on ALU0+ALU1 | Direct FP64 FLOP counting |

- **SIMD efficiency** = `XVE_INST_LANE_ENABLED / XVE_INST_LANE_TOTAL`
- **FP32 FLOPS** = `XVE_OP_FP32 / GpuTime`
- **FP64 FLOPS** = `XVE_OP_FP64 / GpuTime`
- **ICache hit rate** = `ICACHE_HIT / (ICACHE_HIT + ICACHE_MISS)`

### L1 Cache / SLM

| Metric | Unit | Meaning |
|--------|------|---------|
| `LOAD_STORE_CACHE_ACCESS` | events | L1 load/store cache accesses |
| `LOAD_STORE_CACHE_HIT` | events | L1 load/store cache hits |
| `LOAD_STORE_CACHE_BYTE_READ` | bytes | Bytes read through L1 load/store path |
| `LOAD_STORE_CACHE_BYTE_WRITE` | bytes | Bytes written through L1 load/store path |
| `LOAD_STORE_CACHE_PARTIAL_WRITE_COUNT` | events | Partial writes / write-fragment activity |
| `SLM_BANK_CONFLICT_COUNT` | events | Shared local memory bank conflicts |
| `SLM_BYTE_READ` | bytes | SLM bytes read |
| `SLM_BYTE_WRITE` | bytes | SLM bytes written |

- **L1 hit rate** = `LOAD_STORE_CACHE_HIT / LOAD_STORE_CACHE_ACCESS`

### Device Cache (L3)

| Metric | Unit | Meaning | CRI note |
|--------|------|---------|----------|
| `L3_HIT` | events | L3 hits | Same core metric as BMG |
| `L3_MISS` | events | L3 misses | Same core metric as BMG |
| `L3_READ` | events | L3 read requests (64B) | Request-count view |
| `L3_WRITE` | events | L3 write requests (64B) | Request-count view |
| `L3_BYTE_READ` | bytes | L3 bytes read | New in `ComputeBasic` |
| `L3_BYTE_WRITE` | bytes | L3 bytes written | New in `ComputeBasic` |
| `L3_ATOMIC_ACCESS` | events | L3 atomic accesses | Useful for atomics-heavy kernels |
| `L3_SUPERQ_FULL` | % | L3 request queue fully occupied | Backpressure indicator |

- **L3 hit rate** = `L3_HIT / (L3_HIT + L3_MISS)`
- **Average L3 request size check** = compare `L3_BYTE_*` against `L3_READ/WRITE * 64`

### Memory

| Metric | Unit | Meaning | CRI note |
|--------|------|---------|----------|
| `GPU_MEMORY_BYTE_READ` | bytes | Bytes read from GPU memory | Same bandwidth view as BMG |
| `GPU_MEMORY_BYTE_WRITE` | bytes | Bytes written to GPU memory | Same bandwidth view as BMG |
| `GPU_MEMORY_ACTIVE` | % | Time memory system is active | New direct activity counter |
| `GPU_MEMORY_READ` | events | Memory read requests | New |
| `GPU_MEMORY_WRITE` | events | Memory write requests | New |
| `TLB_MISS` | events | Translation misses | Same high-level signal |
| `TLB_PAGE_SIZE_1G` | events | TLB usage / access at 1G page size | New page-size visibility |
| `TLB_PAGE_SIZE_2M` | events | TLB usage / access at 2M page size | New |
| `TLB_PAGE_SIZE_4K` | events | TLB usage / access at 4K page size | New |
| `TLB_PAGE_SIZE_64K` | events | TLB usage / access at 64K page size | New |
| `COMPRESSOR_INPUT` | events | Compression input traffic | Compression pipeline visibility |
| `COMPRESSOR_OUTPUT` | events | Compression output traffic | Compression effectiveness proxy |

- **Memory bandwidth** = `(GPU_MEMORY_BYTE_READ + GPU_MEMORY_BYTE_WRITE) / GpuTime`
- **Compression ratio proxy** = `COMPRESSOR_OUTPUT / COMPRESSOR_INPUT`
- **Large-page usage mix** = compare `TLB_PAGE_SIZE_1G/2M/64K/4K`

### Copy Engine

| Metric | Unit | Meaning |
|--------|------|---------|
| `COPY_ENGINE_READ_REQUEST` | events | Copy-engine read requests |
| `COPY_ENGINE_WRITE_REQUEST` | events | Copy-engine write requests |
| `COPY_ENGINE_REQUEST_STALL` | % | Copy engine stalled on memory path |

CRI exposes transfer-path pressure directly in `ComputeBasic`, which helps distinguish kernel
bottlenecks from copy / staging / memory-movement bottlenecks (feeds the catalog's
`overlap-copy-compute` trigger).

---

## `VectorEngineProfile` `[SPEC]`

Use when `ComputeBasic` says the kernel is compute-heavy and you need **instruction mix** and
**datatype-specific** execution detail.

### Core instruction families

CRI provides detailed counts for instruction families such as: `FP16`, `FP32`, `FP64`, `FP64_2ND`,
`INT16`, `INT32`, `INT64`, `MATH`, `BITCONV`, `BARRIER`, `CONTROL`, `NONDIVERGENT`, `PREDICATION`,
`SEND`.

### CRI-only / expanded datatype visibility

| Metric family | Meaning | Difference vs BMG |
|---------------|---------|-------------------|
| `XVE_INST_EXECUTED_BF16` | BF16 ALU instruction count | New explicit BF16 ALU counting |
| `XVE_INST_EXECUTED_XMX_BF16` | BF16 XMX instruction count | Present on CRI |
| `XVE_INST_EXECUTED_XMX_FP16` | FP16 XMX instruction count | Present on CRI |
| `XVE_INST_EXECUTED_XMX_INT4` | INT4 XMX instruction count | Present on CRI |
| `XVE_INST_EXECUTED_XMX_INT8` | INT8 XMX instruction count | Present on CRI |
| `XVE_INST_EXECUTED_XMX_FP4` | FP4 XMX instruction count | New on CRI |
| `XVE_INST_EXECUTED_XMX_FP8` | FP8 XMX instruction count | New on CRI |
| `XVE_INST_EXECUTED_XMX_FP64` | FP64 XMX instruction count | New on CRI; not on BMG |

> **Difference from BMG:** `XMX_INT2` is **not** available on CRI.

### How to use it

- Confirm whether `joint_matrix` / library GEMM code actually maps to XMX hardware (catalog's
  `offload-xmx-library` "XMX idle" check).
- Separate ALU FP work from XMX tensor work.
- Distinguish `FP16/BF16` matrix paths from new `FP4/FP8/FP64` paths.

### Practical interpretation

- High `XMX_*` with low ALU counts: tensor-dominated kernel.
- High ALU `FP32/FP64` with low `XMX_*`: scalar/vector math dominates.
- `XVE_INST_EXECUTED_XMX_FP64` > 0: CRI is using an **FP64 tensor path** unavailable on BMG.
- `XVE_INST_EXECUTED_BF16` > 0 but low `XMX_BF16`: BF16 work may be on ALU rather than XMX.

---

## `VectorEngineStalls` `[SPEC]`

Use this set to answer **why XVEs are stalled** — the primary source for the catalog's dominant-stall
triggers (see the crosswalk in the BMG shard).

### Common stall categories

| Metric | Meaning | Framework alias |
|--------|---------|-----------------|
| `XVE_STALL_SBID` | Waiting on scoreboard dependencies, often memory-latency related | `SbidStall` |
| `XVE_STALL_SENDWR` | SEND pipeline / writeback congestion | `SendStall` |
| `XVE_STALL_ALUWR` | ALU writeback dependency stalls | (ALU dep sub-case of `PipeStall`) |
| `XVE_STALL_BARRIER` | Barrier / synchronization stalls | `SyncStall` |
| `XVE_STALL_CONTROL` | Control-flow related stalls | `ControlStall` |
| `XVE_STALL_INSTFETCH` | Instruction-fetch stalls | `InstrFetchStall` |
| `XVE_STALL_PIPESTALL` | General pipe-availability stalls | `PipeStall` / `DistStall` |
| `XVE_STALL_OTHER` | Residual / uncategorized stalls | — |

### Thread dispatch stall visibility

CRI includes the same general thread-dispatch coverage as BMG (queue activity and queue stall metrics
such as `THREAD_DISPATCH_QUEUE0_ACTIVE`, `THREAD_DISPATCH_QUEUE0_STALL`, `THREAD_DISPATCH_QUEUE1_ACTIVE`,
`THREAD_DISPATCH_QUEUE1_STALL`, and threadgroup dispatch resource stalls).

### Additional CRI thread-dispatch metrics

| Metric | Unit | Meaning |
|--------|------|---------|
| `THREAD_DISPATCH_GPGPU_COMMON_ENGINE` | events | Dispatches attributed to common GPGPU engine path |
| `THREAD_DISPATCH_GPGPU_COMPUTE_ENGINE` | events | Dispatches attributed to compute engine path |
| `THREAD_DISPATCH_INPUT_AVAILABLE` | % | Dispatcher had input available |
| `THREAD_DISPATCH_OTHER` | events | Other dispatch-path events |

### Interpretation patterns

- High `SBID`: latency hiding is insufficient; increase ILP, prefetch, or improve locality.
- High `SENDWR`: memory message traffic is saturating message paths.
- High `ALUWR`: dependency chains too tight; unroll / reorder to expose more parallelism.
- High `BARRIER`: synchronization overhead is limiting throughput.
- High queue stall + low `THREAD_DISPATCH_INPUT_AVAILABLE`: front-end / dependency starvation.

---

## `L1Profile` `[VERIFY]`

CRI uses **`L1Profile`** instead of the BMG `MemoryProfile` metric group.

The exact counters exposed can vary with your installed `unitrace`, Metrics Discovery package, driver,
and platform firmware. Always check:

```bash
unitrace --metric-list
```

### When to use `L1Profile`

- `ComputeBasic` shows poor L1 / memory efficiency but you need more detail.
- You want CRI-specific visibility into the L1 / load-store / memory path.
- You are validating cache-tiling, prefetching, or access-pattern changes.

### Practical guidance

- Treat `L1Profile` as the CRI replacement for BMG memory deep-dive profiling.
- Pair it with `ComputeBasic` so high-level L3 / memory / TLB trends already have context.
- Because CRI has **no counter-backed SLM bank-conflict metric equivalent to BMG's `MemoryProfile`**,
  the catalog's `pad-slm-arrays` trigger is *heuristic* on CRI — infer conflicts from access pattern
  rather than a direct counter.

---

## Recommended Analysis Workflow on CRI

1. **Run `ComputeBasic` first** — check `XVE_ACTIVE`, `XVE_STALL`, occupancy, L3 hit rate, memory
   bandwidth, copy-engine activity.
2. **If compute-bound, add `VectorEngineProfile`** — inspect ALU/XMX mix, BF16/FP4/FP8/FP64 usage.
3. **If stalled, add `VectorEngineStalls`** — identify the dominant stall family.
4. **If memory-path issues persist, add `L1Profile`** — refine cache / locality / access-pattern
   analysis.

---

## Derived Metrics and Formulas `[SPEC]`

| Derived metric | Formula | Why it matters |
|----------------|---------|----------------|
| XVE active ratio | `XVE_ACTIVE / (XVE_ACTIVE + XVE_STALL)` | Useful-work fraction when XVE dominates |
| SIMD efficiency | `XVE_INST_LANE_ENABLED / XVE_INST_LANE_TOTAL` | Measures lane utilization directly |
| L1 hit rate | `LOAD_STORE_CACHE_HIT / LOAD_STORE_CACHE_ACCESS` | Locality in load/store cache |
| L3 hit rate | `L3_HIT / (L3_HIT + L3_MISS)` | Device-cache effectiveness |
| ICache hit rate | `ICACHE_HIT / (ICACHE_HIT + ICACHE_MISS)` | Instruction-stream locality |
| Memory bandwidth | `(GPU_MEMORY_BYTE_READ + GPU_MEMORY_BYTE_WRITE) / GpuTime` | DRAM pressure |
| FP32 FLOPS | `XVE_OP_FP32 / GpuTime` | Direct FP32 throughput |
| FP64 FLOPS | `XVE_OP_FP64 / GpuTime` | Direct FP64 throughput |
| Copy-engine read/write balance | `COPY_ENGINE_READ_REQUEST` vs `COPY_ENGINE_WRITE_REQUEST` | Transfer-pattern diagnosis |

---

## Thresholds for Analysis `[EMPIRICAL]`

Starting heuristics for CRI. Similar to BMG (see `intel-gpu-hardware-metrics-bmg.md` threshold table),
but interpret with these CRI-specific cautions:

- **Thread occupancy thresholds may differ** from BMG due to different thread-count limits per XVE.
- **ALU utilization must be interpreted with 4 pipes**: each pipe contributes about **25%** of the
  total pipe issue opportunity, so the catalog's `balance-alu-pipes` "one pipe ≫ others" trigger looks
  across four pipes, not three.
- **XMX FP64** workloads may have different throughput behavior than FP16/BF16/INT8 tensor workloads.
