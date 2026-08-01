# Intel Battlemage (BMG / Xe2) Hardware Metrics Reference

Canonical catalog of the Intel Metrics Discovery / `unitrace` metric groups and counters available on
**Intel Battlemage (BMG, Xe2)** GPUs (Arc Pro B60/B70), with units, derived ratios, and empirical
thresholds. This is the *counter-level* companion to the group names used in `sycl-profiler` §4 and the
trigger vocabulary in `sycl-optimization-catalog.md`.

> **Scope.** This shard documents *what each counter means and when it is alarming*. For the
> **strategy** each signal should drive (priority, conflicts, synergies), stay in
> `sycl-optimization-catalog.md` Part 2 — this shard is cited by its "Profiling signals used in
> triggers" section, not a replacement for it. For CRI (Xe3P) deltas see
> `intel-gpu-hardware-metrics-cri.md`.

## Provenance
- `[ARCH]` — architectural facts (pipe count, XMX types, memory type). Stable across BMG boards.
- `[SPEC]` — counter definitions and metric-group membership from Intel Metrics Discovery / `unitrace`.
- `[VERIFY]` — the exact counters exposed depend on the installed `unitrace`, Metrics Discovery
  package, driver, and firmware. Confirm on the running device with `unitrace --metric-list` before
  relying on a specific counter, and record confirmed values in `.sycl/state/lessons.md`.
- Thresholds below are `[EMPIRICAL]` starting heuristics — always re-measure on the target device.

## Framework stall-name crosswalk `[SPEC]`
`sycl-profiler` §4 and `sycl-optimization-catalog.md` use short aliases for the XVE stall reasons; the
raw `VectorEngineStalls` counters this shard lists map to them as follows. Use this table to translate
between a catalog trigger and a `unitrace` CSV column.

| Framework alias | Raw counter (`VectorEngineStalls`) | Meaning |
|---|---|---|
| `SendStall` | `XVE_STALL_SENDWR` | waiting on a memory/SLM message (load/store/atomic) |
| `SbidStall` | `XVE_STALL_SBID` | scoreboard dep — waiting on an outstanding async result |
| `PipeStall` / `DistStall` | `XVE_STALL_PIPESTALL` | pipe conflict / GRF hold / register distance |
| `SyncStall` | `XVE_STALL_BARRIER` | barrier / synchronization wait |
| `ControlStall` | `XVE_STALL_CONTROL` | branch / control-flow wait |
| `InstrFetchStall` | `XVE_STALL_INSTFETCH` | instruction-cache miss |
| (ALU writeback dep) | `XVE_STALL_ALUWR` | waiting for ALU to write GRF/ACC — a low-ILP sub-case of `PipeStall` |

---

## Metric Groups at a Glance `[SPEC]`

| Metric Group | Purpose | Best Use |
|---|---|---|
| `ComputeBasic` | Default, broad top-level compute profile | First-pass profiling and general bottleneck triage |
| `VectorEngineProfile` | Detailed XVE instruction mix and execution profile | ALU/XMX utilization analysis |
| `VectorEngineStalls` | Detailed XVE stall and dispatch breakdown | Root-causing pipeline and dependency stalls |
| `MemoryProfile` | Detailed memory subsystem profiling | Diagnosing bandwidth, cache, SLM, and host/device traffic issues |
| `DeviceCacheProfile` | Detailed L3 client attribution | Understanding which clients drive L3 traffic and misses |

## Which metric group when `[SPEC]`

Decision guide. You almost always open with `VectorEngineStalls` + `ComputeBasic` (the
`profile_metrics.sh` default), read the stall mix + occupancy/bandwidth, then pull **one** follow-up
group based on what that first pass points to. Query only what the current hypothesis needs — each
extra group is another run.

| You already know (roofline + first pass) | Query next | Read these counters / ratios | Then see strategy |
|---|---|---|---|
| Nothing yet — first pass | `VectorEngineStalls` + `ComputeBasic` | Active % vs each `*Stall` %; EU active %, thread occupancy, achieved DRAM BW, L1/L2 hit rate | branch to a row below |
| Memory-bound roofline / dominant `SendStall` | `MemoryProfile` | DRAM read+write bytes vs peak BW, L1/L2/L3 hit rates, SLM bank-conflict rate | `coalesce-memory-access`, `vectorize-vec4`, `tile-data-access`, `pad-slm-arrays` |
| Low cache hit rate / who is thrashing L3 | `DeviceCacheProfile` (+ `L3`) | L3 hit rate, per-client L3 traffic/misses | `prefetch-to-slm`, `tile-data-access` |
| Compute-bound / ALU-pipe imbalance / XMX idle | `VectorEngineProfile` | instruction mix, `ALU0`-`ALU2` issue balance, XMX/DPAS activity | `balance-alu-pipes`, `offload-xmx-library` (route matmul to `joint_matrix`/oneMKL/oneDNN) |
| Low ILP — dominant `PipeStall`/`DistStall` or `SbidStall` | `VectorEngineStalls` (already) + IGC asm | which IPs stall + reason; dependency distance | `increase-ilp`, `software-pipeline` |
| Low occupancy, suspect register pressure | `ComputeBasic` occupancy + IGC asm | thread occupancy %, spill/fill in the asm dump | `reduce-register-pressure` |
| Over-synchronization — dominant `SyncStall` | `VectorEngineStalls` (already) | barrier stall % | `double-buffer-slm`, fewer/rebalanced barriers |

> **BMG specifics:** the memory-subsystem deep-dive group is **`MemoryProfile`**, and SLM
> bank-conflict rate is a first-class counter here. FLOP / SIMD-lane / GRF-occupancy are **not** direct
> counters on BMG, so ILP, SIMD efficiency, and register pressure are read from the stall mix + IGC asm
> (spills), not a dedicated counter — this is the main routing difference vs CRI (see
> `intel-gpu-hardware-metrics-cri.md` → "Which metric group when").

## Architecture-Specific Notes `[ARCH]`

- **Architecture:** Xe2 (desktop / gaming-class GPU repurposed for compute)
- **Execution pipes:** **3 ALU pipes per XVE** (`ALU0`, `ALU1`, `ALU2`)
- **XMX types supported:** `INT2`, `INT4`, `INT8`, `FP16`, `BF16`
- **No FP64 XMX** support on BMG
- **Memory type:** **GDDR6**, so memory efficiency and cache reuse are especially important

---

## 1. ComputeBasic `[SPEC]`

`ComputeBasic` is the **default** and most commonly used metric group. It gives a balanced overview of GPU
activity, front-end utilization, thread dispatch, XVE execution, cache behavior, and memory traffic.

### GPU Overview

| Metric | Unit | Description |
|---|---:|---|
| `GpuTime` | ns | Total GPU time elapsed |
| `GpuCoreClocks` | cycles | Total GPU core clocks elapsed |
| `AvgGpuCoreFrequencyMHz` | MHz | Average GPU core frequency |

### Front End

| Metric | Unit | Description |
|---|---:|---|
| `GPU_BUSY` | % | GPU not idle across all engines |
| `COMMAND_PARSER_COMPUTE_ENGINE_BUSY` | % | Compute queue active time |
| `COMMAND_PARSER_RENDER_ENGINE_BUSY` | % | 3D queue active time |
| `COMMAND_PARSER_COPY_ENGINE_BUSY` | % | Copy queue active time |
| `COMMAND_PARSER_FLUSH_COUNT` | events | Stalling flushes |

### Thread Dispatcher

| Metric | Unit | Description |
|---|---:|---|
| `GPGPU_THREADGROUP_COUNT` | events | GPGPU threadgroups dispatched |
| `ASYNC_GPGPU_THREADGROUP_COUNT` | events | Async GPGPU threadgroups dispatched |
| `ASYNC_GPGPU_THREAD_EXIT_COUNT` | events | EOT messages received |
| `GPGPU_DISPATCH` | % | Time spent dispatching GPGPU threads to XVEs |

### Vector Engine (XVE)

| Metric | Unit | Description |
|---|---:|---|
| `XVE_ACTIVE` | % | At least one pipe active |
| `XVE_STALL` | % | Thread loaded but no pipe active |
| `XVE_THREADS_OCCUPANCY_ALL` | % | Thread slot occupancy |
| `XVE_INST_EXECUTED_ALU0_ALL` | events | ALU0 pipe execution slots |
| `XVE_INST_EXECUTED_ALU1_ALL` | events | ALU1 pipe execution slots |
| `XVE_INST_EXECUTED_ALU2_ALL` | events | ALU2 pipe execution slots |
| `XVE_INST_EXECUTED_SEND_ALL` | events | SEND pipe dispatches |
| `XVE_INST_ISSUED_ALL` | events | Instructions issued to any pipe |
| `XVE_SHARED_FUNCTION_ACCESS_HOLD` | % | Stalled by shared function units |
| `XVE_MULTIPLE_PIPE_ACTIVE` | % | Two or more ALU pipes active simultaneously |
| `XVE_PIPE_ALU0_AND_ALU1_ACTIVE` | % | ALU0 and ALU1 both active |
| `XVE_PIPE_ALU0_AND_ALU2_ACTIVE` | % | ALU0 and ALU2 both active |
| `XVE_INST_EXECUTED_ALU0_ALL_UTILIZATION` | % | ALU0 time utilization |
| `XVE_INST_EXECUTED_ALU1_ALL_UTILIZATION` | % | ALU1 time utilization |
| `XVE_INST_EXECUTED_ALU2_ALL_UTILIZATION` | % | ALU2 time utilization |
| `ICACHE_HIT` | events | Instruction cache hits |
| `ICACHE_MISS` | events | Instruction cache misses |

### L1 Cache / Load-Store Cache (LSC)

| Metric | Unit | Description |
|---|---:|---|
| `LOAD_STORE_CACHE_ACCESS` | events | Load-store cache accesses |
| `LOAD_STORE_CACHE_HIT` | events | Load-store cache hits |
| `LOAD_STORE_CACHE_BYTE_READ` | bytes | Bytes read from LSC, excluding SLM |
| `LOAD_STORE_CACHE_BYTE_WRITE` | bytes | Bytes written to LSC, excluding SLM |
| `LOAD_STORE_CACHE_PARTIAL_WRITE_COUNT` | events | Partial writes that do not fill a subsector |
| `SLM_BANK_CONFLICT_COUNT` | events | Shared local memory bank conflicts |
| `SLM_BYTE_READ` | bytes | SLM bytes read |
| `SLM_BYTE_WRITE` | bytes | SLM bytes written |

### Device Cache (L3)

| Metric | Unit | Description |
|---|---:|---|
| `L3_HIT` | events | L3 hits |
| `L3_MISS` | events | L3 misses |
| `L3_READ` | events | L3 64-byte read requests |
| `L3_WRITE` | events | L3 64-byte write requests |
| `L3_ATOMIC_ACCESS` | events | Atomic accesses to L3 |
| `L3_STALL` | % | L3 bank stalled |

### Memory

| Metric | Unit | Description |
|---|---:|---|
| `GPU_MEMORY_BYTE_READ` | bytes | Device local memory read bytes |
| `GPU_MEMORY_BYTE_WRITE` | bytes | Device local memory write bytes |
| `GPU_MEMORY_BYTE_READ_RATE` | bytes/s | Memory read bandwidth |
| `GPU_MEMORY_BYTE_WRITE_RATE` | bytes/s | Memory write bandwidth |
| `GPU_MEMORY_L3_READ` | events | Memory reads caused by L3 misses |
| `GPU_MEMORY_L3_WRITE` | events | Memory writes caused by L3 invalidations |
| `GPU_MEMORY_REQUEST_QUEUE_FULL` | % | Sequencer queue above fullness threshold |
| `TLB_MISS` | events | TLB misses across all engines |
| `COMPRESSOR_INPUT` | events | 256-byte writes at compressor input |
| `COMPRESSOR_OUTPUT` | events | 256-byte writes at compressor output |
| `HOST_TO_GPUMEM_TRANSACTION_READ` | events | Host 64-byte reads to GPU memory |
| `HOST_TO_GPUMEM_TRANSACTION_WRITE` | events | Host 64-byte writes to GPU memory |
| `SYSMEM_TRANSACTION_READ` | events | System memory 64-byte reads upstream |
| `SYSMEM_TRANSACTION_WRITE` | events | System memory 64-byte writes upstream |

### What ComputeBasic Is Best For

- Determining whether the workload is **compute-bound**, **stall-bound**, or **memory-bound**
- Measuring overall XVE activity and occupancy
- Checking LSC, L3, and memory traffic at a high level
- Establishing a baseline before collecting deeper metric groups

---

## 2. VectorEngineProfile `[SPEC]`

`VectorEngineProfile` extends the `ComputeBasic` XVE coverage with a **detailed instruction mix** and
extra L3 utilization signals.

> Includes all `ComputeBasic` XVE metrics, plus the detailed metrics below.

### Detailed Instruction Breakdown

| Metric | Unit | Description |
|---|---:|---|
| `XVE_INST_EXECUTED_FP16` | events | FP16 ALU instruction slots |
| `XVE_INST_EXECUTED_FP32` | events | FP32 ALU instruction slots |
| `XVE_INST_EXECUTED_FP64` | events | FP64 ALU instruction slots |
| `XVE_INST_EXECUTED_FP64_2ND` | events | FP64 execution on second pipe |
| `XVE_INST_EXECUTED_INT16` | events | INT16 ALU instruction slots |
| `XVE_INST_EXECUTED_INT32` | events | INT32 ALU instruction slots |
| `XVE_INST_EXECUTED_INT64` | events | INT64 instruction slots |
| `XVE_INST_EXECUTED_MATH` | events | Extended math instruction slots |
| `XVE_INST_EXECUTED_BITCONV` | events | Bit manipulation instruction slots |
| `XVE_INST_EXECUTED_CONTROL_ALL` | events | JEU pipe / branch instructions |
| `XVE_INST_EXECUTED_BARRIER` | events | `sync.bar` and `sync.host` instructions |
| `XVE_INST_EXECUTED_NONDIVERGENT` | events | Non-divergent instructions |
| `XVE_INST_EXECUTED_PREDICATION` | events | Instructions with predication enabled |
| `XVE_INST_EXECUTED_SEND_ALL` | events | SEND pipe dispatches |

### XMX Instruction Breakdown

| Metric | Unit | Description |
|---|---:|---|
| `XVE_INST_EXECUTED_XMX_BF16` | events | BF16 XMX slots |
| `XVE_INST_EXECUTED_XMX_FP16` | events | FP16 XMX slots |
| `XVE_INST_EXECUTED_XMX_INT2` | events | INT2 XMX slots |
| `XVE_INST_EXECUTED_XMX_INT4` | events | INT4 XMX slots |
| `XVE_INST_EXECUTED_XMX_INT8` | events | INT8 XMX slots |

### Additional L3 State Metrics

| Metric | Unit | Description |
|---|---:|---|
| `L3_BUSY` | % | L3 active / busy time |
| `L3_INPUT_AVAILABLE` | % | L3 input-side availability |
| `L3_OUTPUT_READY` | % | L3 output-side readiness |
| `L3_SUPERQ_FULL` | % | L3 superqueue full condition |

### What VectorEngineProfile Is Best For

- Distinguishing **FP**, **INT**, **control**, **SEND**, and **XMX** dominated kernels
- Validating whether BMG's **three ALU pipes** are being fed effectively
- Understanding whether a kernel is really tensor/XMX-heavy or mostly conventional ALU work
- Correlating instruction mix with occupancy, stall, and cache behavior (feeds the catalog's
  `offload-xmx-library` / `balance-alu-pipes` triggers)

---

## 3. VectorEngineStalls `[SPEC]`

`VectorEngineStalls` breaks XVE non-productive time into specific dependency and pipeline causes. This
is the **primary source for the catalog's dominant-stall triggers** — see the crosswalk above.

### Stall Breakdown

| Metric | Unit | Description | Framework alias |
|---|---:|---|---|
| `XVE_STALL_SBID` | % | Waiting for scoreboard token / data dependency | `SbidStall` |
| `XVE_STALL_SENDWR` | % | Waiting for SEND message dispatch | `SendStall` |
| `XVE_STALL_ALUWR` | % | Waiting for ALU to write GRF/ACC register | (ALU dep sub-case of `PipeStall`) |
| `XVE_STALL_BARRIER` | % | Waiting for barrier notification | `SyncStall` |
| `XVE_STALL_CONTROL` | % | Waiting for branch completion | `ControlStall` |
| `XVE_STALL_INSTFETCH` | % | Waiting for instruction fetch | `InstrFetchStall` |
| `XVE_STALL_PIPESTALL` | % | Blocked by pipe conflicts, GRF holds, or send holds | `PipeStall` / `DistStall` |
| `XVE_STALL_PS_DEPENDENCY` | % | Pixel shader dependency | — |
| `XVE_STALL_OTHER` | % | Other dependencies such as Flag or EoT | — |

### Thread Dispatch Queue Metrics

| Metric | Unit | Description |
|---|---:|---|
| `THREAD_DISPATCH_QUEUE0_ACTIVE` | % | Thread dispatch queue 0 active |
| `THREAD_DISPATCH_QUEUE1_ACTIVE` | % | Thread dispatch queue 1 active |
| `THREAD_DISPATCH_QUEUE0_STALL` | % | Thread dispatch queue 0 stalled |
| `THREAD_DISPATCH_QUEUE1_STALL` | % | Thread dispatch queue 1 stalled |
| `THREADGROUP_DISPATCH_QUEUE0_RESOURCE_STALL` | % | Queue 0 stalled waiting for SLM, barrier, or BTD resources |
| `THREADGROUP_DISPATCH_QUEUE1_RESOURCE_STALL` | % | Queue 1 stalled waiting for SLM, barrier, or BTD resources |

### What VectorEngineStalls Is Best For

- Root-causing high `XVE_STALL`
- Separating **dependency stalls** from **dispatch stalls** and **pipe conflicts**
- Detecting instruction fetch pressure, barrier pressure, and SEND pressure
- Choosing the catalog row whose trigger names the dominant stall

---

## 4. MemoryProfile `[SPEC]`

`MemoryProfile` focuses on the memory subsystem in more detail than `ComputeBasic`.

> Includes all of the `ComputeBasic` memory metrics, plus **more detailed breakdowns of GPU memory, L3,
> SLM, and host memory traffic**.

### Core Memory Metrics Included

| Metric | Unit | Description |
|---|---:|---|
| `GPU_MEMORY_BYTE_READ` | bytes | Device local memory read bytes |
| `GPU_MEMORY_BYTE_WRITE` | bytes | Device local memory write bytes |
| `GPU_MEMORY_BYTE_READ_RATE` | bytes/s | Memory read bandwidth |
| `GPU_MEMORY_BYTE_WRITE_RATE` | bytes/s | Memory write bandwidth |
| `GPU_MEMORY_L3_READ` | events | Memory reads due to L3 misses |
| `GPU_MEMORY_L3_WRITE` | events | Memory writes due to L3 invalidations |
| `GPU_MEMORY_REQUEST_QUEUE_FULL` | % | Sequencer queue fullness pressure |
| `TLB_MISS` | events | TLB misses |
| `COMPRESSOR_INPUT` | events | 256-byte writes entering compression |
| `COMPRESSOR_OUTPUT` | events | 256-byte writes leaving compression |
| `HOST_TO_GPUMEM_TRANSACTION_READ` | events | Host reads targeting GPU memory |
| `HOST_TO_GPUMEM_TRANSACTION_WRITE` | events | Host writes targeting GPU memory |
| `SYSMEM_TRANSACTION_READ` | events | Upstream system memory reads |
| `SYSMEM_TRANSACTION_WRITE` | events | Upstream system memory writes |
| `SLM_BYTE_READ` | bytes | Shared local memory bytes read |
| `SLM_BYTE_WRITE` | bytes | Shared local memory bytes written |
| `SLM_BANK_CONFLICT_COUNT` | events | Shared local memory bank conflicts |
| `L3_READ` | events | L3 read requests |
| `L3_WRITE` | events | L3 write requests |
| `L3_HIT` | events | L3 hits |
| `L3_MISS` | events | L3 misses |

### What MemoryProfile Is Best For

- Diagnosing **off-chip bandwidth** pressure on GDDR6
- Separating **L3 inefficiency** from true device-memory demand
- Quantifying **SLM usage** and **SLM bank conflicts** (feeds the catalog's `pad-slm-arrays` trigger,
  which is counter-backed on BMG and heuristic on CRI)
- Detecting **host/device transfer overhead** and system memory spill behavior
- Measuring whether memory request queues are saturating

### Practical BMG Interpretation Notes

- High `GPU_MEMORY_BYTE_READ_RATE` / `GPU_MEMORY_BYTE_WRITE_RATE` combined with low cache hit rates
  often indicates a **memory-bound** kernel.
- Rising `GPU_MEMORY_REQUEST_QUEUE_FULL` means memory-side backpressure is building.
- A large gap between `COMPRESSOR_INPUT` and `COMPRESSOR_OUTPUT` indicates effective compression; little
  or no reduction means compression is not helping.

---

## 5. DeviceCacheProfile `[SPEC]`

`DeviceCacheProfile` provides a more detailed view of **L3 client behavior**, helping attribute L3
pressure to specific request sources.

### Detailed L3 Client Breakdown

| Metric | Description |
|---|---|
| `LOAD_STORE_CACHE_L3_READ` | Load-store cache requests to L3 |
| `LOAD_STORE_CACHE_L3_HIT` | Load-store cache hits in L3 |
| `LOAD_STORE_CACHE_L3_WRITE` | Load-store cache writes to L3 |
| `ICACHE_L3_READ` | Instruction cache requests to L3 |
| `ICACHE_L3_HIT` | Instruction cache hits in L3 |
| `SAMPLER_L3_HIT` | Sampler cache requests hitting in L3 |
| `COLOR_L3_ACCESS` | Color cache accesses to L3 |
| `COLOR_L3_HIT` | Color cache hits in L3 |
| `AMFS_L3_ACCESS` | AMFS accesses to L3 |
| `AMFS_L3_HIT` | AMFS hits in L3 |
| `AMFS_L3_WRITE` | AMFS writes to L3 |

### What DeviceCacheProfile Is Best For

- Determining which client is driving L3 traffic and misses
- Distinguishing **load-store**, **instruction**, **sampler**, **color**, and **AMFS** cache pressure
- Explaining poor aggregate L3 efficiency seen in `ComputeBasic`

---

## Derived Ratios and How to Read Them `[SPEC]`

These derived ratios are often more useful than raw counters alone.

| Derived Metric | Formula |
|---|---|
| L3 Hit Rate | `L3_HIT / (L3_HIT + L3_MISS)` |
| LSC Hit Rate | `LOAD_STORE_CACHE_HIT / LOAD_STORE_CACHE_ACCESS` |
| ICache Hit Rate | `ICACHE_HIT / (ICACHE_HIT + ICACHE_MISS)` |
| SLM Bank Conflict Rate | `SLM_BANK_CONFLICT_COUNT / SLM accesses` |
| TLB Miss Rate | `TLB_MISS / memory translation opportunities` |
| Compression Ratio | `COMPRESSOR_INPUT / COMPRESSOR_OUTPUT` |

> Some denominators depend on the profiler or post-processing flow (e.g. SLM conflict rate needs a
> consistent definition of total SLM accesses).

## Empirical Thresholds for Analysis `[EMPIRICAL]`

Starting heuristics — always re-measure on the target device. These back the numeric cutoffs used in
`sycl-optimization-catalog.md` triggers (e.g. L3 hit rate < 0.60).

| Metric | Good | Warning | Critical |
|---|---|---|---|
| `XVE_ACTIVE` | > 70% | 40-70% | < 40% |
| `XVE_STALL` | < 30% | 30-50% | > 50% |
| Thread Occupancy | > 60% | 30-60% | < 30% |
| L3 Hit Rate | > 80% | 60-80% | < 60% |
| LSC Hit Rate | > 70% | 50-70% | < 50% |
| ICache Hit Rate | > 95% | 90-95% | < 90% |
| SLM Bank Conflicts | < 5% of accesses | 5-15% | > 15% |
| GPU Memory Queue Full | < 20% | 20-50% | > 50% |
| TLB Miss Rate | < 1% | 1-5% | > 5% |
| Compression Ratio | > 1.5x | 1.0-1.5x | 1.0x (no compression) |

---

## Quick Triage Guide for BMG `[EMPIRICAL]`

| Symptom | Metrics to Check First | Typical Interpretation | Catalog class |
|---|---|---|---|
| Low overall throughput | `GPU_BUSY`, `XVE_ACTIVE`, `XVE_THREADS_OCCUPANCY_ALL` | Under-filled GPU or poor residency | occupancy-bound |
| High stall time | `XVE_STALL`, then `VectorEngineStalls` metrics | Dependency, barrier, SEND, or pipe-conflict issue | memory-latency / sync |
| Poor cache behavior | `LOAD_STORE_CACHE_HIT`, `L3_HIT`, `ICACHE_HIT` | Weak locality, large working set, or fetch inefficiency | cache |
| Memory bound behavior | `GPU_MEMORY_BYTE_READ_RATE`, `GPU_MEMORY_BYTE_WRITE_RATE`, `GPU_MEMORY_REQUEST_QUEUE_FULL` | Off-chip bandwidth pressure on GDDR6 | memory-bandwidth |
| SLM inefficiency | `SLM_BYTE_READ`, `SLM_BYTE_WRITE`, `SLM_BANK_CONFLICT_COUNT` | Bank conflicts or poor scratchpad access pattern | cache (`pad-slm-arrays`) |
| Unexpected host traffic | `HOST_TO_GPUMEM_TRANSACTION_*`, `SYSMEM_TRANSACTION_*` | Extra host/device or system memory traffic | host/dispatch overlap |
| XMX underutilized | `XVE_INST_EXECUTED_XMX_*` | Tensor hardware not effectively used | compute (`offload-xmx-library`) |

## Recommended Collection Order

1. Start with **`ComputeBasic`**.
2. If XVE utilization is low or ambiguous, collect **`VectorEngineProfile`**.
3. If XVE stalls are high, collect **`VectorEngineStalls`**.
4. If memory bandwidth or cache efficiency looks problematic, collect **`MemoryProfile`**.
5. If L3 is the bottleneck and attribution is unclear, collect **`DeviceCacheProfile`**.

This sequence keeps overhead manageable while narrowing the root cause quickly on BMG GPUs.
