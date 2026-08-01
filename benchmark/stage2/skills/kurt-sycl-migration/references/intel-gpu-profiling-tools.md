# Intel GPU Profiling Tools `[SPEC]`

Inventory of the tools used to measure and diagnose SYCL kernels on Intel Xe2 (Battlemage — Arc Pro
B60/B70) and Xe3P (Crescent Island — CRI) GPUs: what each tool is for, `unitrace`'s collection modes,
how to keep a trace from crashing the tracer, and the IGC assembly dump. This is the *tool reference*
behind the profiling workflow.

> **Where the rest lives.** This shard is the tool catalog; the **procedure** (what to run, in what
> order, and how to turn the output into a diagnosis) is `sycl-profiler` §4. The **counter-level**
> meaning of each metric group and its thresholds is in
> [intel-gpu-hardware-metrics-bmg.md](intel-gpu-hardware-metrics-bmg.md) (Xe2) /
> [intel-gpu-hardware-metrics-cri.md](intel-gpu-hardware-metrics-cri.md) (Xe3P). The device envelope
> and roofline are in [intel-gpu-hardware.md](intel-gpu-hardware.md).

> **Runner rule:** all collection runs through `.sycl/scripts/run.sh` on the Intel GPU runner — the
> helper scripts `sycl-profiler/scripts/profile_metrics.sh` (single kernel) and `profile_e2e.sh`
> (whole workload) wrap the `unitrace` invocations below. Prefer them over calling `unitrace` by hand.

## Device-capability tools `[SPEC]`
- `sycl-ls --verbose`, `zeinfo` (Level Zero), `clinfo` (OpenCL) — enumerate devices and read EU count,
  max work-group size, SLM (`local_mem_size`), sub-group sizes, and backend selection. Use these to
  confirm the `[VERIFY]` rows in [intel-gpu-hardware.md](intel-gpu-hardware.md) on the live device.
- Env: `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` pins the Level Zero GPU backend.

## `unitrace` — the primary profiler `[SPEC]`
Three complementary collection modes (see `sycl-profiler` §4 for how to combine them into a diagnosis):
- `--device-timing` / `--device-timeline` — per-kernel wall-clock and a kernel timeline. Use this
  first to find the hot kernels and rank them (drives `profile_e2e.sh`).
- `--metric-query --group <G>` — hardware counters aggregated **per kernel** (one value per
  invocation). `--metric-sampling --group <G>` instead collects a **time-based counter timeline**
  (pairs with `--sampling-interval` / `--devices-to-sample`). For the available groups, the counters
  inside each, their derived ratios and alarm thresholds, and *which group to query when*, see the
  metrics shards ([BMG](intel-gpu-hardware-metrics-bmg.md) / [CRI](intel-gpu-hardware-metrics-cri.md)).
- `--stall-sampling` — attributes stalls to individual **instruction pointers** and a **reason**
  (`SendStall`, `SbidStall`, `PipeStall`/`DistStall`, `ControlStall`, `SyncStall`, `InstrFetchStall`).
  The single most direct "why is it slow" signal. The alias↔raw-`XVE_STALL_*` mapping is in the BMG
  shard's "Framework stall-name crosswalk".
- `analyzeperfmetrics.py` (ships with `unitrace`) maps stall IPs onto the IGC assembly
  (`<kernel>.asm.ip`) so you see the exact stalling instructions.

Discover the exact groups/counters on the installed system with `unitrace --metric-list` `[VERIFY]`.

## Scoped collection — avoid tracing the whole run `[SPEC]`
Tracing a large end-to-end workload in full can produce huge traces and **crash `unitrace`**
(out-of-memory / core dump). Restrict collection to the region and kernels of interest instead of
tracing start-to-end:
- `--include-kernels <substr>,<substr>` / `--exclude-kernels` — profile only matching kernels; cuts
  overhead and trace size dramatically and is the cheapest mitigation.
- **Temporal / out-of-application control** (Linux, no code change): start with `--start-paused
  --session <name>`, then from another shell `unitrace --resume <name>`, `--pause <name>`,
  `--stop <name>` to bracket exactly the iterations you want. `profile_e2e.sh slice` wraps this.
- **Spatial / in-application control**: start with `--start-paused` and wrap the region with
  `PTI_ENABLE_COLLECTION=1/0` (env) or `__itt_resume()/__itt_pause()` in the app.
- Prefer a **standalone single-kernel benchmark** as the profiling target over the full workload — it
  is small, deterministic, and cannot overflow the tracer (see `sycl-profiler` §2/§4).

## IGC shader / assembly dump `[SPEC]`
`IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=<dir>` emits the generated GEN assembly. Inspect the asm
for **register spills/fills** (the top Xe red flag), SIMD width, DPAS (XMX) density, and uncoalesced
sends.

> **When the dump happens depends on compilation mode.** For **JIT** builds IGC runs at **run time**,
> so set the env vars when *running* the app. For **AOT** builds (`-fsycl-targets=spir64_gen -Xs
> "-device …"`) IGC runs at **build time**, so a runtime dump is **empty** — set the env vars during
> the *build* instead, or profile a **JIT-compiled** build of the kernel. `profile_metrics.sh`
> automates the JIT case and accepts a prebuilt AOT dump via `SHADER_DUMP=prebuilt SHADER_DUMP_DIR=…`.

## Other tools `[SPEC]`
- **Intel VTune Profiler** (GPU Compute / Media Hotspots) — EU activity, occupancy, bandwidth, stalls;
  a GUI/CLI alternative when `unitrace` is unavailable on the runner.
- **Intel Advisor** — GPU roofline analysis (AI vs ridge point, fraction-of-peak).
