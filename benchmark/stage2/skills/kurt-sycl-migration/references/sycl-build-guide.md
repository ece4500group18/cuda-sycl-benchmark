# Intel GPU Build Guide (icpx / oneAPI) `[SPEC]`

Reference for building SYCL kernels with `icpx` for Intel Xe2 (Battlemage — Arc Pro B60/B70) and
Xe3P (Crescent Island — CRI) GPUs: JIT vs AOT, device tokens, optimization flags, oneMKL linking,
environment, and common build errors. Hardware peaks and the per-platform AOT device tokens live in
[intel-gpu-hardware.md](intel-gpu-hardware.md); the roofline/optimization playbook is in
[sycl-optimization-catalog.md](sycl-optimization-catalog.md).

> **Runner rule:** all builds/tests go through `.sycl/scripts/run.sh` on the runner — **never call
> `icpx` directly**. This shard is the flag reference *behind* that wrapper; fold these flags into the
> project's existing build (see `sycl-migration` step 4), don't stand up a separate build tree.

---

## 1. Compiler invocation

### Basic SYCL compilation

```bash
# Source the oneAPI environment first (once per shell) so icpx/ocloc are on PATH.
source /opt/intel/oneapi/setvars.sh

# Compile AND link a SYCL application.
icpx -fsycl -O2 -o my_kernel my_kernel.cpp
```

`-fsycl` enables SYCL language extensions and links the SYCL runtime. It is required at **both**
compile and link — a missing link-time `-fsycl` yields `undefined reference to sycl::…`.

---

## 2. Ahead-of-Time (AOT) compilation

AOT pre-compiles device code for a specific GPU architecture, removing the JIT overhead at first run.
It requires the Intel offline compiler `ocloc` on `PATH` (`apt install intel-ocloc`; verify with
`ocloc compile --help`).

### Per-platform targets

| Platform | `-fsycl-targets` token | ocloc device (`-Xs "-device …"`) |
|----------|------------------------|----------------------------------|
| Arc Pro B60 (Xe2, BMG-G21) | `intel_gpu_bmg_g21` | `bmg-g21` |
| Arc Pro B70 (Xe2, BMG-G31) | `intel_gpu_bmg_g31` | `bmg-g31` |
| Crescent Island (Xe3P) `[VERIFY]` | `intel_gpu_cri` | `cri` |

```bash
# B60 (BMG-G21)
icpx -fsycl -fsycl-targets=intel_gpu_bmg_g21 -O2 -o my_kernel my_kernel.cpp

# B70 (BMG-G31)
icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 -O2 -o my_kernel my_kernel.cpp

# Crescent Island (Xe3P) — token [VERIFY] with `ocloc compile --help` before relying on AOT.
icpx -fsycl -fsycl-targets=intel_gpu_cri -O2 -o my_kernel my_kernel.cpp
```

### The two token namespaces differ (and are versioned)

> The `-fsycl-targets` names (`intel_gpu_bmg_g21`) and the ocloc `-Xs "-device …"` names (`bmg-g21`)
> are **different namespaces** and change between toolchain releases. A generic `intel_gpu_bmg` may be
> **rejected** by newer toolchains (e.g. icpx 2026+). Discover the valid names like this:
>
> - **ocloc `-device` acronyms** — run `ocloc compile --help` and read the `-device <device_type>`
>   list; expand an acronym to concrete stepping versions with `ocloc ids <acronym>` (e.g.
>   `ocloc ids xe-hpg`). An **older** ocloc build won't list `bmg`/`cri` — update ocloc (it ships
>   with the compute-runtime/NEO driver) to a version that supports the target arch.
> - **`-fsycl-targets=intel_gpu_*` names** — `--help` does **not** enumerate these. Read the running
>   device's architecture from `sycl-ls --verbose` (Architecture field) or the device's
>   `ext_oneapi_architecture` info; the token equals the
>   `sycl::ext::oneapi::experimental::architecture` enum name (e.g. `intel_gpu_bmg_g21`). The full
>   list is in the oneAPI DPC++ “Ahead-of-Time Compilation” guide. Treat as `[VERIFY]` per compiler
>   version.

### SPIR-V gen fallback (ocloc form)

```bash
# Equivalent AOT via the SPIR-V generator, selecting the device by ocloc token.
icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg-g21" -O2 -o my_kernel my_kernel.cpp
```

### Multi-target (fat binary)

```bash
# Emit device code for both BMG dies in one binary.
icpx -fsycl -fsycl-targets=intel_gpu_bmg_g21,intel_gpu_bmg_g31 -O2 -o my_kernel my_kernel.cpp
```

---

## 3. JIT compilation

With **no** `-fsycl-targets`, device code is emitted as SPIR-V and JIT-compiled at first run:

```bash
icpx -fsycl -O2 -o my_kernel my_kernel.cpp
```

**Trade-offs:**
- ✓ Portable across Intel GPUs (one binary runs on B60/B70/CRI).
- ✓ Required for the IGC shader-dump / stall-sampling profiling flow (IGC runs at run time — see
  `sycl-profiler`).
- ✗ First-run latency while the kernel is compiled (mitigate with `SYCL_CACHE_PERSISTENT=1`).
- ✗ May miss the last few % of architecture-specific tuning that AOT locks in.

---

## 4. Optimization flags

### General

| Flag | Purpose |
|------|---------|
| `-O2` | Standard optimization — **recommended default**. Device codegen is driven by IGC, so `-O3` rarely improves the kernel. |
| `-O3` | Aggressive host-side optimization; may raise register pressure/compile time. A **tuning** experiment, not the default. |
| `-gline-tables-only` | Profiling line info **without** disabling optimization. |
| `-g` | Full debug info (disables some optimizations). |
| `-DNDEBUG` | Disable assertions in release builds. |
| `-Wall -Wextra` | Enable compiler warnings. |

### SYCL-specific

| Flag | Purpose |
|------|---------|
| `-fsycl` | Enable SYCL compilation mode (compile + link). |
| `-fsycl-targets=<token>` | AOT device target(s) (see §2). |
| `-fsycl-device-code-split=per_kernel` | Split device code per kernel — faster JIT. |
| `-fsycl-dead-args-optimization` | Remove unused kernel arguments. |
| `-fno-sycl-id-queries-fit-in-int` | Allow index spaces > 2³¹. |

### Large GRF mode (tuning knob — leave to `sycl-optimization`)

Large GRF doubles registers per thread (256 vs 128) but **halves** resident threads per Xe-core. It
helps a kernel that would otherwise spill catastrophically (e.g. many matrix accumulators) and hurts
memory-latency-bound kernels with small register footprints. Adopt it only as a measured trial.

```bash
# JIT — pass the raw backend flag:
icpx -fsycl -O2 -Xs '-ze-opt-large-register-file' -o kernel kernel.cpp

# AOT — must use the -options prefix for ocloc:
icpx -fsycl -fsycl-targets=intel_gpu_bmg_g21 -O2 \
    -Xs "-options -ze-opt-large-register-file" -o kernel kernel.cpp
```

> **WARNING:** `-ftarget-register-alloc-mode=<target>:large` does **not** work for BMG targets — use
> the `-Xs` syntax above. AOT and JIT need different `-Xs` spellings (`-options …` for AOT).

---

## 5. Linking math / GEMM libraries

Route dense matmul/conv through a maintained, benchmark-verified library rather than hand-written XMX
(see [sycl-kernel-patterns.md](sycl-kernel-patterns.md)). Build all three with the same `icpx -fsycl`
toolchain and AOT target as the rest of the project.

### oneMKL (standard dense GEMM — the cuBLAS replacement)

```bash
icpx -fsycl -fsycl-targets=intel_gpu_bmg_g21 -O2 \
    -I${MKLROOT}/include \
    -L${MKLROOT}/lib -lmkl_sycl_blas -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core \
    -o my_app my_app.cpp
```

```cmake
find_package(MKL CONFIG)
target_link_libraries(my_app PRIVATE MKL::MKL_DPCPP)
```

### oneDNN (fused GEMM+epilogue / conv — the cuDNN replacement)

Use the **SYCL/DPC++ build** of oneDNN (built with `DNNL_GPU_RUNTIME=SYCL`). Link `-ldnnl` and create
the engine/stream from your SYCL queue via the interop API.

```bash
icpx -fsycl -fsycl-targets=intel_gpu_bmg_g21 -O2 \
    -I${DNNLROOT}/include -L${DNNLROOT}/lib -ldnnl \
    -o my_app my_app.cpp
```

```cmake
find_package(dnnl CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE DNNL::dnnl)
```

See [sycl-kernel-patterns.md](sycl-kernel-patterns.md) and `intel-gpu-software-repos.md` → `oneDNN` for the
SYCL-interop engine/stream setup and primitive descriptors.

### sycl-tla (CUTLASS-style tiled GEMM/attention)

**Header-only — no link step.** Add its include dir to the same `icpx -fsycl` build; the templates
resolve to Xe DPAS ops for the AOT target. Do **not** build the sycl-tla repo itself, and note it
needs a real Xe device (XMX) to run.

```bash
icpx -fsycl -fsycl-targets=intel_gpu_bmg_g21 -O2 \
    -I${SYCL_TLA_ROOT}/include -o my_app my_app.cpp
```

See [sycl-tla-patterns.md](sycl-tla-patterns.md) for the decision rule (plain SYCL vs sycl-tla) and
build wiring.

---

## 6. CMake / build wiring

Fold a SYCL rule into the project's **existing** build rather than duplicating a build tree. The
`sycl-migration` skill ships the canonical templates — use them as the source of icpx flags:
`templates/CMakeLists.sycl.txt` and `templates/toolchain-icpx.cmake`. Minimal shape:

```cmake
cmake_minimum_required(VERSION 3.20)
project(kernel LANGUAGES CXX)

set(CMAKE_CXX_COMPILER icpx)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_AOT "Enable Ahead-of-Time compilation" ON)
set(GPU_TARGET "intel_gpu_bmg_g21" CACHE STRING "AOT target: intel_gpu_bmg_g21 / intel_gpu_bmg_g31 / intel_gpu_cri")

set(SYCL_FLAGS "-fsycl")
if(USE_AOT)
    set(SYCL_FLAGS "${SYCL_FLAGS} -fsycl-targets=${GPU_TARGET}")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SYCL_FLAGS}")

add_executable(kernel kernel.cpp)
target_compile_options(kernel PRIVATE -O2 -Wall)
```

```bash
# Build/test always go through the runner:
.sycl/scripts/run.sh build "cmake -S sycl -B sycl/build -DUSE_AOT=ON -DGPU_TARGET=intel_gpu_bmg_g21 && cmake --build sycl/build -j"
.sycl/scripts/run.sh test  "./sycl/build/kernel"
```

---

## 7. Environment variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `ONEAPI_DEVICE_SELECTOR` | Pin the device backend | `level_zero:gpu` |
| `SYCL_CACHE_PERSISTENT` | Cache JIT kernels between runs | `1` |
| `SYCL_CACHE_DIR` | JIT cache directory | `/tmp/sycl_cache` |
| `IGC_ShaderDumpEnable` | Dump generated GEN assembly | `1` |
| `IGC_DumpToCustomDir` | Directory for shader dumps | `./shader_dump` |
| `ZE_FLAT_DEVICE_HIERARCHY` | Required for some metric profiling | `FLAT` |

`sycl-ls` (or `sycl-ls --verbose`) confirms the device is visible before building.

---

## 8. Common build errors and fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `command not found: icpx` | oneAPI not sourced | `source /opt/intel/oneapi/setvars.sh` |
| `error: unknown target 'intel_gpu_bmg'` | Old compiler / wrong target name | Use `intel_gpu_bmg_g21` (B60) or `intel_gpu_bmg_g31` (B70); or `-Xs "-device bmg-g21"` |
| `SYCL target is invalid: 'intel_gpu_bmg'` | Newer icpx rejects the generic BMG target | Use the die-specific `-fsycl-targets=intel_gpu_bmg_g21` |
| `undefined reference to 'sycl::...'` | Missing `-fsycl` at link | Add `-fsycl` to **both** compile and link |
| `Invalid option: -ze-opt-large-register-file` | Wrong `-Xs` syntax for AOT | AOT uses `-Xs "-options -ze-opt-large-register-file"`; JIT uses `-Xs '-ze-opt-large-register-file'` |
| `register spilling` (warning) | Too many registers used | Reduce local variables / simplify kernel; as a tuning trial, enable large GRF |
| `RetryManager recompilation` (AOT warning) | Kernel too large for normal GRF | Add `-Xs "-options -ze-opt-large-register-file"` |
| `Compilation from SPIR-V failed` | JIT issue with a specific feature | Build AOT for the target instead |
| `No device of requested type available` | No GPU / wrong backend | Check `sycl-ls`; set `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` |
