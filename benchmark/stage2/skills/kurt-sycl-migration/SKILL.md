---
name: sycl-migration
description: >
  Hand-write SYCL 2020 kernels from an analyzed CUDA or Triton algorithm for Intel Xe2/Xe3
  (Battlemage Arc Pro B-series / Crescent Island) GPUs, wire them in as a **backend that mirrors the
  CUDA structure** (same file names, launcher signatures, shared host/driver code, extended build —
  not a standalone SYCL fork), and validate correctness with the project's **inherited** test harness
  against a CPU-reference oracle. Migrations must be **idiomatic**: parallel reductions (group
  collectives, not serial loops) and library GEMM (oneMKL/oneDNN, or sycl-tla). Targets plain SYCL by
  default, or sycl-tla for tiled tensor algebra. Use when: translating a CUDA or Triton kernel to
  SYCL, writing a SYCL kernel, choosing plain-SYCL vs sycl-tla, adding a SYCL backend to an
  icpx/CMake/Make build, authoring a correctness test, establishing a CPU reference. NEVER uses
  dpct/SYCLomatic — kernels are hand-written from the algorithm. Optimization is out of scope (use
  sycl-optimization).
---

# SYCL Migration

Produce a **correct, idiomatic** SYCL 2020 implementation of one source kernel (CUDA *or* Triton),
built and unit-tested. Correctness first — but *idiomatic* means the migration expresses the **same
parallel structure and library usage** the source did: a CUDA reduction migrates to a group-collective
reduction (not a serial loop), and a cuBLAS/cuDNN matmul migrates to a **library GEMM** (oneMKL/oneDNN,
or sycl-tla), not a naive triple loop. Tile-shape / launch-config *tuning* is the `sycl-optimization`
skill's job; choosing the right *algorithm and library* is migration's job. A correctness-only stub
that throws away the source's parallelism (serial reduce, hand-rolled GEMM loop) is **not** an
acceptable migration.

## Hard rules
- **NEVER** use `dpct`, `SYCLomatic`, or any auto-translator. Understand the algorithm (from the
  `cuda-analysis` or `triton-analysis` detail) and hand-write the kernel.
- **SYCL 2020 only** (`sycl::queue`, `nd_range`, USM, `reduce_over_group`, `local_accessor`,
  sub-group collectives). No deprecated APIs.
- **One kernel at a time**: translate → build → unit test → regression → commit. Then the next.
- All builds/tests run through `.sycl/scripts/run.sh` (local or remote GPU host) — never call `icpx`
  directly.

## Required reading
- `sycl-reference` → `references/sycl-kernel-patterns.md` (mapping tables + canonical patterns, the
  `joint_matrix` authoring API notes, and the correctness-validation tolerance table).
- **If the source is Triton**: `sycl-reference` → `references/triton-patterns.md` (program/tile model,
  block pointers + masks, `tl.dot`→XMX, reductions, atomics).
- **If targeting sycl-tla**: `sycl-reference` → `references/sycl-tla-patterns.md` (decision rule,
  collective mainloop/epilogue, build wiring).
- `sycl-reference` → `references/intel-gpu-hardware.md` (sub-group width, SLM, int32 context) and
  `references/sycl-build-guide.md` (the icpx/AOT build & compile flag reference).
- The kernel's `.sycl/state/kernels/<id>.json` (algorithm, source constructs, IO spec, `source_lang`).

When a mapping is non-obvious, consult `sycl-reference` → `references/intel-gpu-software-repos.md` for how a
comparable kernel is written in a local upstream repo (oneDNN, sycl-tla, PyTorch XPU, vLLM XPU,
intel-xpu-backend-for-triton) — extract the pattern, never copy source.

## Procedure (per kernel)

### 1. Establish a correctness reference (CPU oracle)
The **CPU reference is the accuracy oracle**: a host implementation of the kernel's math, computed at
equal-or-higher precision (accumulate in fp64 / careful fp32) so it is the tighter, ground-truth
result. SYCL is compared against it **live** on the runner every test run — there is **no golden
tensor dump**. Inputs are regenerated identically on each side from a stored **seed + shape + dtype**
(a few bytes in the kernel detail), never persisted as data. This is **source-agnostic**: it works
identically whether the original kernel is CUDA or Triton.

Reference priority:
1. **Reuse the project's existing reference AND its test harness** *(preferred — inherit, don't
   reinvent)* — if the project already has kernel tests that compare the original kernel against a
   reference, that reference *and its pass/fail criteria are already written and validated*. Do **not**
   rewrite them and do **not** invent a new tolerance: point the SYCL test at the *same* reference,
   the *same* input generation, and the *same* comparison/tolerance, so the SYCL kernel is gated
   **exactly** the way the original was. The SYCL test simply swaps the original call for the SYCL call.
   - For **CUDA** projects the canonical case is a **per-kernel dev harness** (e.g. `dev/cuda/<k>.cu`
     with a CPU reference like `<k>_cpu(...)`, a shared checker like `validate_result(...)` /
     `common.h`, and a fixed tolerance). **Mirror it**: create the SYCL harness at the parallel path
     (`dev/sycl/<k>.cpp`), `#include`/reuse the *same* CPU reference and the *same* `validate_result`
     + tolerance from `common.h` (port `common.h` to a backend-neutral `common.hpp` **once**, keeping
     the identical checker signature and constants), and change only the device call. Reuse the same
     kernel-version selection / input shapes so the SYCL test is a drop-in of the CUDA one. **Never**
     substitute a looser tolerance or a hand-rolled max-diff check.
   - For **Triton** projects this is usually the **PyTorch eager reference** used by the kernel's test
     (e.g. `torch.testing.assert_close(triton_out, torch_ref)`) — run that torch reference on **CPU**
     to be the oracle. Reuse its tolerance.
2. **CPU reference** *(default, when none exists)* — straightforward host implementation of the same
   math; the live comparison target for SYCL.
3. **Analytical** — closed-form expected output where the math allows (a special case of the oracle).

The original kernel is **not** a per-run dependency. It only *validates the CPU reference once*: if a
capable host is reachable, run the **original kernel** (the CUDA kernel, or the Triton kernel / its
torch reference) vs the CPU reference on the seeded inputs a single time to confirm the CPU oracle
faithfully reproduces the kernel semantics, then record `reference.crosscheck: pass` (and
`reference.crosscheck_source: cuda|triton`) in the kernel detail. Never gate the original against SYCL
directly — both are judged against the CPU oracle, so fp error doesn't stack (triangle inequality). If
no capable host exists, the CPU reference still stands on its own (validated by review / analytical
reasoning).

If no trustworthy reference can be built, set the kernel `status: needs-reference`, log it, and
surface to the orchestrator. **Do not fabricate a passing test.**

Set a tolerance appropriate to dtype and accumulation (e.g. `rtol=1e-5, atol=1e-6` for fp32; looser
for fp16/bf16 and for mixed-precision `tl.dot`/XMX accumulate; must cover reduction-reordering /
non-associativity). See the dtype tolerance table + the `atol·sqrt(K)` accumulation rule in
`sycl-kernel-patterns.md` → *Correctness validation*. Record the seed, input spec, and tolerance in the
kernel detail.

### 2. Choose the target style (plain-sycl vs sycl-tla)
Default is **plain-sycl**. Choose **sycl-tla** only for tiled tensor algebra (GEMM, GEMM+epilogue,
conv-as-GEMM, attention) that is XMX-bound and has a comparable sycl-tla example — see the full
decision rule in `sycl-reference` → `references/sycl-tla-patterns.md`. Even for sycl-tla candidates,
the **pragmatic path** is: migrate a correct plain-SYCL version first (fast to verify), and defer the
sycl-tla re-target to `sycl-optimization` if it falls short of roofline. Record the decision as
`target_style` in the kernel detail with a one-line reason.

### 3. Translate to SYCL 2020
Apply the mapping tables. Guidance:
- Prefer `int32` index math where dimensions fit; map multidimensional problems to `nd_range<2/3>`.
- Do **not** force a sub-group size on simple kernels; only for collectives/XMX/tuned reductions.
- Map warp shuffles to sub-group collectives; `__shared__` to `local_accessor`; atomics to
  `sycl::atomic_ref`; cuBLAS/cuDNN/Thrust to oneMKL/oneDNN/oneDPL.
- **Reductions are REQUIRED to be parallel.** Any kernel whose source uses warp/block reduction,
  `cub::BlockReduce`, cooperative groups, or is a norm/softmax/mean-var/`reduce` → use the parallel
  **reduction recipe** in `sycl-kernel-patterns.md` (one work-group per row + `reduce_over_group`, or a
  grid-stride global reduce). A single work-item looping over the reduced axis is **not** an
  acceptable migration.
- **Matmul/GEMM is REQUIRED to use a library.** Any kernel whose source calls cuBLAS/cuBLASLt/cuDNN
  /CUTLASS (or is a tiled GEMM) → use the **GEMM recipe** in `sycl-kernel-patterns.md` (oneMKL, oneDNN
  for fused ops, or sycl-tla per the decision rule). A naive triple-loop matmul is **not** an
  acceptable migration.
- For a **Triton** source, follow `triton-patterns.md`: one Triton program → one work-group that owns
  a tile; `BLOCK_*` → tile/loop bounds; masked `tl.load/store` → index math + bounds guards;
  `tl.dot` → `joint_matrix`/oneMKL (plain SYCL) or the sycl-tla collective mainloop. Do **not** port
  the autotuner or `num_warps`/`num_stages` — pick one correct launch config now; tuning is
  `sycl-optimization`'s job.
- Keep the original source open to preserve intent; translate the *algorithm*, not line-by-line
  syntax.

### 4. Wire SYCL in as a backend that MIRRORS the CUDA structure (cross-platform seam)
The goal is a **cross-platform application**, not a standalone SYCL fork. SYCL is added as a second
**backend** behind the *same* host/driver code, file names, function signatures, and call sites as
CUDA — only the device layer swaps. Do **not** create a divergent `sycl/` project with its own
`main`, its own headers, and its own tolerances.

**Mirror the CUDA layout:**
- **Kernel headers** — if CUDA kernels live in per-kernel headers (e.g. `llmc/<k>.cuh` exposing an
  inline host launcher `<k>(...)`), add the SYCL counterpart with the **same name and signature**
  (e.g. `llmc/<k>.hpp` or a backend-guarded section), so callers are unchanged. Keep the launcher
  signature identical (same args, same order) — the host code must not know which backend it calls.
- **Shared host/driver code** — reuse the existing driver (`train_gpt2.c`-style: dataloader, tokenizer,
  optimizer glue, allocation, the train/eval loop). Introduce a thin **backend seam** so the driver
  is compiled once and links against either the CUDA or the SYCL kernel implementations. Do not fork
  the driver into a separate SYCL `main`.
- **Backend switch** — select the backend at **compile time** (a build flag / macro that includes the
  CUDA or the SYCL kernel headers and swaps `cudaMalloc/cudaMemcpy/stream` for the SYCL USM/queue
  equivalents behind the same small wrappers). Keep the switch in one place (a `common`/`backend`
  header), not scattered per call site.
- **Build** — **extend the project's existing build** (e.g. add a SYCL target/rule to the `Makefile`
  or CMake) rather than standing up an unrelated build tree. Use `templates/toolchain-icpx.cmake` /
  `templates/CMakeLists.sycl.txt` only as a source of the icpx flags to fold into the existing build;
  prefer adding an `icpx -fsycl` rule alongside the `nvcc` rule so `make <target>` /
  `make <target>_sycl` are siblings.
- **Tests** — mirror the CUDA per-kernel harness at the parallel path (`dev/sycl/<k>.cpp` beside
  `dev/cuda/<k>.cu`) reusing the **same** CPU reference + `validate_result` + tolerance (see step 1),
  and add its build rule beside the CUDA one.
- **If `target_style` is `sycl-tla`**: add the sycl-tla include dir (header-only — no extra link)
  from `.sycl/config.json` → `targets.sycl_tla.include` (fallback: the `reference_repos` sycl-tla
  path) to the *same* extended build. See `sycl-tla-patterns.md` → "Build & include wiring".

**Only when the CUDA structure genuinely cannot be mirrored** (e.g. a pure-Python Triton project with
no C/C++ host to share) fall back to a self-contained SYCL subtree — but still name files and tests to
match the source kernels and reuse the source's reference/tolerance. Record the backend seam / build
location and the shared-vs-forked host decision in `.sycl/state/project.json` (`sycl_build`).

### 5. Build + unit test (via the runner)
Use the project's **extended** build (the SYCL rule you added beside the CUDA one), not a separate
tree. The icpx/AOT flag reference (targets, JIT vs AOT, oneMKL linking, common errors) is in
`sycl-reference` → `references/sycl-build-guide.md`. Examples — adapt to
whatever the project uses:
```bash
# Makefile-based project (mirrors `make <k>` in dev/cuda):
.sycl/scripts/run.sh build "make -C dev/sycl <id>"
.sycl/scripts/run.sh test  "./dev/sycl/<id>"          # same args/version-select as the CUDA harness
# CMake project that already exists: add the SYCL target to it and build that target
.sycl/scripts/run.sh build "cmake --build build -j --target <id>_sycl"
```
Iterate until the test passes within the **inherited** tolerance (step 1) — do not loosen it. Then
run previously passing kernel tests (regression); fix any breakage.

### 6. Record state + commit
- Update `.sycl/state/kernels/<id>.json`: `status: "migrated"`, `sycl_impl` path, `unit_test`
  (path, result, tolerance), `reference.kind/path`, and the recorded `source_lang` + `target_style`.
- Update `.sycl/state/kernels/index.json` status.
- `bash .sycl/scripts/gen-progress.sh`
- `.sycl/scripts/log.sh sycl-agent migrate info "kernel <id> migrated, test pass"`
- Commit code + state together (the orchestrator handles the git commit).
- If you learned a durable mapping trick, append it (with provenance) to `.sycl/state/lessons.md`.

## Output (return to orchestrator)
- Kernel `<id>`: migrated | needs-reference | blocked.
- Unit test: pass/fail + tolerance; reference kind used.
- Build status; any regression fixed.
- Next action: next kernel id or "migration complete".

## Assets
- `templates/CMakeLists.sycl.txt` — icpx/SYCL build flags to **fold into the project's existing
  build** (source of the `-fsycl` flags / sycl-tla include dir); use standalone only when the CUDA
  structure genuinely cannot be mirrored.
- `templates/toolchain-icpx.cmake` — icpx toolchain file.
- `templates/unit_test.cpp` — CPU-oracle comparison harness (rtol/atol), source-agnostic — use only
  when the project has **no** existing test harness to inherit; otherwise mirror the project's harness
  (e.g. `dev/cuda` → `dev/sycl`) and reuse its reference + tolerance.
- `scripts/compare_npy.py` — compare a produced output against a reference with tolerances.

Required migration recipes live in `sycl-reference` → `references/sycl-kernel-patterns.md`: the parallel
**reduction** recipe (row-per-work-group; never a serial loop) and the **GEMM** recipe (oneMKL/oneDNN
library call; never a naive triple loop).
