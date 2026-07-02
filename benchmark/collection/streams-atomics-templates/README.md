# streams, events, shared memory, atomics, templates, and macros collection

Owner: Weixuan Zhang

See `../README.md` for the shared workflow, CSV columns, `SOURCE.txt`
format, and snapshot rules. This file records the coverage matrix for
streams, events, shared memory, atomics, templates, and macros collection examples. Snapshots of every non-excluded candidate
are stored under `sources/<id>/`.

## Sources and licenses (verified 2026-07-02)

| Source | License | Notes |
|---|---|---|
| NVIDIA cuda-samples | NVIDIA CUDA Samples License | Current collection source. See the repository LICENSE and each candidate's `SOURCE.txt` for provenance information. |

## Coverage matrix

Dimensions (a candidate "covers" a cell if its kernels exercise it):

- **Feature** — primary CUDA language/runtime feature
- **Memory** — global, shared, managed, pinned, async allocation
- **Synchronization** — block, stream, event, atomic, graph execution
- **Programming model** — CUDA Runtime, WMMA/Tensor Core, CUDA Graphs, CCCL/libcu++
- **Implementation style** — plain kernels, template-heavy kernels, macro-driven configuration

| Candidate | Feature | Memory | Synchronization | Programming model | Implementation style |
|---|---|---|---|---|---|
| atomics-01 | device atomics | global | atomic | CUDA Runtime | intrinsic API |
| atomics-02 | system-scope atomics | managed | system atomic | CUDA Runtime | intrinsic API |
| atomics-03 | histogram atomics | shared + global | atomic | CUDA Runtime | library module |
| atomics-04 | threadFence reduction | shared + global | atomic + threadfence | CUDA Runtime | synchronization pattern |
| events-01 | events | pinned + global | event | CUDA Runtime | asynchronous execution |
| events-02 | CUDA Graphs | shared + global | graph + event | CUDA Graphs | graph execution |
| events-03 | CUDA Graphs | global | graph + event | CUDA Graphs | performance benchmark |
| shared-memory-01 | shared memory | shared | block | CUDA Runtime | tiled transpose |
| shared-memory-02 | dynamic shared memory | dynamic shared | block | CUDA Runtime | basic kernel |
| shared-memory-03 | shared-memory tiling | dynamic shared | block | WMMA | Tensor Core GEMM |
| shared-memory-04 | async shared-memory pipeline | dynamic shared | block | WMMA + `cuda::pipeline` | BF16 Tensor Core GEMM |
| shared-memory-05 | async shared-memory pipeline | dynamic shared | block | WMMA + `cuda::pipeline` | TF32 Tensor Core GEMM |
| streams-01 | stream priorities | global | stream + event | CUDA Runtime | multi-stream scheduling |
| streams-02 | asynchronous streams | pinned + global | stream + event | CUDA Runtime | overlap memcpy/kernel |
| streams-03 | HyperQ | shared + global | stream | CUDA Runtime | concurrent kernels |
| streams-04 | async memory pools | async allocation | stream + event | CUDA Runtime | memory management |
| streams-05 | IPC memory pools | async allocation | stream | CUDA Runtime | multi-process |
| streams-06 | overlapping transfers | pinned + global | stream + event | CUDA Runtime | memcpy overlap |
| templates-01 | template kernels | shared | block | CUDA Runtime | template specialization |
| templates-02 | CUDA C++ templates | shared | block | CUDA Runtime | generic programming |
| templates-03 | libcu++ RNG | global | block | libcu++ | template library |
| templates-04 | mdspan | shared + global | block | CCCL / libcu++ | template library |

### Dark cells / known gaps

- Warp-level synchronization (`__syncwarp`) as the primary feature rather than as part of WMMA.
- Cooperative Groups examples centered on synchronization primitives instead of memory management.
- Modern C++ (`cuda::std`) examples involving streams or atomics.
- Device-side C++ metaprogramming beyond libcu++/CCCL examples.
- Macro-heavy multi-file CUDA projects outside the NVIDIA ecosystem.

## Dedup policy

Many CUDA samples implement similar algorithms (e.g., GEMM or reduction), but expose different CUDA language features. Variants are retained only when they cover distinct feature combinations—for example, conventional shared-memory kernels versus WMMA Tensor Core implementations, or classic CUDA Runtime APIs versus modern CCCL/libcu++ abstractions. Algorithmic duplicates that do not expand feature coverage are excluded.
