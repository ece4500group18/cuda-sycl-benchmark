# overlap-copy-compute

**Class:** Host / dispatch overlap (host-side)  ·  **Priority:** High  ·  **Impact:** up to 2× when transfer-bound

## When to apply
The workload is **host–device transfer-bound**: the timeline shows H2D/D2H copies **serialized** with
kernels, and the GPU sits idle during transfers (or the host blocks on `.wait()` between copy and
compute). This is not a kernel-body problem — the kernel may be optimal — but the *schedule* leaves the
copy engine and compute engine idle in turn.

## Transformation
Overlap transfers with compute: **chunk** the data, use **asynchronous** USM copies, and let independent
work run concurrently via an **out-of-order queue** (or multiple in-order queues) with event
dependencies. While chunk *k* computes, chunk *k+1* copies in.

Before — copy-all, then compute, then copy-back (serial):
```cpp
q.memcpy(d_in, h_in, n * sizeof(float)).wait();     // whole H2D, GPU idle
q.parallel_for(/* process all n */ ...).wait();      // compute, copy engine idle
q.memcpy(h_out, d_out, n * sizeof(float)).wait();    // whole D2H, GPU idle
```

After — pipeline chunks on an out-of-order queue (copy overlaps compute):
```cpp
sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{} /* drop for OOO */};
const int CH = (n + NCHUNK - 1) / NCHUNK;
std::vector<sycl::event> done(NCHUNK);
for (int c = 0; c < NCHUNK; ++c) {
    const int off = c * CH, len = sycl::min(CH, n - off);
    auto h2d = q.memcpy(d_in + off, h_in + off, len * sizeof(float));          // async H2D
    auto ker = q.parallel_for(sycl::nd_range<1>(round_up(len), wg),
                              h2d, [=](sycl::nd_item<1> it){ /* process chunk c */ });
    done[c] = q.memcpy(h_out + off, d_out + off, len * sizeof(float), ker);    // async D2H
}
sycl::event::wait(done);   // single sync at the end; chunk k compute overlaps chunk k+1 copy
```
Use **pinned/host USM** (`malloc_host`) for `h_in`/`h_out` to make the async copies faster.

## Correctness invariants
- Each chunk's kernel must depend on its **own** H2D event, and its D2H on its **own** kernel event — get
  the event chain right or you copy stale/partial data.
- Chunk boundaries must tile the full range exactly once; keep the per-kernel tail guard.
- One final `wait` before reading `h_out`; do not read a chunk's output before its D2H event completes.

## Verify it took effect
- Timeline shows copy and compute **overlapping** across chunks; GPU idle gaps during transfers shrink.
- End-to-end wall time drops for transfer-bound cases even though per-kernel time is unchanged.

## Pitfalls / conflicts
- **Conflicts:** none intrinsic — but too many tiny chunks add per-submit overhead; too few kill the
  overlap. Tune `NCHUNK`.
- **Synergizes:** `sycl-graphs` (record the per-chunk pipeline and replay it across iterations).
- This is a **host-side** strategy — it changes scheduling, not the kernel. Apply after the kernel itself
  is optimized, when profiling shows transfer, not compute, is the limiter.
