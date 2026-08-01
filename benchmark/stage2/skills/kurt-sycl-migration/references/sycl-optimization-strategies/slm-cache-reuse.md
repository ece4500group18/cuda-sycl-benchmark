# slm-cache-reuse

**Class:** Memory-latency  ·  **Priority:** High  ·  **Impact:** 20–60%

## When to apply
L2/L3 hit rate < 0.60 **and** the same neighborhood is reused by many work-items in one work-group — a
stencil/convolution where each source element feeds several outputs. Without staging, each source
element is fetched once *per output point* instead of once *per tile*.

## Transformation
Load a tile **plus its halo** into SLM once, barrier, then read all neighbors from SLM.

Before — each output re-reads its whole neighborhood from global:
```cpp
q.parallel_for(sycl::nd_range<1>(global, wg), [=](sycl::nd_item<1> it) {
    const int i = it.get_global_id(0);
    if (i >= n) return;
    float sum = 0.f;
    for (int k = -2; k <= 2; ++k) {                 // 5 global reads / output, heavy overlap
        int idx = sycl::clamp(i + k, 0, n - 1);
        sum += in[idx] * w[k + 2];
    }
    out[i] = sum;
});
```

After — stage tile+halo in SLM, each element fetched once:
```cpp
constexpr int HALO = 2;
q.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> t(sycl::range<1>(wg + 2 * HALO), h);
    h.parallel_for(sycl::nd_range<1>(global, wg), [=](sycl::nd_item<1> it) {
        const int gid = it.get_global_id(0), lid = it.get_local_id(0);
        const int base = it.get_group(0) * wg;
        t[lid + HALO] = in[sycl::min(base + lid, n - 1)];        // center
        if (lid < HALO) {                                       // halos
            t[lid]                = in[sycl::max(base + lid - HALO, 0)];
            t[wg + HALO + lid]    = in[sycl::min(base + wg + lid, n - 1)];
        }
        sycl::group_barrier(it.get_group());
        if (gid < n) {
            float sum = 0.f;
            for (int k = -2; k <= 2; ++k) sum += t[lid + HALO + k] * w[k + 2];  // SLM reads
            out[gid] = sum;
        }
    });
});
```

## Correctness invariants
- The halo width must equal the stencil radius; clamp halo indices at the array bounds.
- One barrier between fill and use; the tile is read-only afterward so no second barrier is needed here.
- Boundary handling (clamp/reflect/zero) must match the original kernel exactly.

## Verify it took effect
- L2/L3 hit rate rises; DRAM read bytes drop toward `1×` the tile size (from `~stencil-width×`).
- `SendStall` from redundant global reads shrinks.

## Pitfalls / conflicts
- **Conflicts:** `reduce-slm-allocation`; `pad-slm-arrays` if padding pushes the tile into the next SLM
  bucket and cuts residency.
- **Synergizes:** `prefetch-to-slm`, `tile-data-access`.
- **Anti-pattern reminder:** staging *read-only, non-reused* data in SLM is neutral-to-negative on Intel
  (L1 already caches it) — only stage genuinely reused neighborhoods.
