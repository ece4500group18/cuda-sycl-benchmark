# prefetch-to-slm

**Class:** Memory-latency  ·  **Priority:** High  ·  **Impact:** 20–50% (kernels with reuse)

## When to apply
L2/L3 hit rate < 0.60 (`DeviceCacheProfile` / `L3`) **and** the kernel already has a tile loop with
**predictable reuse** — e.g. a tiled GEMM/stencil where every work-item in the group rereads the same
global tile from DRAM. Staging the tile in SLM once turns N global reads into one.

## Transformation
Cooperatively load the next global tile into a `local_accessor`, barrier, then compute from SLM. Classic
tiled matmul shape.

Before — every work-item streams A/B from global each k-step:
```cpp
q.parallel_for(sycl::nd_range<2>({P, P}, {TILE, TILE}), [=](sycl::nd_item<2> it) {
    const int row = it.get_global_id(0), col = it.get_global_id(1);
    float sum = 0.f;
    for (int k = 0; k < n; ++k)
        sum += A[row * n + k] * B[k * n + col];   // repeated DRAM reads, low L3 hit
    if (row < n && col < n) C[row * n + col] = sum;
});
```

After — stage each tile in SLM, reuse across the work-group:
```cpp
q.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 2> As({TILE, TILE}, h), Bs({TILE, TILE}, h);
    h.parallel_for(sycl::nd_range<2>({P, P}, {TILE, TILE}), [=](sycl::nd_item<2> it) {
        const int row = it.get_global_id(0), col = it.get_global_id(1);
        const int lr = it.get_local_id(0), lc = it.get_local_id(1);
        float sum = 0.f;
        for (int t = 0; t < P; t += TILE) {
            As[lr][lc] = (row < n && t + lc < n) ? A[row * n + t + lc] : 0.f;
            Bs[lr][lc] = (col < n && t + lr < n) ? B[(t + lr) * n + col] : 0.f;
            sycl::group_barrier(it.get_group());        // tile is ready
            for (int k = 0; k < TILE; ++k)
                sum += As[lr][k] * Bs[k][lc];           // reads hit SLM, not DRAM
            sycl::group_barrier(it.get_group());        // before overwriting the tile
        }
        if (row < n && col < n) C[row * n + col] = sum;
    });
});
```

## Correctness invariants
- **Two barriers per tile:** one after filling SLM (before use), one after use (before the next fill) —
  dropping either is a data race.
- Pad partial tiles with the loop's identity (`0.f` for a sum) so out-of-range lanes are harmless.
- SLM footprint = `2 * TILE * TILE * sizeof(elem)`; keep it under the occupancy budget.

## Verify it took effect
- L2/L3 hit rate rises; DRAM read bytes (`MemoryProfile`) drop by ~the reuse factor.
- `SbidStall`/`SendStall` from global loads shrinks; a modest `SyncStall` from the barriers appears
  (expected — net win if reuse is high).

## Pitfalls / conflicts
- **Conflicts:** `reduce-slm-allocation`, `tune-work-group-size` — the SLM tile can cap residency; if
  occupancy drops below the pre-tiling level, shrink `TILE` or fall back.
- **Synergizes:** `slm-cache-reuse`, `tile-data-access`, `double-buffer-slm` (hide the fill latency).
- **Anti-pattern reminder:** do **not** stage *read-only, non-reused* vectors in SLM — Intel L1 already
  caches those; SLM staging only pays when many work-items reuse the tile.
