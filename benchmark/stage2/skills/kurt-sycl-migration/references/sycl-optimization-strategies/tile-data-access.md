# tile-data-access

**Class:** Cache  ·  **Priority:** Critical  ·  **Impact:** 20–60% (esp. BMG GDDR6)

## When to apply
L2/L3 hit rate < 0.60 while traversing a **regular 2D/3D neighborhood** (matmul, stencil, image filter,
transpose). A naive row-major sweep evicts data before it is reused, so the working set thrashes the
cache. Blocking the iteration into cache-sized tiles restores temporal locality.

## Transformation
Restructure the loops to process one **tile** (a small 2D/3D block) at a time so each loaded cache line
is fully reused before eviction. This is the cache-level counterpart to `prefetch-to-slm` (which then
stages the tile on-chip).

Before — full-row sweep, poor reuse:
```cpp
q.parallel_for(sycl::nd_range<2>({M, N}, {1, wgx}), [=](sycl::nd_item<2> it) {
    const int r = it.get_global_id(0), c = it.get_global_id(1);
    if (r >= M || c >= N) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += A[r * K + k] * B[k * N + c];  // B column strides → thrash
    C[r * N + c] = acc;
});
```

After — tiled iteration (block the K/N dims to cache-friendly tiles):
```cpp
constexpr int T = 32;                                   // tile ~ fits L1/L2 working set
q.parallel_for(sycl::nd_range<2>({M, N}, {T, T}), [=](sycl::nd_item<2> it) {
    const int r = it.get_global_id(0), c = it.get_global_id(1);
    float acc = 0.f;
    for (int k0 = 0; k0 < K; k0 += T)                   // walk K in tiles: A/B lines reused within
        for (int k = k0; k < sycl::min(k0 + T, K); ++k)
            if (r < M && c < N) acc += A[r * K + k] * B[k * N + c];
    if (r < M && c < N) C[r * N + c] = acc;
});
```

## Correctness invariants
- Tiling only reorders independent accumulations — the result must be identical (mind float
  reassociation tolerance).
- Handle partial edge tiles (`min(k0 + T, K)`); keep bounds guards on `r`/`c`.
- Choose `T` so the tile working set fits the target cache level — measure, don't assume.

## Verify it took effect
- L2/L3 hit rate rises above the trigger; DRAM read bytes (`MemoryProfile`) drop.
- `SendStall`/`SbidStall` from cache misses shrinks.

## Pitfalls / conflicts
- **Conflicts:** none directly, but tiling adds index/loop overhead — it must be repaid by hit-rate gain.
- **Synergizes:** `prefetch-to-slm`, `slm-cache-reuse` (stage the tile on-chip once cache-blocked).
- This is often an **enabling transform**: tiling alone can add barriers/overhead and only pays once a
  downstream SLM strategy exploits the tile.
