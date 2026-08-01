# double-buffer-slm

**Class:** Synchronization  ·  **Priority:** Critical  ·  **Impact:** 20–40%

## When to apply
`SyncStall` is the dominant `VectorEngineStalls` entry in a **tiled load→compute loop**: the whole
work-group barriers, waits for the next tile to load, then computes, idling at the barrier each
iteration. Double-buffering overlaps the *next* tile's load with the *current* tile's compute.

## Transformation
Allocate **two SLM buffers**; while computing from buffer *cur*, load the next tile into buffer *nxt*,
then swap. This removes the load-wait barrier from the critical path.

Before — single buffer, barrier stalls every tile:
```cpp
sycl::local_accessor<float, 1> t(sycl::range<1>(TILE), h);
for (int s = 0; s < S; ++s) {
    t[lid] = g[s * TILE + lid];
    sycl::group_barrier(grp);            // wait for load  → SyncStall
    acc += use(t[lid]);
    sycl::group_barrier(grp);            // wait before overwrite → SyncStall
}
```

After — two buffers, load(next) overlaps compute(cur):
```cpp
sycl::local_accessor<float, 2> t({2, TILE}, h);       // double buffer
int cur = 0;
t[cur][lid] = g[0 * TILE + lid];                       // prologue load
sycl::group_barrier(grp);
for (int s = 0; s < S; ++s) {
    int nxt = cur ^ 1;
    if (s + 1 < S) t[nxt][lid] = g[(s + 1) * TILE + lid];  // load next while...
    acc += use(t[cur][lid]);                                // ...computing current
    sycl::group_barrier(grp);                               // single barrier / iter
    cur = nxt;
}
```

## Correctness invariants
- The prologue loads tile 0 exactly once before the loop; the look-ahead load is guarded (`s + 1 < S`).
- The barrier must sit **after** issuing the next load and the current compute so both are visible before
  the swap — one barrier per iteration.
- SLM footprint **doubles** — recheck the occupancy/SLM budget (see `reduce-slm-allocation`).

## Verify it took effect
- `SyncStall` share drops sharply; barrier idle time falls.
- The load and compute of adjacent tiles overlap in the timeline.

## Pitfalls / conflicts
- **Conflicts:** `reduce-slm-allocation` (double buffering doubles SLM), `switch-grf-mode:small`.
- **Synergizes:** `prefetch-to-slm` (the two together fully hide fill latency).
- Only worth it when `SyncStall` is actually load-wait dominated; if barriers come from a reduction,
  prefer `reduce-barriers`/`subgroup-reduction`.
