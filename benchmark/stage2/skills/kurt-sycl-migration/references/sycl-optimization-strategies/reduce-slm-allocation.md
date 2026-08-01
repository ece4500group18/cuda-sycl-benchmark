# reduce-slm-allocation

**Class:** Occupancy-bound  ·  **Priority:** Medium  ·  **Impact:** 10–40% when occupancy-limited

## When to apply
SLM per work-group **exceeds ~32 KB** and only **one work-group fits per Xe-core**, so SLM — not
registers — is the residency limiter. Shrinking the SLM footprint lets a second work-group co-reside.

## Transformation
Cut the SLM footprint: use a smaller tile, stage only the data with genuine reuse, reuse one SLM buffer
across phases instead of allocating several, or narrow the SLM element type.

Before — two full 32-bit tiles, 32 KB+/group → 1 group resident:
```cpp
sycl::local_accessor<float, 2> As({64, 64}, h);   // 16 KB
sycl::local_accessor<float, 2> Bs({64, 64}, h);   // 16 KB  → 32 KB total
```

After — smaller tiles (and/or narrower type) → 2 groups resident:
```cpp
sycl::local_accessor<float, 2> As({32, 32}, h);   // 4 KB
sycl::local_accessor<float, 2> Bs({32, 32}, h);   // 4 KB  → 8 KB total, higher occupancy
// or: store the tile as sycl::bfloat16 to halve bytes when precision allows
```

## Correctness invariants
- Shrinking the tile changes the loop trip count / number of tiles — keep the tiling loop and boundary
  padding consistent.
- Reusing one SLM buffer across phases needs a barrier between phases so no phase reads another's data.
- Results unchanged; this is a footprint edit.

## Verify it took effect
- SLM/WG drops below the threshold; **two+ work-groups** now reside per Xe-core and occupancy rises.
- Net speedup even if each group does slightly more iterations (smaller tiles).

## Pitfalls / conflicts
- **Conflicts:** `prefetch-to-slm`, `slm-cache-reuse`, `double-buffer-slm`, `pad-slm-arrays` — these all
  *add* SLM; this strategy trims it. Balance staging benefit against occupancy.
- **Synergizes:** `tune-work-group-size` (re-tune WG once a second group fits).
- Cutting SLM too far can push data back to DRAM and re-create a bandwidth/latency stall — measure the
  trade.
