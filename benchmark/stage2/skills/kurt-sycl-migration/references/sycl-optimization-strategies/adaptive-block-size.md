# adaptive-block-size

**Class:** Occupancy-bound  ·  **Priority:** Medium  ·  **Impact:** up to 2× on small dims

## When to apply
A **single hardcoded work-group / block size** is used across very different problem shapes, so small
dimensions waste work-items (most lanes masked off) while large ones are fine. The fix is to choose the
launch size **per shape** instead of one-size-fits-all.

## Transformation
Select the work-group size (and, where relevant, the sub-group size) from the problem dimensions at
launch — small groups / sub-group-sized launches for tiny dims, 256+ for large.

```cpp
static int pick_wg(int n) {
    if (n <= 64)   return 16;    // sub-group sized: don't spawn 256 lanes for 64 elements
    if (n <= 256)  return 64;
    if (n <= 4096) return 128;
    return 256;                  // large: the usual sweet spot
}
void run(sycl::queue& q, const float* src, float* dst, int n) {
    const int wg = pick_wg(n);
    const int global = ((n + wg - 1) / wg) * wg;
    q.parallel_for(sycl::nd_range<1>(global, wg), [=](sycl::nd_item<1> it) {
        const int i = it.get_global_id(0);
        if (i < n) dst[i] = /* ... */ src[i];
    });
}
```

## Correctness invariants
- Every chosen size must stay a multiple of the sub-group width and within `max_work_group_size`.
- The kernel body is identical across sizes — only the launch geometry changes.
- Keep the tail guard; `global` rounds up for every branch.

## Verify it took effect
- On small shapes, SIMD lane utilization / occupancy rises and time drops; large shapes unchanged.
- No single shape regresses versus the previous fixed size.

## Pitfalls / conflicts
- **Conflicts:** none.
- **Synergizes:** `tune-work-group-size` (this is its per-shape generalization).
- Keep the selection table small and data-driven; if a kernel uses SLM sized from WG, make sure each
  branch's SLM footprint is valid.
