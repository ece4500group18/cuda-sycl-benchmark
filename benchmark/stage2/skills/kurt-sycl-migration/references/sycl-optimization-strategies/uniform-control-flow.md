# uniform-control-flow

**Class:** Control-flow  ·  **Priority:** Medium  ·  **Impact:** 10–30%

## When to apply
SIMD efficiency < 0.70 (lanes-enabled / lanes-total) or clear **divergence** symptoms: work-items in the
same sub-group take different branches based on their data, so the hardware serializes the divergent
paths and masks off idle lanes. The goal is to make branch decisions **uniform within a sub-group**.

## Transformation
Reorganize so lanes in a sub-group take the **same** path — sort/bucket work by branch outcome, hoist a
loop-invariant condition to a uniform (per-group) decision, or partition the launch by case.

Before — data-dependent path diverges within the sub-group:
```cpp
q.parallel_for(sycl::nd_range<1>(g, wg), [=](sycl::nd_item<1> it) {
    const int i = it.get_global_id(0);
    if (i >= n) return;
    if (type[i] == 0) dst[i] = pathA(src[i]);   // neighbors disagree → both paths run
    else              dst[i] = pathB(src[i]);
});
```

After — bucket by `type` so each launch/sub-group is uniform:
```cpp
// Host: partition indices into idxA / idxB by type once (or pre-sorted upstream).
q.parallel_for(sycl::nd_range<1>(gA, wg), [=](sycl::nd_item<1> it) {   // all type 0
    const int t = it.get_global_id(0);
    if (t < nA) { int i = idxA[t]; dst[i] = pathA(src[i]); }           // uniform path
});
q.parallel_for(sycl::nd_range<1>(gB, wg), [=](sycl::nd_item<1> it) {   // all type 1
    const int t = it.get_global_id(0);
    if (t < nB) { int i = idxB[t]; dst[i] = pathB(src[i]); }           // uniform path
});
```

## Correctness invariants
- The partition must cover **every** element exactly once across the buckets (no drops, no duplicates).
- Bucketed launches must produce the same `dst[i]` as the merged kernel for each `i`.
- If uniformity comes from a hoisted invariant, confirm it is truly uniform per sub-group/group.

## Verify it took effect
- SIMD efficiency rises above the trigger; `ControlStall` from divergence drops.
- Per-lane active-mask utilization improves in the profile.

## Pitfalls / conflicts
- **Conflicts:** `coalesce-memory-access` — bucketing by branch outcome can scatter memory accesses and
  hurt locality; re-profile bandwidth after, and re-coalesce if needed.
- **Synergizes:** `eliminate-branches` (for the cheap cases predicate instead of bucketing).
- The partitioning/sort has a cost — it must be repaid by the divergence removed; skip it for mildly
  divergent kernels.
