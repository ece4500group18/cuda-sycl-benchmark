# lock-free-atomics

**Class:** Synchronization  ·  **Priority:** Medium  ·  **Impact:** 5–20× on reduction-heavy

## When to apply
`SyncStall` is high with **global atomics**, or L3 atomic traffic is elevated — many work-items
contend on a few global atomic addresses (histogram bins, a global accumulator, a counter). The
serialization is at the atomic, not a barrier.

## Transformation
Cut global contention with a **privatization / hierarchical** pattern: accumulate locally first
(registers → sub-group reduce → one SLM atomic per group), then do a **single global atomic per
work-group** instead of one per work-item.

Before — every item hits the global atomic:
```cpp
q.parallel_for(sycl::nd_range<1>(g, wg), [=](sycl::nd_item<1> it) {
    const int i = it.get_global_id(0);
    if (i < n) {
        int bin = classify(src[i]);
        sycl::atomic_ref<int, sycl::memory_order::relaxed,
                         sycl::memory_scope::device>(hist[bin]).fetch_add(1);  // global contention
    }
});
```

After — sub-group/group privatize, one global atomic per group per bin:
```cpp
q.submit([&](sycl::handler& h) {
    sycl::local_accessor<int, 1> lhist(sycl::range<1>(NBINS), h);
    h.parallel_for(sycl::nd_range<1>(g, wg), [=](sycl::nd_item<1> it) {
        const int lid = it.get_local_id(0);
        for (int b = lid; b < NBINS; b += it.get_local_range(0)) lhist[b] = 0;
        sycl::group_barrier(it.get_group());
        const int i = it.get_global_id(0);
        if (i < n)
            sycl::atomic_ref<int, sycl::memory_order::relaxed,
                             sycl::memory_scope::work_group>(lhist[classify(src[i])]).fetch_add(1);
        sycl::group_barrier(it.get_group());
        for (int b = lid; b < NBINS; b += it.get_local_range(0))       // one global atomic / bin / group
            if (lhist[b])
                sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device>(hist[b]).fetch_add(lhist[b]);
    });
});
```

## Correctness invariants
- SLM histogram must be **zero-initialized** and barriered before use and before the global flush.
- Use `memory_scope::work_group` for the SLM atomics and `device` for the final global ones.
- The two-level sum must total exactly the single-level result — validate bin counts against the
  baseline.

## Verify it took effect
- Global atomic / L3 atomic traffic drops ~by the work-group size; `SyncStall` from contention falls.
- Reduction-heavy kernels (histograms, scatter-add) speed up dramatically.

## Pitfalls / conflicts
- **Conflicts:** none — but SLM privatization adds a barrier and SLM footprint (watch occupancy).
- **Synergizes:** `reduce-barriers`, `coalesce-memory-access`, `subgroup-reduction`.
- If bins are too many to privatize in SLM, privatize the hottest bins only or tile the bin range.
