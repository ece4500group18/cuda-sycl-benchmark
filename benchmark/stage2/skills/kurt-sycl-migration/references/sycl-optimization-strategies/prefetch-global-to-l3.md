# prefetch-global-to-l3

**Class:** Memory-latency  ·  **Priority:** High  ·  **Impact:** 10–30%

## When to apply
`SbidStall` is dominant with **indirect / pointer-chasing / gather** global loads — the address of the
next load depends on data just read (`values[index[i]]`), so the hardware can't prefetch it and each
access stalls on the full DRAM latency. Unlike `prefetch-to-slm`, there is **no predictable tile reuse**
to stage, so SLM staging doesn't apply; a software prefetch into L3 does.

## Transformation
Issue a **look-ahead prefetch** for a future iteration's target so it reaches L3 before the consuming
iteration needs it, using `sycl::ext::oneapi::experimental::prefetch`.

Before — gather stalls on each indirect load:
```cpp
q.parallel_for(sycl::nd_range<1>(global, wg), [=](sycl::nd_item<1> it) {
    const int i = it.get_global_id(0);
    if (i >= n) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k)
        acc += values[index[i * K + k]];        // indirect: unpredictable, full-latency stall
    out[i] = acc;
});
```

After — prefetch the next target into L3 while consuming the current:
```cpp
namespace se = sycl::ext::oneapi::experimental;
constexpr int LOOK = 4;                          // prefetch distance (tune)
q.parallel_for(sycl::nd_range<1>(global, wg), [=](sycl::nd_item<1> it) {
    const int i = it.get_global_id(0);
    if (i >= n) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        if (k + LOOK < K)
            se::prefetch(&values[index[i * K + k + LOOK]], sizeof(float));  // warm L3 ahead of use
        acc += values[index[i * K + k]];
    }
    out[i] = acc;
});
```

## Correctness invariants
- Prefetch is a **hint** — it never changes results and a wrong/late prefetch only wastes bandwidth.
- Guard the look-ahead index (`k + LOOK < K`) so you don't compute an out-of-range address.
- The prefetch distance must be large enough to hide latency but small enough not to evict the data
  before use — tune it (typically 2–8).

## Verify it took effect
- `SbidStall` on the indirect loads drops; L3 hit rate for the gathered data rises.
- Timeline shows the prefetch overlapping the consuming iteration's compute.

## Pitfalls / conflicts
- **Conflicts:** `prefetch-to-slm` / `slm-cache-reuse` when the same working set is already staged
  on-chip — don't prefetch to L3 what you're also caching in SLM.
- **Synergizes:** `increase-ilp`, `software-pipeline` (both give the prefetch time to land).
- Over-prefetching wastes bandwidth and can evict useful lines — measure the distance, don't max it.
