# multi-output-per-item

**Class:** Launch / dispatch overhead  ·  **Priority:** Medium  ·  **Impact:** 1.2–2×

## When to apply
Tiny kernels where **launch/dispatch overhead is visible** relative to kernel run time — each work-item
produces one output, so the grid is huge and per-item setup (index math, pointer setup, prologue)
dominates the actual work. Having each item produce several outputs amortizes that fixed cost and raises
arithmetic intensity per thread.

## Transformation
Give each work-item a small **strip** of outputs (grid-stride or contiguous block), reusing loaded
operands and setup across them.

Before — one output per item, large grid:
```cpp
q.parallel_for(sycl::nd_range<1>(global, wg), [=](sycl::nd_item<1> it) {
    const int i = it.get_global_id(0);
    if (i < n) dst[i] = a * src[i] + bias;      // 1 output / item
});
```

After — `OUT` outputs per item (smaller grid, shared setup):
```cpp
constexpr int OUT = 4;
const int global = ((n + wg * OUT - 1) / (wg * OUT)) * wg;
q.parallel_for(sycl::nd_range<1>(global, wg), [=](sycl::nd_item<1> it) {
    const int base = it.get_global_id(0) * OUT;
#pragma unroll
    for (int k = 0; k < OUT; ++k) {
        const int i = base + k;
        if (i < n) dst[i] = a * src[i] + bias;   // shared a/bias, amortized launch cost
    }
});
```

## Correctness invariants
- The per-item strip must cover the full range exactly once; keep the tail guard on **every** output
  (`base + k < n`).
- `global` rounds up to whole work-groups over the reduced grid.
- Results unchanged — only the item→output mapping changes.

## Verify it took effect
- Grid size and launch/dispatch overhead drop; per-thread work rises.
- End-to-end time improves for the small-kernel case even though total FLOPs are unchanged.

## Pitfalls / conflicts
- **Conflicts:** `reduce-register-pressure` — more outputs per item means more live values; if occupancy
  falls or spills appear, lower `OUT`.
- **Synergizes:** `increase-ilp` (the multiple outputs are independent in-flight work), `fuse-passes`.
- For a *sequence of small kernels* dominated by submit overhead rather than per-item setup, prefer
  `sycl-graphs`.
