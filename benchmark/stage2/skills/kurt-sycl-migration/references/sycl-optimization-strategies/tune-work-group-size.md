# tune-work-group-size

**Class:** Occupancy-bound  ·  **Priority:** Medium  ·  **Impact:** 5–20%

## When to apply
Occupancy < 50% **or** dispatch-queue resource stalls (`THREADGROUP_DISPATCH_QUEUE*` resource stall,
SLM-limited residency). The work-group size is a first-order occupancy knob: too small underfills the
Xe-core; too large caps residency via register/SLM budgets. **256 is the common sweet spot** on Xe2/Xe3.

## Transformation
Make the work-group size a tunable and sweep the sensible candidates (128 / 256 / 512), keeping it a
multiple of the sub-group width (16/32).

```cpp
void run(sycl::queue& q, const float* src, float* dst, int n, int wg /* swept: 128/256/512 */) {
    const int global = ((n + wg - 1) / wg) * wg;      // always round up to whole groups
    q.parallel_for(sycl::nd_range<1>(global, wg), [=](sycl::nd_item<1> it) {
        const int i = it.get_global_id(0);
        if (i < n) dst[i] = /* ... */ src[i];
    });
}
// Try wg ∈ {128, 256, 512}; keep the fastest correct one. Default 256 if untuned.
```

## Correctness invariants
- `global` must round **up** to a whole number of work-groups; keep the tail guard (`i < n`).
- WG size must be a multiple of the sub-group size and within the device max (`max_work_group_size`).
- If the kernel uses SLM, its per-group SLM footprint scales with WG — re-check the SLM budget when
  enlarging.

## Verify it took effect
- Thread occupancy rises toward the residency limit; dispatch-queue resource stalls drop.
- The swept best size beats the default on the target shape.

## Pitfalls / conflicts
- **Conflicts:** `reduce-slm-allocation` and XMX block-size tuning — SLM/tile choices constrain the
  viable WG range; co-tune them.
- **Synergizes:** `reduce-register-pressure`, `adaptive-block-size` (pick size per shape).
- **Anti-pattern reminder:** a hardcoded `WG=256` for *all* inputs regressed up to −522% on tiny dims —
  pair with `adaptive-block-size` when shapes vary widely.
