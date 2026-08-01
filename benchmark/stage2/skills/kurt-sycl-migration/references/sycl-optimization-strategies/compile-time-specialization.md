# compile-time-specialization

**Class:** Compute-bound  ·  **Priority:** High  ·  **Impact:** up to 3× on specialized paths

## When to apply
Dimensions/flags are **known at launch** but the kernel treats them as runtime values, so the hot path
carries dynamic loop bounds and runtime **bounds checks**. Making them compile-time constants unlocks
bounds-check elimination, full unrolling, and constant folding.

## Transformation
Template the kernel on the compile-time dims and dispatch to the matching instantiation with a `switch`
over the common cases, falling back to a generic version for the rest.

Before — runtime dim in the hot loop:
```cpp
void run(sycl::queue& q, const float* src, float* dst, int n, int D) {
    q.parallel_for(sycl::nd_range<1>(g, wg), [=](sycl::nd_item<1> it) {
        const int i = it.get_global_id(0);
        if (i >= n) return;
        float acc = 0.f;
        for (int k = 0; k < D; ++k) acc += src[i * D + k];   // D runtime: bounds check, no unroll
        dst[i] = acc;
    });
}
```

After — templated on `D`, switch-dispatched:
```cpp
template <int D>
void run_impl(sycl::queue& q, const float* src, float* dst, int n) {
    q.parallel_for(sycl::nd_range<1>(g, wg), [=](sycl::nd_item<1> it) {
        const int i = it.get_global_id(0);
        if (i >= n) return;
        float acc = 0.f;
#pragma unroll
        for (int k = 0; k < D; ++k) acc += src[i * D + k];   // D constant: fully unrolled
        dst[i] = acc;
    });
}
void run(sycl::queue& q, const float* src, float* dst, int n, int D) {
    switch (D) {
        case 64:  run_impl<64>(q, src, dst, n);  break;
        case 128: run_impl<128>(q, src, dst, n); break;
        case 256: run_impl<256>(q, src, dst, n); break;
        default:  /* generic runtime-D fallback */ ;
    }
}
```

## Correctness invariants
- The `default` fallback must handle any unlisted dimension correctly — do not silently skip it.
- Specialized and generic paths must produce identical results; validate at least one specialized case
  and the fallback.
- Keep the specialization set small (the common shapes) to bound compile time and binary size.

## Verify it took effect
- IGC asm for the specialized path shows the loop fully unrolled with the bounds check gone.
- Compute-bound time drops on the specialized shapes; no regression on the fallback.

## Pitfalls / conflicts
- **Conflicts:** none.
- **Synergizes:** `unroll-hot-loop` (compile-time bounds make unrolling free), `fast-integer-division`
  (constant divisors fold to shifts).
- Over-specializing (too many template instances) bloats the binary and compile time — specialize only
  hot, frequent shapes.
