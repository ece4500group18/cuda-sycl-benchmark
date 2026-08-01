# native-math

**Class:** Compute-bound  ·  **Priority:** High  ·  **Impact:** ~1.3×

## When to apply
The kernel is **compute-bound** with transcendental calls (`sycl::exp/sin/cos/pow/log/tanh/rsqrt`) in a
hot loop, and the use case **tolerates reduced precision** — activations (GELU/softmax/sigmoid),
embeddings, graphics-style math. The default math functions expand to multi-instruction accurate
sequences; the `native::` variants map to hardware approximations.

## Transformation
Swap the accurate call for its `sycl::native::` counterpart where precision permits.

Before — accurate transcendentals:
```cpp
q.parallel_for(sycl::nd_range<1>(global, wg), [=](sycl::nd_item<1> it) {
    const int i = it.get_global_id(0);
    if (i < n) {
        float x = src[i];
        dst[i] = 1.f / (1.f + sycl::exp(-x));      // accurate exp: many instructions
    }
});
```

After — native approximation on the hot path:
```cpp
q.parallel_for(sycl::nd_range<1>(global, wg), [=](sycl::nd_item<1> it) {
    const int i = it.get_global_id(0);
    if (i < n) {
        float x = src[i];
        dst[i] = 1.f / (1.f + sycl::native::exp(-x));   // hardware approx, fewer instrs
    }
});
```

## Correctness invariants
- `native::*` has **reduced accuracy and narrower valid input range** — validate against the accurate
  baseline with a relaxed tolerance and record it.
- Do **not** use on numerically sensitive paths (iterative solvers, accumulation of many terms,
  anything feeding a reduction where error compounds).
- Guard input domains (e.g. `native::log` of near-zero, `native::rsqrt` of zero) as the reference did.

## Verify it took effect
- `PipeStall`/ALU instruction count on the math drops; the transcendental lowers to a short native
  sequence in the IGC asm.
- Compute-bound kernel time improves; correctness gate passes at the relaxed tolerance.

## Pitfalls / conflicts
- **Conflicts:** none.
- **Synergizes:** `unroll-hot-loop` (fewer, cheaper ops per iteration).
- If accuracy fails the gate, revert to the accurate call — the speedup is never worth a wrong result.
