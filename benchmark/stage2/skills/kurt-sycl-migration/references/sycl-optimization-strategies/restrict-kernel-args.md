# restrict-kernel-args

**Class:** Compute-bound / codegen  ·  **Priority:** High  ·  **Impact:** 5–20%

## When to apply
A pointer-heavy kernel where the IGC asm shows **redundant reloads** of values that could stay in
registers, or **conservative (scalar) codegen** where vectorization was expected. By default the
compiler must assume the USM pointer arguments may **alias**, so it reloads after every store and cannot
safely widen/reorder. If you know the buffers are disjoint, telling the compiler unlocks better code.

## Transformation
Assert non-aliasing with `[[intel::kernel_args_restrict]]` on the kernel (covers all pointer args), or
`__restrict__` on individual pointers. Zero code motion — just a promise the compiler can exploit.

Before — assumed aliasing forces reloads:
```cpp
q.parallel_for(sycl::nd_range<1>(global, wg), [=](sycl::nd_item<1> it) {
    const int i = it.get_global_id(0);
    if (i < n) {
        out[i]  = a[i] + b[i];
        out2[i] = a[i] * 2.f;      // compiler may reload a[i]: could out[] alias a[]?
    }
});
```

After — restrict the args (disjoint buffers):
```cpp
q.parallel_for(sycl::nd_range<1>(global, wg),
    [=](sycl::nd_item<1> it) [[intel::kernel_args_restrict]] {   // no arg aliases another
        const int i = it.get_global_id(0);
        if (i < n) {
            float ai = a[i];       // loaded once, kept in a register
            out[i]  = ai + b[i];
            out2[i] = ai * 2.f;
        }
    });
```

## Correctness invariants
- **Only apply when the buffers are genuinely disjoint.** If any two restricted pointers overlap
  (including aliased views of the same USM allocation), the result is **undefined** — this is a
  correctness contract, not a hint.
- In-place kernels where input and output are the same buffer must **not** use it.
- With truly disjoint buffers the results are identical; only codegen changes.

## Verify it took effect
- IGC asm shows fewer reloads and, often, wider/vectorized loads that were previously scalar.
- ALU `PipeStall` / instruction count on the affected path drops.

## Pitfalls / conflicts
- **Conflicts:** none — but it is a footgun if the disjointness assumption is wrong; audit the call
  sites before adding it.
- **Synergizes:** `vectorize-vec4`, `increase-ilp`, `unroll-hot-loop` — restrict often unlocks the
  widening/reordering those depend on.
- Near-free to try; it is one of the first codegen wins in the recommended ordering (step 0).
