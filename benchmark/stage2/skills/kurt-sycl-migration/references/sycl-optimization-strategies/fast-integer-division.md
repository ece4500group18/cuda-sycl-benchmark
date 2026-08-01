# fast-integer-division

**Class:** Compute-bound  ·  **Priority:** Medium  ·  **Impact:** 5–20%

## When to apply
The hot path does **int64 division or modulo by a non-power-of-2 divisor** in index math — flattening
multi-dimensional indices, computing row/col from a linear id, or ring-buffer wraps. Integer
division is a long, non-pipelined operation on Xe; a runtime-constant divisor can be replaced by a
multiply-shift.

## Transformation
Precompute a **magic-number reciprocal** on the host and use multiply-high + shift in the kernel, or —
simplest and often enough — **narrow to 32-bit** and hoist the divisor so the compiler emits the fast
sequence. For power-of-2 divisors, use shift/mask directly.

Before — int64 div/mod per element:
```cpp
q.parallel_for(sycl::nd_range<1>(global, wg), [=](sycl::nd_item<1> it) {
    const long i = it.get_global_id(0);
    if (i >= n) return;
    long row = i / D;                 // 64-bit div in the hot path
    long col = i % D;                 // 64-bit mod
    dst[i] = src[row * D + col];
});
```

After — 32-bit indices + a single div reused for mod:
```cpp
q.parallel_for(sycl::nd_range<1>(global, wg), [=](sycl::nd_item<1> it) {
    const int i = (int)it.get_global_id(0);       // fits: dims validated < 2^31
    if (i >= n) return;
    const int row = i / D;                         // one 32-bit div...
    const int col = i - row * D;                   // ...mod via mul-sub (no second div)
    dst[i] = src[row * D + col];
});
// If D is a runtime constant reused across launches, precompute a magic multiplier
// (libdivide-style) and replace `i / D` with mul-high + shift.
```

## Correctness invariants
- Narrowing to `int` is only valid when **all indices fit in 32 bits** — verify `n`, `D`, and their
  products against `INT_MAX` at the boundary; otherwise keep 64-bit or use the magic-number path.
- `col = i - row * D` must use the same `row` (truncating division) — exact for non-negative indices.
- Magic-number constants must be derived for the exact divisor; validate the full index range.

## Verify it took effect
- IGC asm shows the division replaced by multiply/shift (or a 32-bit div) instead of the int64 divide
  sequence; ALU `PipeStall` on index math drops.

## Pitfalls / conflicts
- **Conflicts:** none.
- **Synergizes:** `balance-alu-pipes` (moves work off the divide unit), `compile-time-specialization`
  (a constant `D` folds the division automatically — prefer that when the dim is known).
- **Anti-pattern reminder:** `size_t`/64-bit stride arithmetic where 32-bit suffices regressed −35% —
  narrow when the dims fit.
