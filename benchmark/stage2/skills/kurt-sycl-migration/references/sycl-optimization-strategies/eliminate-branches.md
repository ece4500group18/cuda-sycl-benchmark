# eliminate-branches

**Class:** Control-flow  ·  **Priority:** High  ·  **Impact:** 10–25%

## When to apply
`ControlStall` is high, or the control-instruction ratio (control ops / issued ops) > 0.15. Branches in
the hot path cost control-flow instructions and, when lanes disagree, force the SIMD engine to execute
both sides. Replacing data-dependent branches with **branchless select/arithmetic** removes the control
overhead.

## Transformation
Convert small `if/else` value choices into `sycl::select` / arithmetic masks so all lanes execute one
straight-line path.

Before — per-lane branch (ReLU + clamp):
```cpp
q.parallel_for(sycl::nd_range<1>(g, wg), [=](sycl::nd_item<1> it) {
    const int i = it.get_global_id(0);
    if (i >= n) return;
    float x = src[i];
    float y;
    if (x < 0.f) y = 0.f;                 // divergent branch
    else if (x > 6.f) y = 6.f;            // another branch
    else y = x;
    dst[i] = y;
});
```

After — branchless clamp:
```cpp
q.parallel_for(sycl::nd_range<1>(g, wg), [=](sycl::nd_item<1> it) {
    const int i = it.get_global_id(0);
    if (i >= n) return;
    float x = src[i];
    dst[i] = sycl::clamp(x, 0.f, 6.f);    // straight-line, no control divergence
});
```

## Correctness invariants
- The branchless form must be **semantically identical** for every input, including boundary values
  (`clamp` is inclusive at both ends).
- Only convert branches that are cheap on both sides — do **not** make lanes execute genuinely
  expensive work they would have skipped (that regresses).
- Keep the coarse tail guard (`i >= n`) as a branch; it's uniform, not divergent.

## Verify it took effect
- `ControlStall` share and control-instruction ratio drop; the branch disappears from the IGC asm in
  favor of select/min/max.
- SIMD lane utilization improves on previously divergent inputs.

## Pitfalls / conflicts
- **Conflicts:** none — but avoid flattening a branch that guards a large/expensive block; measure.
- **Synergizes:** `uniform-control-flow`, `balance-alu-pipes` (selects move work to the ALU pipes).
- Predicating heavy work is the anti-goal; predicate only cheap value choices.
