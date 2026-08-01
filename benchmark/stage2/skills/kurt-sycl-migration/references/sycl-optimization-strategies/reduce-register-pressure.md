# reduce-register-pressure

**Class:** Occupancy-bound  ·  **Priority:** Critical  ·  **Impact:** 20–50%

## When to apply
Thread occupancy < 30% (`ComputeBasic`) **and** the IGC asm dump shows **register spills** (spill/fill
to scratch memory). Too many live values per work-item force the compiler to spill, and high per-thread
GRF use caps how many threads are resident per Xe-core.

## Transformation
Shrink the live-value working set: narrow variable scopes, recompute cheap values instead of holding
them, split a monolithic kernel, or **reduce the ILP/unroll factor** that inflated register use. Often
the fix is undoing an over-aggressive earlier optimization.

Illustrative — free registers by lowering an over-high ILP width and scoping temporaries:
```cpp
// Before: ILP=8 keeps 8 loads + 8 partials live → spills, occupancy 22%
constexpr int ILP = 8;
float x[ILP], acc[ILP];   // large live set

// After: ILP=2 fits the GRF budget; occupancy recovers, net faster despite less ILP
constexpr int ILP = 2;
float x0 = src[i0], x1 = src[i1];     // minimal live values
dst[i0] = f(x0); dst[i1] = f(x1);     // temporaries die immediately
```
Other levers: hoist nothing you can cheaply recompute; prefer `switch-grf-mode:large` to *afford* the
pressure only when occupancy is already adequate.

## Correctness invariants
- Recomputing instead of caching must yield identical values.
- Splitting a kernel must preserve any ordering/synchronization the single kernel provided.
- Results unchanged — this is a resource edit, not an algorithm change.

## Verify it took effect
- IGC asm shows **zero spills**; per-thread GRF use drops.
- Thread occupancy (`ComputeBasic`) rises; the kernel speeds up even though each thread does less ILP.

## Pitfalls / conflicts
- **Conflicts:** `increase-ilp`, `vectorize-vec4`, `unroll-hot-loop` — these *raise* pressure; this
  strategy often means dialing one of them back. Re-profile after co-tuning.
- **Synergizes:** `tune-work-group-size`, `switch-grf-mode`.
- Do not cut so far that the kernel becomes latency-bound again — find the occupancy/ILP balance by
  measuring, not by maximizing either extreme.
