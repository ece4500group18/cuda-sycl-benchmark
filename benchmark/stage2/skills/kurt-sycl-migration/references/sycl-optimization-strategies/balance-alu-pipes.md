# balance-alu-pipes

**Class:** Compute-bound  ·  **Priority:** Medium  ·  **Impact:** 10–25%

## When to apply
The kernel is ALU-bound and one **ALU-pipe utilization** metric is far higher than the others — e.g. the
float pipe is saturated while the integer/extended-math pipes are idle. Xe cores have multiple ALU pipes
(3 on BMG / 4 on CRI) that can co-issue; a workload leaning on one pipe leaves the rest stalled.

## Transformation
Redistribute work across pipes: move independent ops to the underused pipe, hoist address/integer math
off the float pipe, or convert a float-pipe operation to an equivalent on an idle pipe. The exact edit
is workload-specific; the goal is to flatten the per-pipe utilization.

Illustrative — offload index math and use a fused op to relieve the float pipe:
```cpp
// Before: heavy float pipe (mul+add chain) + repeated float index scaling in the same pipe
float acc = 0.f;
for (int k = 0; k < K; ++k)
    acc += a[i * K + k] * b[k] * scale;          // scale mul on the float pipe every iter

// After: hoist the invariant, use fma so mul+add co-issue efficiently, keep index math integer
float acc = 0.f;
for (int k = 0; k < K; ++k)
    acc = sycl::fma(a[i * K + k], b[k], acc);    // one fused op / iter
acc *= scale;                                     // scale once, off the hot loop
```

## Correctness invariants
- `sycl::fma(x, y, z)` computes `x*y+z` with a single rounding — validate if the extra precision vs
  separate mul/add matters for your tolerance.
- Hoisting an invariant out of the loop must not change which values it multiplies.
- Any op moved to a different pipe must be numerically equivalent.

## Verify it took effect
- Re-profiled per-pipe utilization is **flatter** (the previously-idle pipe now carries work); overall
  ALU throughput rises.
- Compute-bound kernel time improves without a bandwidth or occupancy change.

## Pitfalls / conflicts
- **Conflicts:** `offload-xmx-library` on GEMM loops — if the loop should be on XMX, balancing its XVE
  pipes is the wrong fix; offload instead.
- **Synergizes:** `increase-ilp` (more independent ops to spread), `eliminate-branches` (frees the
  control path).
- This is a fine-tuning strategy — apply after the dominant stall and occupancy limiters are resolved.
