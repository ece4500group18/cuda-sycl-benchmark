# increase-ilp

**Class:** Memory-latency  ·  **Priority:** Critical  ·  **Impact:** 15–40%

## When to apply
`SbidStall` is the dominant entry in the `VectorEngineStalls` mix (scoreboard waits on outstanding
async/memory results), **or** `SendStall` > 40% *and* the IGC asm dump shows spare registers (no
spills). The kernel issues a load, then immediately consumes it, so each XVE thread stalls on its own
dependency chain instead of overlapping independent work.

## Transformation
Give each work-item several **independent** accumulators / in-flight loads so the scoreboard has other
work to cover the latency. Process a small strip per item instead of one element.

Before — one dependent chain per item:
```cpp
q.parallel_for(sycl::nd_range<1>(global, wg), [=](sycl::nd_item<1> it) {
    const int i = it.get_global_id(0);
    if (i < n) {
        float x = src[i];              // load
        dst[i] = f(x);                 // consume immediately → SbidStall
    }
});
```

After — `ILP` independent lanes per item (strip-mined, unrolled):
```cpp
constexpr int ILP = 4;                                  // independent chains per item
const int global = ((n + wg * ILP - 1) / (wg * ILP)) * wg;
q.parallel_for(sycl::nd_range<1>(global, wg), [=](sycl::nd_item<1> it) {
    const int base = it.get_global_id(0) * ILP;
    float x[ILP];
#pragma unroll
    for (int k = 0; k < ILP; ++k)                        // issue all loads first...
        if (base + k < n) x[k] = src[base + k];
#pragma unroll
    for (int k = 0; k < ILP; ++k)                        // ...then consume: loads overlap
        if (base + k < n) dst[base + k] = f(x[k]);
});
```

## Correctness invariants
- Keep the **tail guard** on every indexed access (`base + k < n`); `global` is rounded up.
- The `ILP` lanes must be **truly independent** — no lane reading another lane's output.
- Register cost grows ~linearly with `ILP`; do not exceed the point where spills appear (see conflicts).

## Verify it took effect
- `SbidStall` share drops in the re-profiled `VectorEngineStalls` mix; active-thread latency hiding
  improves.
- IGC asm shows the loads hoisted **above** their consumers (batched `send`s before the dependent ALU),
  not interleaved one-load-one-use.

## Pitfalls / conflicts
- **Conflicts:** `reduce-register-pressure`, `switch-grf-mode:small` — more in-flight values need more
  GRF; if occupancy < 30% or spills appear, `ILP` is too high. Re-profile occupancy after raising it.
- **Synergizes:** `software-pipeline`, `vectorize-vec4` — stack only after each is an individual win.
- Do not raise `ILP` blindly past 4–8; measure — the gain saturates and then reverses via spills.
