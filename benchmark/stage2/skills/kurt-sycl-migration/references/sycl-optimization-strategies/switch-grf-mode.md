# switch-grf-mode

**Class:** Occupancy-bound  ·  **Priority:** High  ·  **Impact:** 10–30%

## When to apply
Occupancy < 50% with **high register pressure**. Intel GPUs expose a GRF (general register file) mode
trade-off: **large-GRF** gives each thread more registers (fewer spills, fewer resident threads);
**small-GRF** gives more resident threads (higher occupancy, tighter register budget). Pick the side
that matches the kernel's true limiter.

## Transformation
Set the GRF mode via a kernel attribute / build flag rather than rewriting code:

- Register-heavy kernel that **spills** in default mode → **large GRF** (afford the pressure):
```cpp
q.parallel_for(
    sycl::nd_range<1>(global, wg),
    [=](sycl::nd_item<1> it) [[intel::grf_size(256)]] {   // large GRF: 256 regs/thread
        /* register-hungry body, now spill-free */
    });
```
- Occupancy-starved kernel with **modest** register need → **small GRF** (more threads):
```cpp
q.parallel_for(
    sycl::nd_range<1>(global, wg),
    [=](sycl::nd_item<1> it) [[intel::grf_size(128)]] {   // small GRF: more resident threads
        /* light body that benefits from higher occupancy */
    });
```
(Equivalent to the `-ze-opt-large-register-file` / auto-GRF compiler controls; prefer the attribute so
it is scoped to the kernel.)

## Correctness invariants
- GRF mode is a **performance** knob only — results are identical in either mode.
- Confirm the toolchain honors the attribute for the target device; otherwise use the build flag.

## Verify it took effect
- **Large GRF:** spills disappear in the IGC asm; per-thread registers rise.
- **Small GRF:** thread occupancy rises with no new spills.
- The change is only a win if the *limiter* moved — re-profile occupancy vs spills.

## Pitfalls / conflicts
- **Conflicts:** `increase-ilp` and `vectorize-vec4` in **small** mode — those need registers small mode
  removes; combining them can spill.
- **Synergizes:** `reduce-register-pressure` (reduce demand *and* pick large GRF), `tune-work-group-size`
  (re-tune WG after occupancy changes).
- This is an **enabling transform**: small-GRF may regress alone until you re-tune the work-group size to
  use the extra residency.
