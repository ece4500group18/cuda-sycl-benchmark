# offload-xmx-library

**Class:** Compute-bound  ·  **Priority:** Critical  ·  **Impact:** 5–10× over XVE

## When to apply
The kernel has a **GEMM or convolution** shape **and** the IGC asm shows **no DPAS** instructions
(`XMX idle`) — the matmul is running on the XVE vector ALUs instead of the matrix engine. This is the
single highest-impact fix for dense linear algebra.

## Transformation
**Route the operation to a maintained library** — **oneDNN**, **sycl-tla**, or **oneMKL** — instead of
hand-writing an XMX kernel. This framework does **not** hand-roll `joint_matrix`.

Before — hand-written triple loop on the XVE:
```cpp
q.parallel_for(sycl::nd_range<2>({M, N}, {16, 16}), [=](sycl::nd_item<2> it) {
    const int r = it.get_global_id(0), c = it.get_global_id(1);
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += A[r * K + k] * B[k * N + c];  // no DPAS, XMX idle
    C[r * N + c] = acc;
});
```

After — oneDNN matmul primitive (uses XMX/DPAS):
```cpp
// oneDNN: describe memory + a matmul primitive on a SYCL-backed engine/stream
dnnl::engine eng(dnnl::engine::kind::gpu, 0);
dnnl::stream strm(eng);
auto md_a = dnnl::memory::desc({M, K}, dt::f16, tag::ab);
auto md_b = dnnl::memory::desc({K, N}, dt::f16, tag::ab);
auto md_c = dnnl::memory::desc({M, N}, dt::f32, tag::ab);
dnnl::matmul::primitive_desc pd(eng, md_a, md_b, md_c);
dnnl::matmul(pd).execute(strm, {{DNNL_ARG_SRC, mem_a},
                                {DNNL_ARG_WEIGHTS, mem_b},
                                {DNNL_ARG_DST, mem_c}});
strm.wait();
```
For a fused/custom tile shape a library cannot express, use **sycl-tla** (CUTLASS-style, DPAS-tiled) —
see `sycl-tla-patterns.md` — **not** a hand-written `joint_matrix` loop.

## Correctness invariants
- Match the library's expected **layouts and data types**; convert once at the boundary, not per-tile.
- Keep the **accumulator in FP32** even with FP16/BF16 inputs.
- Validate against the reference GEMM within the type's tolerance.

## Verify it took effect
- IGC asm now contains DPAS instructions; XMX utilization counters rise from zero.
- Throughput approaches the roofline compute peak for the shape.

## Pitfalls / conflicts
- **Conflicts:** `balance-alu-pipes` on the *same* loop — once on XMX, ALU-pipe tuning of that loop is
  moot.
- **Synergizes:** `reduce-data-type-width` (feed FP16/BF16 to DPAS).
- **Anti-pattern reminder:** hand-rolled `joint_matrix` where a library fits is brittle and reinvents
  vendor tuning — only reach for it when no library expresses the shape, then follow the K-loop
  double-buffering rules in `sycl-tla-patterns.md`.
