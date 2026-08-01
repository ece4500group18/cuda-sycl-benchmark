# CUDA-to-SYCL Precision Migration Skill

You are migrating a CUDA kernel to SYCL. The #1 cause of verification failure is **floating-point precision mismatch**, not API errors. Follow these rules IN ORDER before writing any code.

---

## Rule 1: Prevent Compiler Reassociation (CRITICAL)

`icpx -O2` inlines host helper functions and reassociates float expressions (e.g. `60*(2*h-1)` becomes `120*h-60`), which flips rounding at `.5` boundaries and breaks `lrintf` quantization.

**Fix:** Mark ALL host-side helper functions with `__attribute__((noinline))`:

```cpp
// WRONG — compiler will inline and reassociate
static inline float hash01(unsigned i) { ... }

// CORRECT — forces float32 rounding at call boundary
__attribute__((noinline)) static float hash01(unsigned i) { ... }
```

This applies to ANY function whose return value feeds into `lrintf`, `rintf`, or quantization math.

---

## Rule 2: Rounding — Use rint, Never round

CUDA `lrintf` / `__float2int_rn` rounds to nearest-even. SYCL `sycl::round` rounds half-away-from-zero (different!).

```cpp
// WRONG — produces 3.8e-02 max error in quantization
int8_t q = (int8_t)sycl::round(x * scale);

// CORRECT — matches CUDA lrintf semantics
int8_t q = (int8_t)sycl::rint(x * scale);
// or on host: (int8_t)lrintf(x * scale);
```

---

## Rule 3: Power Function — Use exp/log, Never pow

`sycl::pow(base, x)` has different fp32 precision than CUDA `powf`. Always rewrite:

```cpp
// WRONG — precision diverges from CUDA
float theta = pos * sycl::pow(10000.0f, -2.0f * i / dim);

// CORRECT — bit-matches CUDA powf
float theta = pos * sycl::exp(-2.0f * i / dim * sycl::log(10000.0f));
```

General form: `pow(b, x)` → `exp(x * log(b))`

---

## Rule 4: Build Flags (if you control compilation)

If you can modify build configuration, add `-fp-model=precise` to prevent x87 excess-precision and unsafe reassociation globally. In this sandbox, edit `remote_config.json` and append `"-fp-model=precise"` to the `"extra_sycl_flags"` array.

If you cannot modify build flags, Rule 1 (noinline) is your code-level equivalent.

---

## Rule 5: Always Use In-Order Queue

SYCL 2020 queues are **out-of-order by default**. USM memcpy and kernel launches have NO implicit ordering. Always create an in-order queue:

```cpp
// WRONG — data race: kernel may read before memcpy completes
sycl::queue q(sycl::gpu_selector_v);

// CORRECT — guarantees sequential execution
sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
```

---

## Rule 6: SYCL API Quick Reference

| CUDA | SYCL |
|------|------|
| `threadIdx.x + blockIdx.x * blockDim.x` | `item.get_global_id(0)` |
| `blockDim.x` | `item.get_local_range(0)` |
| `__shared__ float smem[N]` | `sycl::local_accessor<float, 1> smem(sycl::range<1>(N), item)` |
| `__syncthreads()` | `sycl::group_barrier(item.get_group())` |
| `atomicAdd(&addr, val)` | `sycl::atomic_ref<int, ...>(addr).fetch_add(val)` |
| `cudaMalloc` / `cudaFree` | `sycl::malloc_device<T>(n, q)` / `sycl::free(ptr, q)` |
| `cudaMemcpy` | `q.memcpy(dst, src, bytes)` |
| `__expf(x)` | `sycl::exp(x)` |
| `__logf(x)` | `sycl::log(x)` |
| `rsqrtf(x)` | `sycl::rsqrt(x)` |
| `fabsf(x)` | `sycl::fabs(x)` |

---

## Rule 7: Reductions

Use `sycl::reduce_over_group` for block reductions. Do NOT hand-write tree reductions unless the kernel requires custom logic:

```cpp
float sum = sycl::reduce_over_group(item.get_group(), val, sycl::plus<float>());
```

---

## Workflow

1. Read `main.cu` and identify: helper functions, rounding calls, pow calls, reductions
2. Apply Rules 1-3 to your translation BEFORE compiling
3. Write `main.sycl.cpp`
4. Build and run; if verification fails with small numerical error (< 0.1), re-check Rules 1-3
5. Specifically: if error is ~1e-5 to 1e-2, it is almost certainly Rule 1 (noinline) or Rule 2 (rint)
