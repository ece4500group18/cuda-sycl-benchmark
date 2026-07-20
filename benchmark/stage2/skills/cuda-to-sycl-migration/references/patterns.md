# CUDA-to-SYCL repair patterns

## Execution and indexing

- Translate a one-dimensional launch to `parallel_for(nd_range<1>{global,
  local}, ...)`; use `get_global_linear_id()` when CUDA flattens the launch.
- For multidimensional kernels, write down the CUDA formula first and map each
  `blockIdx`, `threadIdx`, `blockDim`, and `gridDim` term explicitly to an
  `nd_item` query. Test dimension order rather than mechanically swapping names.
- Round the global range up only when the kernel retains its original bounds check.

## Memory and dependencies

- `cudaMalloc` -> `malloc_device` or `malloc_shared`; `cudaMemcpy` -> `queue.memcpy`.
- Preserve stream order with an in-order queue or explicit event dependencies.
  Never assume commands on an out-of-order queue execute in submission order.
- Capture USM pointers and scalar values by value in device lambdas. Avoid
  references to host stack objects.
- Check allocation failures and call `wait_and_throw()` before host reads or frees.

## Shared memory and reductions

- Replace static or dynamic `__shared__` storage with a `local_accessor` created
  in the command-group scope and captured by the kernel.
- Replace `__syncthreads()` with a work-group barrier using the correct fence space.
- Preserve active-lane conditions in tree reductions. For non-power-of-two sizes,
  explicitly handle the tail instead of assuming a balanced tree.
- Prefer SYCL group algorithms only when their identity value, associativity,
  numeric order, and result placement match the CUDA code.

## Atomics and warp assumptions

- Construct `atomic_ref<T, order, scope, address_space>` for the actual target
  address space. Use device scope for cross-work-group communication and
  work-group scope only for local memory within one group.
- Replace warp intrinsics with subgroup operations only after checking the
  required subgroup size. Requesting a fixed subgroup is valid only if the Intel
  target reports support; otherwise implement a portable local-memory path.

## Libraries

- Thrust algorithms often map to oneDPL device policies, but a direct SYCL kernel
  is acceptable when it preserves semantics and avoids unavailable libraries.
- cuBLAS/cuFFT/cuRAND/cuSPARSE/cuSOLVER may map to oneMKL domains. Check layout,
  transpose convention, indexing base, scaling, RNG engine/distribution, and
  synchronization. A mathematically similar API call is not automatically an
  output-equivalent migration.
- Do not retain NVIDIA binary formats, CUDA-only headers, or a dependency on an
  NVIDIA device.

## Advanced features

- Re-express CUDA graphs/streams as explicit SYCL event DAGs if portable graph
  extensions are unavailable.
- Replace texture reads with ordinary memory and explicit interpolation/addressing
  logic unless an available SYCL image feature exactly matches the case.
- Dynamic parallelism and cooperative groups usually require algorithmic
  restructuring into multiple queue submissions with explicit dependencies.

## Failure triage

1. Compile error: remove CUDA syntax; inspect template types, address spaces,
   lambda captures, and missing SYCL namespace qualification.
2. Runtime exception: inspect selector, allocation lifetime, work-group limits,
   subgroup assumptions, and missing waits.
3. Wrong output: compare initialization, indexing, barriers, event dependencies,
   reduction order, library layouts, output count, and output precision.
