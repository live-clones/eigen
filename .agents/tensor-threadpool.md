# Tensor and Thread-Pool Changes

Use this guide for `unsupported/Eigen/Tensor`, `Eigen/ThreadPool`, Core's custom GEMM thread-pool backend, and explicit
thread-pool devices. The repository-root `AGENTS.md` still applies.

## Compatibility and risk

Tensor and ThreadPool are foundational to TensorFlow and other downstream users. "Unsupported" describes Tensor's
API-stability policy, not its importance. Changes to signatures, header layout, evaluation order, allocation,
synchronization, numerical behavior, or performance can have a large downstream impact.

- Prefer additive changes and preserve public header paths. Use `<unsupported/Eigen/Tensor>` and
  `<Eigen/ThreadPool>`; never expose implementation-header includes to users.
- Paths below `unsupported/Eigen/CXX11/` are backward-compatibility forwarding shims only. New code must use the
  canonical `unsupported/Eigen/` headers and must not add new headers under `CXX11/`.
- Preserve `EIGEN_DEVICE_FUNC` on code reachable by CUDA, HIP, or SYCL device evaluation.
- Treat evaluator flags, layouts, scalar/packet/block paths, zero-sized tensors, aliasing, and asynchronous object
  lifetimes as part of the behavior under test.
- Changes to contraction, reduction, convolution, morphing, scheduling, or the cost model are performance-sensitive.
  Add or update a benchmark and compare representative shapes, layouts, thread counts, and scalar types.
- Call out intentional compatibility or performance changes prominently in the merge request.

## Keep the threading mechanisms separate

### OpenMP

OpenMP is Core's primary implicit multithreading mechanism and covers the algorithms listed in
`doc/TopicMultithreading.dox`. It is controlled through the compiler's OpenMP support, `Eigen::setNbThreads`, and the
OpenMP runtime. Do not infer that every algorithm in that list is also supported by the custom GEMM thread pool.

### `EIGEN_GEMM_THREADPOOL`

This macro selects Eigen's custom thread-pool backend for general dense matrix-matrix products only. It is mutually
exclusive with OpenMP. Define it before including Eigen, create an `Eigen::ThreadPool`, and register that pool with
`Eigen::setGemmThreadPool(&pool)` before concurrent GEMM work begins.

The registered pointer is process-global state and the pool remains caller-owned. It must outlive all GEMM using it;
do not replace it while a product is running. `Eigen::setNbThreads` controls the active thread limit, while registering
a pool resets that limit to the pool's thread count. Passing `nullptr` currently queries the registered pool; it does
not clear the registration. Treat `doc/TopicMultithreading.dox` and
`Eigen/src/Core/products/Parallelizer.h` as the current API and implementation references.

### `CoreThreadPoolDevice`

`Eigen::CoreThreadPoolDevice` is an explicit device for parallel Core coefficient-wise assignment:

```cpp
#include <Eigen/ThreadPool>

Eigen::ThreadPool pool(thread_count);
Eigen::CoreThreadPoolDevice device(pool);
destination.device(device) = expression;
```

It is distinct from implicit GEMM parallelization. Changes belong with the device/evaluator tests represented by
`test/assignment_threaded.cpp`, not only the GEMM tests.

### Tensor `ThreadPoolDevice`

Define `EIGEN_USE_THREADS` before `<unsupported/Eigen/Tensor>`, then construct a `ThreadPoolDevice` over an existing
`ThreadPoolInterface` and evaluate explicitly:

```cpp
Eigen::ThreadPool pool(pool_threads);
Eigen::ThreadPoolDevice device(&pool, execution_threads);
output.device(device) = expression;
```

The device does not own the pool. The pool, allocator, input storage, output storage, and callback state must remain
alive until synchronous evaluation returns or asynchronous completion is signaled. Tensor's executor, contraction,
reduction, and device code have `ThreadPoolDevice`-specific paths; a serial `DefaultDevice` test alone is insufficient.
See `unsupported/Eigen/src/Tensor/README.md` and `TensorDeviceThreadPool.h`.

## Evaluator capability flags

`PacketAccess`, `BlockAccess`, `PreferBlockAccess`, and `RawAccess` are independent claims, and the executor combines
them: vectorization follows `PacketAccess` while tiling follows `BlockAccess && PreferBlockAccess`. Declining one does
not decline the other, so a leaf that cannot support tiled evaluation can still be packet-evaluated.

The two paths differ in how they treat evaluator state, and this is the asymmetry to reason about before widening a
flag:

- The coefficient path copies the evaluator per worker range, so a functor mutating evaluator-owned state stays
  thread-local.
- Tiled evaluation captures one evaluator by reference and calls `evalBlock()` from every task concurrently, so the same
  functor instance is shared. A generator with mutable state is a data race on that path even though its `packetOp()`
  is otherwise correct.

`TensorEvaluator` knows its `Device`, so a capability may legitimately depend on it: declining `BlockAccess` only for
`ThreadPoolDevice` keeps serial tiling while removing the race. Because tileability is a compile-time property and the
thread count is not, that decision also applies to a single-threaded pool; prefer the conservative answer.

A lazy block restarts its indices from the block origin. A functor is only safe to serve that way if it is genuinely
index-independent and repeatable. The existence of a nullary `operator()()` does not establish that: a functor may also
provide `operator()(Index)`, which `nullary_wrapper` selects for coefficient evaluation, so a lazily blocked expression
and the same expression evaluated linearly produce different values.

Reduction fast paths carry an equivalent obligation. Splitting an inner-most reduction across independent accumulators
and folding them together permutes the operands rather than only re-associating them, so it requires an associative and
commutative combine with `initialize()` as its identity, not merely a reducer that can combine partial accumulators.
Document which property the trait promises and name the trait after it.

## Cost model

`costPerCoeff()` drives thread-count selection, so it has to describe the path the evaluator actually takes. When a
packet path is conditional, mirror the same condition in the cost: charge the nested evaluator as vectorized only when
the packet path really forwards packets, and as scalar when it gathers or scatters lane by lane. `TensorStriding.h`
passes `vectorized && m_inputStrides[innerDim] == 1` down to the nested evaluator and is the reference for the gather
case. Overstating vectorization here is not a cosmetic error — a nested `exp()` charged as vectorized instead of scalar
has selected one thread where the workload wanted five.

## Scheduling changes

- Preserve the `ThreadPoolInterface` contract, including `Schedule`, `ScheduleWithHint`, `CurrentThreadId`,
  cancellation behavior, and caller ownership.
- Test one-thread and multi-thread execution, work invoked from a worker, completion/wakeup behavior, and shutdown with
  pending or cancelled work when those paths are affected.
- Avoid blocking a worker on work that can only run on the same exhausted pool. Make callback and barrier lifetime
  rules explicit in code when they are not self-evident.
- `DenseBase::Random()` and `setRandom()` use `std::rand` and are not re-entrant. Do not call them concurrently;
  pre-generate inputs or use thread-local `<random>` generators through `NullaryExpr`.
- Cost-model and grain-size changes need both small-workload overhead measurements and large-workload throughput
  measurements. Check oversubscription and nested parallelism rather than assuming more threads are faster.
- Benchmark only on an otherwise idle system, one benchmark process at a time, and report repeated measurements rather
  than a single timing.

## Validation

- Thread-pool internals: run the affected `threads_*` target, especially event-count, run-queue, non-blocking-pool, or
  fork-join tests.
- Custom GEMM pool: run `product_threaded` and the ordinary product tests affected by the change.
- Core explicit device: build and run the assignment-threaded test represented by `test/assignment_threaded.cpp` if
  it is registered in the current test configuration.
- Tensor pool/device changes: run `tensor_thread_pool`, `tensor_executor`, and the focused operation tests such as
  contraction or reduction.
- Tensor behavior shared with accelerators: also follow `simd-gpu.md` and run the locally available device tests.
- Report unavailable sanitizers, GPU toolchains, platforms, and downstream TensorFlow validation explicitly.

Use `test/CMakeLists.txt`, `unsupported/test/CMakeLists.txt`, and the checked-out CMake configuration as the source of
truth for target names. Do not maintain a duplicate test or backend inventory here.
