# Plan: Replace Eigen's RNG with thread-safe PCG

## Context

Eigen's `Random()` / `setRandom()` / `internal::random<Scalar>()` all bottom out at `std::rand()`, which has two major problems:
1. **Not thread-safe** — uses a single global state, causing data races in multi-threaded code. Documented with `\not_reentrant` warnings everywhere.
2. **Poor quality** — `std::rand()` is typically a 32-bit LCG with ~15-31 bits of entropy per call. Multiple calls are needed to fill 32/64-bit types.

The Tensor module (`TensorRandom.h`) already has a PCG-XSH-RS implementation with per-instance state, proving the approach works in Eigen's codebase. The core `RandomImpl.h` even has a `TODO: replace or provide alternatives to this`.

## Approach

Replace `eigen_random_device` (the `std::rand()` wrapper) with PCG-XSH-RS using `thread_local` state on host. This is the minimal change — all the distribution machinery (`random_bits_impl`, `random_float_impl`, `random_int_impl`, etc.) stays unchanged. The `scalar_random_op` functor remains stateless. On GPU, keep `std::rand()` as fallback (GPU users should use TensorRandom or cuRAND).

## Changes

### 1. `Eigen/src/Core/RandomImpl.h` — Replace engine

Before the existing `eigen_random_device` struct (line 45):

- Add `pcg_xsh_rs_step(uint64_t* state, uint64_t increment) -> unsigned` — the shared PCG step function (also usable by TensorRandom)
- Add `EIGEN_HAS_THREAD_LOCAL_RANDOM` macro, reusing the existing `EIGEN_AVOID_THREAD_LOCAL` pattern from `Memory.h:64-75`
- Add `eigen_tl_random_state()` returning a `thread_local` struct with `{state, inc}`, default-seeded
- Add `set_random_seed(uint64_t seed)` to re-seed the calling thread's state

Replace `eigen_random_device`:
- `ReturnType` changes from `int` to `unsigned`
- `Entropy` changes from `meta_floor_log2<RAND_MAX+1>` (~15-31) to `32`
- `Highest` changes from `RAND_MAX` to `0xFFFFFFFFu`
- `run()` calls `pcg_xsh_rs_step` on the thread-local state (host) or falls back to `std::rand()` (GPU / no `thread_local`)

Impact on `random_bits_impl`: the loop at line 65 runs fewer iterations (32 bits/call vs ~15). The `static_cast<Scalar>(RandomDevice::run())` at line 66 changes from `int`→unsigned to `unsigned`→unsigned (no functional change since `Scalar` is always unsigned here).

### 2. `Eigen/src/Core/Random.h` — Public seeding API + doc updates

- Add `Eigen::setRandomSeed(uint64_t seed)` — public wrapper around `internal::set_random_seed()`
- Remove `\not_reentrant` from all doc comments on `Random()` and `setRandom()`
- Add note: "Each thread has independent random state. Use `Eigen::setRandomSeed()` to control seeding per-thread."

### 3. `test/main.h` — Seed new engine

At line 912, after `srand(g_seed)`, add:
```cpp
Eigen::internal::set_random_seed(static_cast<uint64_t>(g_seed));
```

### 4. `test/rand.cpp` — New tests

- Add reproducibility test: seed, generate, re-seed with same value, verify identical sequence
- Add thread-safety test (guarded by `EIGEN_HAS_THREAD_LOCAL_RANDOM`): spawn threads each generating random values, verify no crashes and different threads produce different sequences
- Existing histogram tests validate distribution quality — they should pass unchanged since PCG has good uniformity

### 5. `unsupported/Eigen/src/Tensor/TensorRandom.h` — Unify PCG, fix UB

- Replace inline `PCG_XSH_RS_generator()` body with call to shared `pcg_xsh_rs_step()` from `RandomImpl.h`
- Fix union-based type punning in `RandomToTypeUniform<float>` and `<double>` — replace with `numext::bit_cast`

### 6. `doc/TopicMultithreading.dox` — Update docs

Replace the "not re-entrant" warning with documentation that `Random()`/`setRandom()` are now thread-safe via per-thread state.

## Verification

1. Build all tests: `ninja -C /tmp/eigen_build -j8 buildtests`
2. Run `test/rand` (all 15 subtests) — validates range, histogram uniformity, all scalar types
3. Run new thread-safety and reproducibility subtests
4. Run `test/basicstuff`, `test/product_extra`, `test/diagonalmatrices` — they use `setRandom()` extensively
5. Run `clang-format-17` on modified files
