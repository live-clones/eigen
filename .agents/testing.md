# Testing Eigen Changes

Use this guide when adding or changing tests. The checked-out source is authoritative:

- [`test/main.h`](../test/main.h) configures and runs the test framework and aggregates the shared helpers.
- [`test/numerical_test_helpers.h`](../test/numerical_test_helpers.h) defines numerical comparison, assertion, and
  tolerance helpers.
- [`test/product_test_helpers.h`](../test/product_test_helpers.h) defines matrix-product error bounds.
- [`test/random_matrix_helper.h`](../test/random_matrix_helper.h) and
  [`test/type_test_helpers.h`](../test/type_test_helpers.h) define random-matrix and type utilities.
- [`cmake/EigenTesting.cmake`](../cmake/EigenTesting.cmake) defines test registration and splitting.
- [`test/CMakeLists.txt`](../test/CMakeLists.txt) and
  [`unsupported/test/CMakeLists.txt`](../unsupported/test/CMakeLists.txt) register the suites.
- [`cmake/EigenConfigureTesting.cmake`](../cmake/EigenConfigureTesting.cmake) defines aggregate build and check
  targets.

## Configure And Build

Configure a dedicated build directory. Unit tests are excluded from CMake's default `all` target, although a bare
build may still build enabled auxiliary libraries.

```bash
cmake -G Ninja -S . -B build
cmake --build build --target buildtests
ctest --test-dir build --parallel --output-on-failure
```

Useful aggregate targets are `BuildOfficial`, `BuildUnsupported`, `buildsmoketests`, `buildtests_gpu`, `check`, and
`check_gpu`. Build and run one test explicitly when possible:

```bash
cmake --build build --target bdcsvd_3
ctest --test-dir build -R '^bdcsvd_3$' --output-on-failure
```

Run the generated wrappers from the build directory because they invoke the configured build tool relative to their
working directory:

```bash
cd build
./buildtests.sh <regex>
./check.sh <regex>
```

They filter registered parent names such as `bdcsvd`, not generated part names such as `bdcsvd_3`; use the explicit
target recipe for one part.

Use a separate build directory for each materially different configuration. Do not rewrite one cache and describe
the result as a second test run.

```bash
cmake -G Ninja -S . -B build-row-major -DEIGEN_DEFAULT_TO_ROW_MAJOR=ON
cmake -G Ninja -S . -B build-no-vector -DEIGEN_TEST_NO_EXPLICIT_VECTORIZATION=ON
```

Consult the top-level [`CMakeLists.txt`](../CMakeLists.txt) and nearby test CMake files for current options instead of
copying an option inventory into documentation.

## Current Test Framework

Eigen currently uses its own framework, not GoogleTest:

1. Add `test/<name>.cpp` or `unsupported/test/<name>.cpp`.
2. Include `main.h`, then the public umbrella header for tests of public behavior. A focused test of a private utility
   may include its implementation header only when that matches an established nearby pattern; never present such a
   path as a user include.
3. Use `VERIFY`, `VERIFY_IS_EQUAL`, `VERIFY_IS_APPROX`, and the other helpers exposed through `test/main.h`.
4. End with `EIGEN_DECLARE_TEST(<name>) { ... }`.
5. Register the source with `ei_add_test(<name>)` in the matching `CMakeLists.txt`, then reconfigure.

Keep `test/main.h` limited to framework configuration, registration, shared-helper aggregation, and the test driver.
Put reusable utilities in a narrowly named helper header; include it from `main.h` only when most tests need it.

For compile-failure coverage, use the established `failtest/` pattern. Its `_ok` target must compile and its `_ko`
target must fail with `EIGEN_SHOULD_FAIL_TO_BUILD` defined.

## Split Tests

`ei_add_test` scans the source for `CALL_SUBTEST_N`, `EIGEN_TEST_PART_N`, and `EIGEN_SUFFIXES;...` markers.

- With `EIGEN_SPLIT_LARGE_TESTS=ON`, every discovered suffix becomes an executable `<name>_<N>` compiled with
  `EIGEN_TEST_PART_<N>=1`; the parent `<name>` target builds all parts.
- `EIGEN_SUFFIXES;...` supplies an explicit suffix list when ordinary source scanning cannot see macro-generated or
  conditional parts.
- With splitting off, tests containing only `CALL_SUBTEST_N` or `EIGEN_SUFFIXES` fold into one `<name>` executable
  compiled with `EIGEN_TEST_PART_ALL=1`.
- An explicit `EIGEN_TEST_PART_N` marker forces splitting even when the option is off. If any such marker is present,
  all suffixes discovered in that source are emitted.

`ctest -R '^<name>$'` does not match split parts. Use `ctest -R '<name>'` for every part or anchor one generated name.

After changing subtest registration, reconfigure and read back the generated target list rather than assuming it. Two
failure modes are silent: a subtest function that no longer has a `CALL_SUBTEST` call still compiles and still looks
like coverage, and a part reached only through a macro such as `CALL_SUBTESTS_TYPES_LAYOUTS` is not built at all under
`EIGEN_SPLIT_LARGE_TESTS=ON` unless an `EIGEN_SUFFIXES` marker lists it. Both have gone unnoticed for years in this
repository. Confirm that every subtest function in the source is called, and that the configured parts match the
suffixes the source intends.

## Coverage That Can Fail

A test that passes when the change is reverted is not coverage. Establish that it fails at the parent commit, and when
that is impractical, establish by construction that it reaches the new code.

- A capability flag needs both halves: a `STATIC_CHECK` that the flag has the expected value for the expression under
  test, and an evaluation through the ordinary public path that consults it. Calling the new method directly through a
  helper proves the method works, not that anything selects it — such a test keeps passing when the flag reverts to
  `false`. Compose an expression that actually prefers the path, evaluate it through assignment on the relevant devices
  and layouts, and compare against the coefficient-wise result.
- Pin both ends of an opt-in trait with `STATIC_CHECK`: a type that must be in, and a type that must stay out.
- Do not verify a subset of the result. When a test functor writes only some coefficients, a raw or block consumer still
  copies the whole buffer, so skipping the rest in verification means corruption there cannot fail the test.
  Zero-initialize the destination and check every coefficient.
- Exercise the customization points a user is documented to have, not only the built-in specializations: a functor that
  declares no traits, one that declares them partially, and one with an extra overload. Several correctness regressions
  in this repository were invisible because every in-tree specialization happened to satisfy the new precondition.

## Configurations The Test Suite Cannot See

Enumerate the configurations and instantiations that compile the new code, then check the ones the default build omits.

- `test/main.h` undefines `NDEBUG`, and `Eigen/src/Core/util/Macros.h` derives `EIGEN_NO_DEBUG` from it, so no test in
  the suite ever compiles an `EIGEN_NO_DEBUG` code path. If the change makes a buffer size, member, or branch depend on
  that macro, verify it with a standalone `-DNDEBUG` program and say so.
- The converse also holds: the body of an `eigen_assert` is only type-checked where assertions are enabled. An
  assertion that calls a member the argument type does not have compiles cleanly in every `NDEBUG` build, including the
  benchmarks, and breaks ordinary user builds.
- Instantiate the public argument types, not only the internal ones a nearby test happened to use. A test written
  against an internal dimension type does not cover the type users pass.
- Run an `EIGEN_DEFAULT_TO_ROW_MAJOR` build when layout is in play, and pin the layout explicitly where a test aliases
  one object's storage through another view whose default layout is fixed.
- Cover `EIGEN_TEST_NO_EXPLICIT_VECTORIZATION`, `EIGEN_UNALIGNED_VECTORIZE=0`, and a narrower
  `EIGEN_DEFAULT_DENSE_INDEX_TYPE` when the change reasons about packets, alignment, or index width.

## Numerical Assertions

`VERIFY_IS_APPROX` is a convenient broad comparison, not a machine-epsilon guarantee. `test_precision<T>()` uses
`NumTraits<T>::dummy_precision()` generically and currently specializes float to `1e-3` and double/long double to
`1e-6`. Do not use it alone to claim ULP accuracy, backward stability, or IEEE special-value conformance.

For numerical kernels, add explicit named bounds based on epsilon, dimension, conditioning, or a backward-error
model as appropriate. Check NaN, infinity, and signed zero explicitly when their distinction matters. Follow
[`numerics.md`](numerics.md) for solver, packet, and scalar-math coverage.

Two ways a comparison silently accepts everything, both of which have shipped here:

- The tolerance is computed by the operation under test. A product error bound formed as `(A.cwiseAbs() * B.cwiseAbs())`
  routes the tolerance through the code paths the test is supposed to be checking. Accumulate such a bound explicitly,
  in a wider type, outside the implementation being tested.
- The comparison admits a non-finite value. `error <= tolerance` is true when both are infinite, and
  `if (error > bound)` never fires for a NaN error. Assert the negation, and reject a non-finite tolerance where it is
  computed so a later caller cannot reintroduce the hole.

Run reproducible failures directly with a fixed seed and repeat count:

```bash
EIGEN_REPEAT=10 EIGEN_SEED=1 build/test/foo_3
build/test/foo_3 r10 s1
```

## External BLAS And Shim Libraries

`EIGEN_TEST_EXTERNAL_BLAS=ON` finds a system BLAS, defines `EIGEN_USE_BLAS`, and links that BLAS into applicable
official tests. With it off, ordinary tests exercise Eigen's normal implementation; they do not transparently use
the in-tree `eigen_blas` library. `EIGEN_BUILD_BLAS` and `EIGEN_BUILD_LAPACK` separately build Eigen's ABI shim
libraries, which are also used to satisfy some optional sparse-backend links. There is currently no
`EIGEN_TEST_EXTERNAL_LAPACK` option.

Report the exact targets, CTest regexes, configurations, compiler, and seeds run. Also report relevant hardware or
optional backends that were unavailable locally.
