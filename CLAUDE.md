# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Eigen is a C++ template library for linear algebra (matrices, vectors, numerical solvers). It is **header-only** — there is nothing to compile for library usage. The public headers live in `Eigen/` and `unsupported/Eigen/`. The project requires C++14 (C++17 will be required in future releases).

Repository: https://gitlab.com/libeigen/eigen

## Build & Test Commands

```bash
# Configure (from a build directory)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DEIGEN_BUILD_TESTING=ON

# Build all tests
make buildtests -j$(nproc)

# Run all tests
ctest --test-dir build

# Run a single test or tests matching a pattern
ctest --test-dir build -R product_small

# Build and run a single test target directly
make -C build product_small && ./build/test/product_small

# Build BLAS/LAPACK implementations (optional)
cmake .. -DEIGEN_BUILD_BLAS=ON -DEIGEN_BUILD_LAPACK=ON
```

## Code Formatting

Uses clang-format with Google style base and 120-char line limit. CI enforces formatting via a `checkformat` stage.

```bash
clang-format-17 -i <changed-files>
```

**Always format changed files with `clang-format-17` before committing.** CI uses clang-format-17; other versions may produce different output.

The `.clang-format` config defines custom `StatementMacros` (`EIGEN_STATIC_ASSERT`, etc.) and `AttributeMacros` (`EIGEN_STRONG_INLINE`, `EIGEN_DEVICE_FUNC`, etc.). `SortIncludes` is disabled.

## Architecture

### Expression Templates

Eigen uses expression templates for lazy evaluation with zero-copy semantics. The type hierarchy:

- `EigenBase` → `DenseBase` → `MatrixBase` (matrix expressions) / `ArrayBase` (coefficient-wise expressions)
- Operations like `+`, `*`, `transpose()` return lightweight expression objects (`CwiseBinaryOp`, `CwiseUnaryOp`, `Product`, etc.)
- Evaluation is deferred until assignment via `AssignEvaluator` and `CoreEvaluators`

### Module System

Each module (e.g. `Eigen/Core`, `Eigen/QR`, `Eigen/SparseCore`) is a single header that includes its implementation from `Eigen/src/<Module>/`. Key modules:

- **Core**: Matrix/Array classes, basic operations, vectorization, memory, products
- **Cholesky, LU, QR, SVD, Eigenvalues**: Dense decompositions
- **SparseCore + SparseLU/SparseQR/SparseCholesky**: Sparse linear algebra
- **Geometry**: Transforms, quaternions, rotations
- **ThreadPool**: Multi-threading support (in `unsupported/`)

### Vectorization / Architecture Backends

Architecture-specific SIMD code lives in `Eigen/src/Core/arch/<backend>/`:

- **x86**: `SSE/`, `AVX/`, `AVX512/`
- **ARM**: `NEON/`, `SVE/`
- **Other**: `AltiVec/`, `ZVector/`, `MSA/`, `LSX/`, `HVX/`, `RVV10/`
- **Accelerators**: `GPU/`, `SYCL/`, `HIP/`
- **Compiler intrinsics**: `clang/` (clang vector extensions)
- **Fallback**: `Default/`

Auto-detection happens in `src/Core/util/ConfigureVectorization.h`. The `Packet` abstraction (`GenericPacketMath.h`) provides a uniform interface over all backends.

### Matrix Product Kernels (Performance-Critical)

The GEMM implementation follows a GEBP (General Block Panel) approach:

- `products/GeneralBlockPanelKernel.h` — core micro-kernel and blocking logic
- `products/GeneralMatrixMatrix.h` — top-level GEMM dispatch
- `products/GeneralMatrixVector.h` — GEMV specializations
- Architecture-specific kernels in `arch/<backend>/` directories (e.g. `arch/NEON/GeneralBlockPanelKernel.h`)

### Key Macros

- `EIGEN_DEVICE_FUNC` — marks functions for GPU compilation
- `EIGEN_STRONG_INLINE` / `EIGEN_ALWAYS_INLINE` — inlining hints critical for performance
- `EIGEN_STATIC_ASSERT` — compile-time checks with Eigen-specific diagnostics
- `EIGEN_USE_BLAS` / `EIGEN_USE_MKL` / `EIGEN_USE_AOCL` — enable external BLAS backends

## Testing

Tests are in `test/` with one `.cpp` per test. The CMake macro `ei_add_test()` (defined in `cmake/EigenTesting.cmake`) registers tests with CTest. Tests use Eigen's own test framework (`test/main.h`), not Google Test.

Tests with external solver backends (CHOLMOD, UMFPACK, KLU, SuperLU, PaStiX, SPQR) are conditionally built based on library availability.

## CI

GitLab CI (`.gitlab-ci.yml` + `ci/` directory) with stages: `checkformat` → `build` → `test` → `deploy`. Tests run on Linux and Windows with multiple compiler configurations.
