# Repository Guidelines

## Project Structure & Module Organization
`Eigen/` contains the main header-only library, with implementation details under `Eigen/src/`. Experimental and less-stable extensions live in `unsupported/Eigen/`, with matching tests in `unsupported/test/`. Stable unit tests are in `test/`, while `benchmarks/`, `demos/`, `doc/`, `blas/`, and `lapack/` cover performance work, examples, docs, and optional numerical backends. Build logic is rooted in `CMakeLists.txt` and helper modules under `cmake/` and `ci/`.

## Build, Test, and Development Commands
Configure an out-of-tree build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DEIGEN_BUILD_TESTING=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Use `-DEIGEN_BUILD_DEMOS=ON` or `-DEIGEN_BUILD_DOC=ON` only when working on demos or documentation. Run a focused test with `ctest --test-dir build -R basicstuff` or `ctest --test-dir build -R cxx11_tensor`. CI uses separate `checkformat`, `build`, and `test` stages.

## Coding Style & Naming Conventions
This is a C++14 codebase. Follow `.clang-format`: Google-based style, 2-space indentation, and a 120-column limit. Preserve include order; `SortIncludes` is disabled. Header/module macros use upper snake case such as `EIGEN_CORE_MODULE_H`. Test source files are usually lowercase with underscores in `test/` (for example `basicstuff.cpp`) and mirror the CMake test name; unsupported tests often keep feature-style names such as `EulerAngles.cpp`.

Run formatting and static analysis before submitting:

```bash
clang-format -i path/to/file.cpp
clang-tidy path/to/file.cpp -- -I.
```

## Testing Guidelines
Add or update tests for every bug fix or feature change. Register new tests through `ei_add_test(...)` in `test/CMakeLists.txt` or `unsupported/test/CMakeLists.txt`. Prefer targeted regression coverage near the affected module, then run the smallest relevant `ctest -R ...` filter before a full suite pass.

## Commit & Pull Request Guidelines
Recent history favors short, imperative subjects with an area prefix, for example `CI: Reduce artifact size...` or `Fix imag_ref for real scalar types...`. Keep one logical change per commit and one feature or bug fix per merge request. GitLab merge requests should include a clear description, linked issue (`Fixes #123` when applicable), test evidence, and documentation or changelog updates for substantial features.
