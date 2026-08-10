# Formatting And CI

Use the checked-out configuration as the source of truth. [`.gitlab-ci.yml`](../.gitlab-ci.yml) defines stages and
includes; [`ci/*.gitlab-ci.yml`](../ci) and [`ci/scripts/`](../ci/scripts) define the actual jobs. Default MR pipelines
run a limited smoke matrix; labels such as `affected-tests`, `all-tests` and `gpu-tests`, plus scheduled or manually
started pipelines, enable broader jobs. A green default MR pipeline is not proof that every supported configuration was
exercised.

Build jobs publish the configured build directory as an artifact. Their paired test jobs consume that artifact and
run CTest without rebuilding. When changing either side, keep the test job's `needs`, CTest label or filter, and the
corresponding build target consistent; otherwise CTest can discover tests whose executables are absent.

## Test Tiers On Merge Requests

Three tiers, in increasing cost:

| Tier | Trigger | What runs |
|---|---|---|
| smoke | every MR | the fixed list in [`cmake/EigenSmokeTestList.cmake`](../cmake/EigenSmokeTestList.cmake), usually one part per test, at baseline ISA on x86-64, aarch64 and riscv64 |
| affected | `affected-tests` label | every test the diff can reach, all parts, on x86-64 AVX2 and aarch64, plus the ISA of any packet-math backend the diff touches |
| full | `all-tests` label | the whole suite across the entire compiler and ISA matrix |

The affected tier exists because the smoke list samples: it is broad but shallow, so a change confined to one module
gets only the one part of each related test that the list happens to name. Reach for `affected-tests` when a change is
module-local and you want depth without paying for the full matrix.

[`scripts/affected_tests.py`](../scripts/affected_tests.py) computes the selection in the `select:tests` job and writes
`affected/targets.txt` and `affected/ctest_regex.txt`, which the paired build and test jobs consume through
`EIGEN_CI_BUILD_TARGET_FILE` and `EIGEN_CI_CTEST_REGEX_FILE`. Run it locally the same way CI does:

```bash
python3 scripts/affected_tests.py --base-sha $(git merge-base origin/master HEAD)
python3 scripts/test_affected_tests.py     # unit tests, also run by the CI job
```

Selection follows the textual `#include` graph, ignoring preprocessor guards, so it is a strict superset of the real
compile dependency and never drops an affected test. Because Eigen is header-only and the umbrella headers are hubs,
a change under `Eigen/src/Core` typically reaches every test and the selector degrades to `buildtests` — that is the
correct answer, not a failure. Changes to CMake, `ci/`, or the BLAS/LAPACK shims also force the full suite, since they
invalidate the mapping itself.

Two properties are easy to break when editing this path. Targets absent from a configuration (optional dependencies
such as CHOLMOD or SYCL) must be filtered against `ninja -t targets` after cmake configure, because ninja aborts on an
unknown target. And a missing selection artifact must fail the job rather than fall through to the default target,
which would silently build everything.

### Backend-Triggered Configurations

Every job in the default smoke matrix builds at baseline ISA, so a change under `Eigen/src/Core/arch/AVX512` gets no
AVX-512 compilation at all unless someone applies `all-tests`. Under the `affected-tests` label the tier adds the
configuration that targets the backend the diff touches, through `rules:changes:`:

| Backend directory | Added configuration |
|---|---|
| `arch/SSE` | x86-64 gcc-10 baseline, AVX, and AVX-512DQ |
| `arch/AVX` | x86-64 gcc-10 AVX and AVX-512DQ |
| `arch/AVX512` | x86-64 gcc-10 AVX-512DQ |
| `arch/NEON` | 32-bit arm (aarch64 already runs unconditionally) |
| `arch/AltiVec` | ppc64le gcc-14 |
| `arch/LSX` | loongarch64 gcc-14 |
| `arch/RVV10` | riscv64 gcc-15 |
| `arch/SVE`, `arch/SME` | the full SME build, compile-only |

A wider x86 configuration compiles the narrower backends' headers, which is why SSE fans out to three builds. SVE and
SME get compile coverage rather than a selection because their per-SVL test jobs already filter to a curated target
subset through `EIGEN_CI_CTEST_REGEX`, which a selection would fight with.

`arch/ZVector`, `arch/MSA`, `arch/HVX` and the `arch/GPU`, `arch/HIP` and `arch/SYCL` backends have no matching test
configuration, so a change there gets only the two unconditional jobs. The GPU backends have their own `gpu-tests`
label. When adding a runner for one of these, add the trigger here too.

## Worktree-Safe Formatting

Inspect `git status --short` before formatting and preserve unrelated changes. Eigen requires `clang-format-17`
exactly. Format only files owned by the task:

```bash
clang-format-17 -i path/to/file.cpp path/to/header.h
clang-format-17 --dry-run --Werror path/to/file.cpp path/to/header.h
git clang-format --binary clang-format-17 --diff <base-sha>
```

`.clang-format` intentionally disables include sorting and registers Eigen-specific macros and attributes. Do not
reorder includes or restyle those macros manually.

[`scripts/format.sh`](../scripts/format.sh) rewrites every matching file in the tree in parallel. Run it only when the
worktree is clean or every affected change is owned by the task. Review `git diff` afterward in either case.

## Local Checks

Run checks relevant to the changed files and report unavailable tools:

```bash
codespell --config setup.cfg path/to/changed-file
reuse lint
```

The whole-tree codespell invocation used by CI can expose pre-existing findings. Do not modify unrelated files merely
to make a local broad scan clean. In the current CI configuration, clang-format, codespell, and clang-tidy jobs are
`allow_failure`; treat their diagnostics as review findings anyway. The REUSE job is blocking.

Source-like files normally carry an inline SPDX copyright and license header using the file type's comment syntax.
Files that should not carry inline comments need coverage in [`REUSE.toml`](../REUSE.toml). To process selected new
source files with the repository helper, pass them explicitly because its default scan considers tracked files:

```bash
python3 scripts/add_spdx_headers.py --paths path/to/new-file.cpp
```

## Clang-Tidy

Use the CI driver rather than invoking clang-tidy directly on an implementation header; the driver routes such a
header through its public umbrella include.

```bash
cmake -G Ninja -S . -B .tidy-build \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DEIGEN_BUILD_TESTING=ON
ci/scripts/run-clang-tidy.sh <base-sha> .tidy-build
```

The driver examines files committed between `<base-sha>` and `HEAD`; uncommitted-only edits are not included. Eigen's
`.clang-tidy` policy is authoritative. Do not apply generic `modernize-*` or `cppcoreguidelines-*` campaigns.

## Before Review

1. Inspect `git diff` and `git diff --check`.
2. Format the exact changed source files with clang-format-17.
3. Run the focused builds and tests documented in [`testing.md`](testing.md).
4. Run applicable spelling, REUSE, and clang-tidy checks.
5. State what ran, what did not run, and why. Do not claim coverage from jobs or hardware that were unavailable.
