# Comparison harness contracts

This file is the single source of truth for the cross-library benchmark comparison harness. Everything else in
`benchmarks/comparison/` is built against it. Where a choice below is arbitrary, it has been **made** and marked
*(arbitrary, fixed)*; do not relitigate it in an implementation, change it here first.

Read [`../../AGENTS.md`](../../AGENTS.md) and [`../../.agents/benchmarking.md`](../../.agents/benchmarking.md) first.
This document refines them for the comparison harness; it does not override them.

Companion files: [`ops.toml`](ops.toml) (the operation and shape registry) and
[`result_schema.json`](result_schema.json) (JSON Schema for one run's output).

## 0. Scope, versioning, and layout

Phase 1 is a vertical slice: **GEMM, one machine, one reference vendor**. Every contract here is written so that
adding an operation, a machine, or a vendor is a change to `ops.toml`, a machine config, or a CMake option, and
never a change to `run.py`, `reduce.py`, `render.py`, or `plots.py`.

Two constraints bind every consumer:

- **Incremental.** Every consumer must render correctly from a partial dataset — one machine, one vendor, or one op.
  A combination that was not measured is stated as such (section 6). It is never silently dropped and never rendered
  as zero.
- **Extensible.** Nothing in the Python or the rendered output may enumerate operations, vendors, ISA targets, or
  scalar types in code. They come from `ops.toml`, the machine configs, and the data.

Three independent version numbers, all semantic:

| Version | Lives in | Bump when |
|---|---|---|
| `ops.toml` `schema_version` | `ops.toml` | the shape of `ops.toml` changes (not when an op is added) |
| result `schema_version` | `result_schema.json` (`const`) | the result file shape changes |
| harness `version` | `run.py` `HARNESS_VERSION` | `run.py` behaviour changes observably |

Layout, with the agent that owns each path:

```
benchmarks/comparison/
  CONTRACTS.md          this file                                   (contracts)
  ops.toml              operation + shape registry                  (contracts)
  result_schema.json    JSON Schema for one run's output            (contracts)
  bench_compare.h       REGISTER_COMPARISON_POINT, arm metadata, main (C++ agent)
  bench_gemm_compare.cpp GEMM, both arms                            (C++ agent)
  CMakeLists.txt        comparison targets                          (CMake agent)
  vendors.cmake         the reference-BLAS vendor table              (CMake agent)
  machines/<id>.toml    per-machine config                          (machines agent)
  run.py reduce.py render.py plots.py                               (python agents)
    ^-- all four import the shared grammar, registry and merged-document
        helpers from _common.py, so the name parser, flop evaluator and
        config-id algorithm have exactly one definition each
  tests/  tests/fixtures/                                           (test agent)
  results/              generated, git-ignored
benchmarks/bench_common.h   shared flop + counter helpers           (C++ agent)
```

`benchmarks/bench_common.h` deliberately sits one level up: it is general benchmark infrastructure that
`Core/`, `Cholesky/`, and the comparison tree all use. Both headers are included **repo-root-relative**:

```cpp
#include "benchmarks/bench_common.h"
#include "benchmarks/comparison/bench_compare.h"
```

which already resolves, because `eigen_add_benchmark` puts `${EIGEN_SOURCE_DIR}` (the repository root) on the
include path. No CMake include-directory change is needed. *(arbitrary, fixed)*

`benchmarks/CMakeLists.txt` must gain `set(CMAKE_CXX_STANDARD 17)` / `set(CMAKE_CXX_STANDARD_REQUIRED ON)`. The
benchmarks project sets no standard today, yet `Core/bench_gemm.cpp` already uses `if constexpr` in its `HAVE_BLAS`
block, which would fail under gcc-10's default `gnu++14`. Eigen's C++14 rule (`AGENTS.md` rule 3) constrains
`Eigen/` headers, not benchmark consumers.

---

## 1. The benchmark name grammar

This is the join key between the C++ registrations and the Python reducer. It is the most load-bearing thing in this
document.

### 1.1 Grammar

```
name    := op "/" arm "/" scalar dims [ threads ]
op      := /[A-Z][A-Z0-9_]*/          a key of [ops] in ops.toml
arm     := /[a-z][a-z0-9_]*/          "eigen", or a vendor key
scalar  := "f16" | "bf16" | "f32" | "f64" | "c32" | "c64"
dims    := ( "/" dimname ":" value )+ at least one
dimname := /[a-z][a-z0-9_]*/          a member of the op's shape family "dims", in that order
value   := /[0-9]+/                   a non-negative decimal integer
threads := "/threads:" /[0-9]+/       present only when ->Threads(n) was used
```

Concrete names:

```
GEMM/eigen/f64/m:1024/n:1024/k:1024
GEMM/openblas/f64/m:1024/n:1024/k:1024
GEMM/eigen/c64/m:4096/n:96/k:96
GEMV/accelerate/f32/m:10000/n:100
POTRF/eigen/f64/n:512
TRSM_LLNN/mkl/f64/n:2048/nrhs:16
GESDD/eigen/f64/m:10000/n:1000
FULLPIVLU/eigen/f64/m:512/n:512
GEMM/eigen/f64/m:2048/n:2048/k:2048/threads:8
```

Legal characters in a whole name: `A-Z a-z 0-9 _ : / .` — nothing else. In particular **space, comma, `-`, `<`, `>`,
`(`, `)`, `[`, `]`, `*`, `+`, `%` are forbidden.** `--benchmark_filter` takes a POSIX extended regex, and names also
travel through shell arguments, CSV, Doxygen tables, and filenames; the restricted set survives all of them. `<`/`>`
in particular are what raw `BENCHMARK_TEMPLATE` registration produces, which is exactly why every comparison
registration sets `->Name(...)` explicitly instead.

### 1.2 How the name is produced

`->Name()` supplies fields 0-2; Google Benchmark appends the dimensions from `->ArgNames()` + `->Args()` and the
optional `/threads:N`. Verified against Google Benchmark 1.9.5:

```cpp
BENCHMARK(BM_X)->Name("GEMM/eigen/f64")->ArgNames({"m", "n", "k"})->Args({1024, 1024, 1024});
// full name: GEMM/eigen/f64/m:1024/n:1024/k:1024
```

`->ArgNames()` is what makes the dimensions **self-describing**. The parser recovers dimension *names*, not
positions, so `POTRF/.../n:512` and `GEMV/.../m:512/n:512` cannot be confused, an op with three dimensions cannot be
misread as one with two, and reordering a family's `dims` is caught rather than silently transposing a table.
Registering a comparison benchmark without `->ArgNames()` is a contract violation: the parser rejects any dimension
field lacking a `:`.

### 1.3 How the name is parsed

```python
def parse_benchmark_name(run_name: str) -> dict:
    fields = run_name.split("/")
    op, arm, scalar = fields[0], fields[1], fields[2]
    threads, shape, dim_order = 1, {}, []
    for field in fields[3:]:
        key, _, value = field.partition(":")
        if not _ or not value.isdigit():
            raise ValueError(f"unparseable field {field!r} in {run_name!r}")
        if key == "threads":
            threads = int(value)
        elif key in RESERVED_DIMS:      # see below
            raise ValueError(f"unsupported registration form {key!r} in {run_name!r}")
        else:
            dim_order.append(key)
            shape[key] = int(value)
    return {"op": op, "arm": arm, "scalar": scalar,
            "shape": shape, "shape_dims": dim_order, "threads": threads}
```

`RESERVED_DIMS = {"repeats", "iterations", "min_time", "min_warmup_time", "manual_time", "real_time", "big_o",
"rms", "process_time"}`. These are the other suffixes Google Benchmark can append; a comparison benchmark must not
use the registration forms that produce them, and seeing one is an error rather than a silently ignored field.
`threads` is the single reserved field the parser accepts, and it is never a shape dimension.

It is also **not** the harness's thread count. `/threads:N` is Google Benchmark's own multi-threaded registration
form (`->Threads(n)`), which no comparison benchmark uses: threading is controlled through the library environment
instead (section 7), so every emitted name parses back as `threads = 1`. A distilled row is therefore recorded
under the run's **configured** thread count, and a name that spells the field out anyway must agree with it or the
two are describing different measurements. Keying the plan on the parsed value instead measured a `--threads 8`
grid in full and then discarded every row of it as `not_implemented`.

Validation after parsing, all mandatory:

1. `op` is a key of `[ops]` in `ops.toml`. `run.py`, distilling a binary's raw output, reports the row as
   `unknown_op` and skips it with a warning; `reduce.py`, asked to accept a finished result file, refuses it
   with exit 2 (section 4.2). Distilling tolerates a binary that registers more than the registry knows;
   merging does not, because by then the number would reach a published table.
2. `scalar` is in that op's `scalars` list.
3. `dim_order` equals `shape_families[ops[op].shape_family].dims` exactly, including order. A mismatch is a hard
   error, not a warning: it means the C++ and the registry have diverged, and every number from that binary is
   suspect.
4. `(op, arm, scalar, shape, threads)` is unique within one result file.

Collisions are impossible by construction: field 0 disambiguates operations, field 1 disambiguates arms, field 2
disambiguates scalar variants, and dimension arity differences are absorbed because each dimension carries its name.
Two ops with different dimension counts never produce the same string because their `op` fields differ.

### 1.4 Aggregate rows — use `run_name`, never `name`

With `--benchmark_repetitions=N --benchmark_report_aggregates_only=false`, Google Benchmark emits `N` rows with
`"run_type": "iteration"` **and** four rows with `"run_type": "aggregate"` whose `"name"` carries a suffix:
`_mean`, `_median`, `_stddev`, `_cv`.

Every row — iteration and aggregate — also carries `"run_name"`, which is the **suffix-free** full name. Consumers
MUST join on `run_name` and MUST NOT strip suffixes from `name`. Stripping is genuinely unsafe here: `_` is legal
inside an arm key and inside an op key (`TRSM_LLNN`), so a suffix-stripping parser would corrupt real names.

The harness keeps the `iteration` rows and **discards** the vendor aggregates, computing median/MAD/min/max itself
(section 5). *(arbitrary, fixed — it makes the dispersion statistic identical across every contribution regardless
of the Google Benchmark version that produced it.)*

### 1.5 Variants

An operation that specialises a mnemonic — a transpose combination, a `side`/`uplo` choice — gets its own `ops.toml`
key with an underscore-suffixed name (`GEMM_TN`, `TRSM_LLNN`) and sets `base_mnemonic` and `variant` so renderers
group it under the parent. Non-integer parameters are never encoded as shape dimensions, because dimension values
are integers by grammar.

### 1.6 The counter, and the one trap in it

Comparison benchmarks emit exactly one counter, named **`GFLOPS`** (the existing house name, kept for continuity).

**The JSON value is flops per second, not gigaflops.** `benchmark::Counter::kIs1000` selects the base for the
console's `k`/`M`/`G` suffix and does not scale the JSON value. Measured on 1.9.5: a counter constructed with
`flops = 1.0` over a 2 µs iteration reports `499999.99` in JSON, i.e. `1/2e-6`. Every consumer divides by `1e9`
itself, once, at the boundary defined in section 5.

---

## 2. `benchmarks/bench_common.h` — shared flop and counter helpers

Today `(NumTraits<Scalar>::IsComplex ? 8 : 2)` is copy-pasted into six files (`Core/bench_gemv.cpp`,
`bench_dot.cpp`, `bench_symv.cpp`, `bench_syr.cpp`, `bench_syr2.cpp`, `bench_trmv.cpp`) and a symmetric-factorization
cost loop is duplicated between `Cholesky/bench_cholesky.cpp` and `Cholesky/bench_bunchkaufman.cpp`. This header
subsumes both.

**Every helper reproduces the value the existing benchmark already reports, exactly.** Adopting the header in an
existing file must be a pure refactor that moves no published number. See the note on `trmvFlops` below.

```cpp
// benchmarks/bench_common.h
// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0
#ifndef EIGEN_BENCHMARKS_BENCH_COMMON_H
#define EIGEN_BENCHMARKS_BENCH_COMMON_H

#include <benchmark/benchmark.h>
#include <Eigen/Core>

namespace eigen_bench {

using Eigen::Index;

// Multiplier turning a count of real multiply-add PAIRS into scalar flops:
// 2 for real, 8 for complex (4 real multiplies + 4 real adds per complex fma).
template <typename Scalar>
constexpr double flopScale();

// Multiplier turning a REAL-arithmetic flop count into the same count for
// Scalar: 1 for real, 4 for complex. Invariant: flopScale<S>() == 2 * complexFactor<S>().
// This is the bridge from ops.toml: an op's scalar flop count is
//   complexFactor<Scalar>() * <ops.OP.flops.real evaluated on the shape>.
template <typename Scalar>
constexpr double complexFactor();

// ---- Level 1 -------------------------------------------------------------
template <typename Scalar> double dotFlops(Index n);                 // scale * n
template <typename Scalar> double axpyFlops(Index n);                // scale * n

// ---- Level 2 -------------------------------------------------------------
template <typename Scalar> double gemvFlops(Index m, Index n);       // scale * m * n
template <typename Scalar> double symvFlops(Index n);                // scale * n * n
template <typename Scalar> double trmvFlops(Index n);                // scale * n * n  (see note)
template <typename Scalar> double syrFlops(Index n);                 // scale * n * (n + 1) / 2
template <typename Scalar> double syr2Flops(Index n);                // scale * n * (n + 1)

// ---- Level 3 -------------------------------------------------------------
template <typename Scalar> double gemmFlops(Index m, Index n, Index k);   // scale * m * n * k
template <typename Scalar> double trsmFlops(Index n, Index nrhs);         // complexFactor * n * n * nrhs

// ---- Factorizations and decompositions -----------------------------------
// Closed form of the summation loop in Cholesky/bench_cholesky.cpp and
// Cholesky/bench_bunchkaufman.cpp: sum_j 2*((n-1-j)*j + (n-1-j) + j)
//   == n*(n-1)*(n-2)/3 + 2*n*(n-1)
// Evaluated in double so the cubic term cannot overflow Index.
template <typename Scalar> double symmetricFactorizationFlops(Index n);
template <typename Scalar> double getrfFlops(Index m, Index n);      // cf * (m*n^2 - n^3/3)
template <typename Scalar> double geqrfFlops(Index m, Index n);      // cf * (2*m*n^2 - 2*n^3/3)
template <typename Scalar> double gesddFlops(Index m, Index n);      // cf * (8*m*n^2 + 4*n^3/3), nominal
template <typename Scalar> double syevFlops(Index n);                // cf * 9*n^3, nominal

// ---- Counters ------------------------------------------------------------
// `flops` is the flop count of ONE benchmark iteration. The emitted JSON value
// is flops per second; kIs1000 affects console rendering only (section 1.6).
inline benchmark::Counter GflopsCounter(double flops) {
  return benchmark::Counter(flops, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::kIs1000);
}

// Canonical counter name for every comparison benchmark.
inline constexpr const char* kFlopCounterName = "GFLOPS";

// state.counters[kFlopCounterName] = GflopsCounter(flops);
inline void setFlopRate(benchmark::State& state, double flops);

// Byte-rate counter for bandwidth-bound level-1/2 work. Counter name "BYTES/s".

}  // namespace eigen_bench
#endif
```

Notes that are part of the contract:

- `GflopsCounter` is the existing wrapper in `BLAS/bench_blas.cpp:27`, moved verbatim. That file should include the
  header and delete its local copy.
- `trmvFlops` returns `scale * n * n`, matching `Core/bench_trmv.cpp` today. The exact count for a triangular
  matrix-vector product is `scale * n * (n + 1) / 2`; the existing benchmark overcounts by roughly 2x. The helper
  preserves the existing value so adoption changes no published number. Correcting it is a separate change with its
  own justification, and `ops.toml` will need a matching `flops.real` when TRMV is registered.
- `Index` arguments, never `int`. `state.range(i)` returns `int64_t`; convert once at the top of the benchmark body.
- All products are formed in `double` before multiplying by the scale, so a cubic term in a 4-digit dimension cannot
  overflow.

---

## 3. `benchmarks/comparison/bench_compare.h` and `REGISTER_COMPARISON_POINT`

### 3.1 What the caller writes

```cpp
#include "benchmarks/bench_common.h"
#include "benchmarks/comparison/bench_compare.h"

template <typename Scalar>
static void BM_GemmEigen(benchmark::State& state) { /* ... */ }

template <typename Scalar>
static void BM_GemmReference(benchmark::State& state) { /* ... */ }

// clang-format off
#define GEMM_DIM_NAMES {"m", "n", "k"}

#define GEMM_POINT(...)                                                                             \
  REGISTER_COMPARISON_POINT(GEMM, f32, float,              BM_GemmEigen, BM_GemmReference, GEMM_DIM_NAMES, __VA_ARGS__) \
  REGISTER_COMPARISON_POINT(GEMM, f64, double,             BM_GemmEigen, BM_GemmReference, GEMM_DIM_NAMES, __VA_ARGS__) \
  REGISTER_COMPARISON_POINT(GEMM, c32, eigen_bench::c32_t, BM_GemmEigen, BM_GemmReference, GEMM_DIM_NAMES, __VA_ARGS__) \
  REGISTER_COMPARISON_POINT(GEMM, c64, eigen_bench::c64_t, BM_GemmEigen, BM_GemmReference, GEMM_DIM_NAMES, __VA_ARGS__)

#define GEMM_SIZES(POINT) \
  POINT(2,2,2) POINT(3,3,3) /* ... the grid from ops.toml shape_families.square3 ... */

GEMM_SIZES(GEMM_POINT)

#undef GEMM_SIZES
#undef GEMM_POINT
#undef GEMM_DIM_NAMES
// clang-format on

EIGEN_BENCH_COMPARISON_MAIN();
```

The grid is one object-like list macro, shared by both arms by construction, so the two arms cannot drift onto
different size sets. It is still declarative — no `Apply()` and no `benchmark::internal::Benchmark*` internal API
(`.agents/benchmarking.md`) — but it is a list of points rather than the `->Args()` arrow chain used elsewhere in
`benchmarks/`, because **the two arms of a comparison must be registered adjacently**. Google Benchmark runs
instances in registration order; an arrow chain applied to two `Benchmark` objects emits one arm's whole grid and
then the other's, which over the default grid puts minutes of thermal and background drift between the two numbers
a ratio is formed from, systematically and always in the same direction. Step 4 of `.agents/benchmarking.md`
requires the alternating order instead. *(measured: on an M4, `GEMM/accelerate/f64` at 1024^3 read 3.49e11 flop/s
run cold and 2.96e11 flop/s after ~60 s of preceding load in the same process — a 15% arm-to-arm bias with nothing
in the result file disclosing it.)*

`DIM_NAMES` must be passed as the **name** of an object-like macro. Braces do not protect commas from macro
argument splitting, so a literal `{"m", "n", "k"}` in the argument list would be three arguments.

### 3.2 What it expands to

```cpp
#define EIGEN_BENCH_STRINGIZE_(x) #x
#define EIGEN_BENCH_STRINGIZE(x) EIGEN_BENCH_STRINGIZE_(x)

#ifdef EIGEN_BENCH_REFERENCE_ARM
#define REGISTER_COMPARISON_POINT(MNEMONIC, SCALAR_TAG, SCALAR_TYPE, EIGEN_FN, REF_FN, DIM_NAMES, ...) \
  BENCHMARK_TEMPLATE(EIGEN_FN, SCALAR_TYPE)                                                            \
      ->Name(#MNEMONIC "/eigen/" #SCALAR_TAG)->ArgNames(DIM_NAMES)->Args({__VA_ARGS__});               \
  BENCHMARK_TEMPLATE(REF_FN, SCALAR_TYPE)                                                              \
      ->Name(#MNEMONIC "/" EIGEN_BENCH_STRINGIZE(EIGEN_BENCH_REFERENCE_ARM) "/" #SCALAR_TAG)           \
      ->ArgNames(DIM_NAMES)->Args({__VA_ARGS__});
#else
#define REGISTER_COMPARISON_POINT(MNEMONIC, SCALAR_TAG, SCALAR_TYPE, EIGEN_FN, REF_FN, DIM_NAMES, ...) \
  BENCHMARK_TEMPLATE(EIGEN_FN, SCALAR_TYPE)                                                            \
      ->Name(#MNEMONIC "/eigen/" #SCALAR_TAG)->ArgNames(DIM_NAMES)->Args({__VA_ARGS__});
#endif
```

Semantics, all normative:

- `MNEMONIC` and `SCALAR_TAG` are bare tokens, stringized by the macro. `SCALAR_TYPE` is the C++ type.
- The Eigen arm always registers. The reference arm registers **only** when `EIGEN_BENCH_REFERENCE_ARM` is defined.
- CMake passes the arm key as a **bare token**: `-DEIGEN_BENCH_REFERENCE_ARM=openblas`. The header stringizes it, so
  CMake never has to escape a quote. *(arbitrary, fixed)*
- The two arms are spelled out in the macro body rather than delegated to a shared per-arm macro, because passing
  `DIM_NAMES` down another macro-argument level would split it on its own commas.
- The macro body is a sequence of complete statements and needs no trailing `;` at the call site.
- One binary carries at most one reference arm. Comparing two vendors means two binaries and two result files, which
  merge cleanly (section 5) because the arm key differs.
- Both arms must be timed identically: same grid macro, same `flops_per_iteration` helper, same `DoNotOptimize` /
  `ClobberMemory` discipline, same allocation and fill outside the timed loop, and result validation outside the
  timed loop.

### 3.3 Arm metadata and the custom `main`

`run.py` must not guess the reference library's version. The binary reports it, using
`benchmark::AddCustomContext`, whose keys land verbatim in the JSON `context` object (verified on 1.9.5).

```cpp
namespace eigen_bench {
const char* referenceArmKey();          // "openblas" ... ; "" when no reference is linked
bool hasReference();
std::string referenceLibraryName();     // display name
std::string referenceLibraryVersion();  // queried from the library at runtime
std::string referenceLibraryPath();     // "" when unknown

// Publishes eigen_bench.* keys into the Google Benchmark JSON context.
// MUST run before benchmark::Initialize().
void publishArmContext();
}
```

Required context keys, all strings:

```
eigen_bench.schema_version              "1.0.0"
eigen_bench.reference_arm               "" when no reference is linked
eigen_bench.reference_library_name
eigen_bench.reference_library_version   "unknown" only as a last resort; see below
eigen_bench.reference_library_path
eigen_bench.reference_interface         "lp64" | "ilp64" | ""
eigen_bench.reference_threading         "sequential" | "openmp" | "pthreads" | "tbb" | "gcd" | ""
eigen_bench.eigen_commit                40 hex chars
eigen_bench.eigen_dirty                 "true" | "false"
eigen_bench.compiler_id
eigen_bench.compiler_version
eigen_bench.cxx_standard
eigen_bench.cxx_flags                   the exact flag string
eigen_bench.isa_target
eigen_bench.eigen_nb_threads
eigen_bench.thread_env                  JSON object literal, thread-count env vars actually set
eigen_bench.ops_toml_sha256
```

Version query per vendor, best available first, falling back to the CMake-supplied
`EIGEN_BENCH_REFERENCE_VERSION_FALLBACK` string, and only then to `"unknown"` — which obliges `run.py` to emit a
`provenance_gaps` entry:

| Arm | Query |
|---|---|
| `openblas` | `openblas_get_config()` / `OPENBLAS_VERSION` |
| `mkl` | `mkl_get_version_string()` |
| `blis`, `aocl` | `bli_info_get_version_str()` |
| `armpl` | `armplversion()` / `ARMPL_BUILD_STRING` |
| `nvpl` | `nvpl_blas_get_version()` |
| `accelerate` | no query; CMake fallback = the macOS product version |
| `netlib` | no query; CMake fallback = the packaged version |
| `eigenblas` | Eigen's own version macros |

```cpp
#define EIGEN_BENCH_COMPARISON_MAIN()                     \
  int main(int argc, char** argv) {                       \
    ::eigen_bench::publishArmContext();                   \
    ::benchmark::Initialize(&argc, argv);                 \
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1; \
    ::benchmark::RunSpecifiedBenchmarks();                \
    ::benchmark::Shutdown();                              \
    return 0;                                             \
  }                                                       \
  static_assert(true, "")  /* consume the trailing ; */
```

Defining `main` in the translation unit is safe alongside the `benchmark::benchmark_main` that `eigen_add_benchmark`
links: the archive member supplying `main` is only extracted while the symbol is still undefined. Verified by
linking and running such a binary against 1.9.5 — the custom `main` runs and its custom context appears.

---

## 4. CLI surfaces

Common to all four tools: `-h/--help` prints usage and exits 0. Diagnostics go to **stderr**; machine-readable
output goes to **stdout** only when the output path is `-`. `--` terminates option parsing. Repeatable options are
marked; comma-separated lists are accepted wherever a repeatable option is, and are equivalent. No tool ever writes
outside its declared output path or directory. Python 3.11+ (`tomllib`).

### 4.1 `run.py` — measure

```
run.py --machine ID [options]
```

| Flag | Default | Meaning |
|---|---|---|
| `--machine ID` | *required* | key of `machines/<ID>.toml` |
| `--build-dir DIR` | `build-comparison` | CMake build dir for `benchmarks/` |
| `--ops LIST` | every op with `status = "implemented"` | repeatable; `all` accepted |
| `--scalars LIST` | `f64` | repeatable |
| `--arms LIST` | `eigen` plus every reference arm the build configured | repeatable; `eigen` is always included |
| `--groups LIST` | each family's `default_groups` | repeatable; shape-group subset |
| `--threads LIST` | `1` | repeatable; see `--allow-sequential-eigen` |
| `--allow-sequential-eigen` | off | permit `--threads N>1` against a build whose Eigen has no OpenMP |
| `--repetitions N` | `10` | `--benchmark_repetitions` |
| `--min-time SPEC` | `0.5s` | `--benchmark_min_time`, verbatim |
| `--filter REGEX` | none | ANDed with the generated filter |
| `--results-dir DIR` | `benchmarks/comparison/results` | |
| `--out PATH` | `<results-dir>/<machine>/<eigen_short>/<run_id>.json` | `-` writes to stdout |
| `--configure` / `--no-configure` | `--configure` | run the CMake configure step |
| `--build` / `--no-build` | `--build` | build the comparison targets |
| `-j N` | `os.cpu_count()` | build parallelism |
| `--dry-run` | off | print the plan (targets, filters, cell count) to stdout, measure nothing, exit 0 |
| `--list-cells` | off | print one `op arm scalar dim:val...` line per planned cell to stdout, exit 0 |
| `--allow-dirty` | off | permit a dirty Eigen worktree |
| `--allow-noisy` | off | permit load average above the machine config's `max_load_avg` |
| `--note TEXT` | none | free text into `provenance.run.notes` |
| `--isa LIST` | the machine's `default_isa_target` | repeatable; one result file per ISA, which is the only way to select one |
| `--machines-dir DIR` | `benchmarks/comparison/machines` | where `--machine` is resolved; makes the harness testable without writing into `machines/` |
| `--ops-toml PATH` | `benchmarks/comparison/ops.toml` | |
| `--schema PATH` | `benchmarks/comparison/result_schema.json` | |
| `-v` / `--verbose` | off | repeatable |
| `--version` | | print harness version, exit 0 |

stdin is never read. Exit codes:

| Code | Meaning |
|---|---|
| 0 | success — including a run in which every cell was recorded as `not_measured` *because nothing was runnable*; that is a valid partial result |
| 1 | usage error |
| 2 | configuration error: missing machine config, invalid `ops.toml`, unknown op/scalar/arm/group, or `--threads N>1` against a build whose Eigen has no OpenMP without `--allow-sequential-eigen` |
| 3 | refused: dirty Eigen worktree without `--allow-dirty` |
| 4 | refused: machine too noisy without `--allow-noisy` |
| 5 | build failure |
| 6 | every benchmark executable failed at runtime, leaving nothing measured and nothing classifiable — or a unit planned runnable cells and measured none of them, which is a broken run rather than a coverage statement |
| 7 | the produced file failed `result_schema.json` validation — a harness bug. The file is still written, with the suffix `.invalid.json`, so it can be inspected |

### 4.2 `reduce.py` — merge result files into the intermediate

```
reduce.py [options] [FILE ...]
```

`FILE` arguments are result files. With no `FILE`, or with a literal `-`, newline-separated paths are read from
stdin, so `find ... | reduce.py` works.

| Flag | Default | Meaning |
|---|---|---|
| `--glob PATTERN` | none | repeatable; expanded relative to cwd |
| `--out PATH` | `-` (stdout) | merged intermediate |
| `--merge INTO` | none | additively merge into an existing merged file |
| `--ops-toml PATH` | `benchmarks/comparison/ops.toml` | |
| `--schema PATH` | `benchmarks/comparison/result_schema.json` | |
| `--validate` / `--no-validate` | `--validate` | validate every input against the schema |
| `--skip-invalid` | off | warn and continue instead of failing on an invalid input |
| `--baseline ARM` | the single non-`eigen` arm present | reference arm for ratios; ambiguity is an error, including under `--merge`, where the reference arms of both documents are counted rather than the base's being inherited |
| `--on-conflict {latest,first,error,keep-all}` | `latest` | two contributions for one cell key |
| `--inconclusive-rule {mad-overlap,none}` | `mad-overlap` | see section 5.4 |
| `--pretty` / `--compact` | `--pretty` | `indent=2`, `sort_keys=True`, so merged files diff cleanly |
| `-v` / `--verbose` | off | repeatable |

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | usage error |
| 2 | an input failed schema validation (without `--skip-invalid`), or referenced an op absent from `ops.toml` |
| 3 | conflicting contributions under `--on-conflict error` |
| 4 | no input files resolved |

### 4.3 `render.py` — Doxygen table, website markdown, coverage manifest

```
render.py [options] [MERGED]
```

`MERGED` defaults to `-` (stdin).

| Flag | Default | Meaning |
|---|---|---|
| `--format LIST` | `all` | repeatable; `doxygen`, `markdown`, `coverage`, `all` |
| `--out-dir DIR` | none | write the default filenames below |
| `--out PATH` | none | single-file output; `-` is stdout. Requires exactly one `--format`. Mutually exclusive with `--out-dir` |
| `--config ID` | every config in the file | repeatable |
| `--op OP` | every op | repeatable |
| `--scalar TAG` | every scalar | repeatable |
| `--baseline ARM` | from the merged file | |
| `--metric {gflops,time,ratio}` | `gflops` | |
| `--not-measured-token STR` | `n/a` | rendered text for a missing cell |
| `--title STR` | derived | document title |
| `-v` / `--verbose` | off | |

Exactly one of `--out-dir` and `--out` is required. Default filenames in `--out-dir`:

```
comparison_tables.dox   Doxygen page, Doxygen markdown tables, no raw HTML
comparison_tables.md    website markdown
coverage.md             coverage manifest, human readable
coverage.json           coverage manifest, machine readable
```

Exit: 0 success; 1 usage error; 2 the merged input failed to parse, or named an op absent from `ops.toml`.

### 4.4 `plots.py`

```
plots.py --out-dir DIR [options] [MERGED]
```

`MERGED` defaults to `-` (stdin). Matplotlib is forced to the `Agg` backend; no window is ever opened.

| Flag | Default | Meaning |
|---|---|---|
| `--out-dir DIR` | *required* | |
| `--format {svg,png,both}` | `svg` | |
| `--kind LIST` | `rate-vs-size` | repeatable; `rate-vs-size`, `ratio-vs-size`, `bar`, `roofline` |
| `--config ID`, `--op OP`, `--scalar TAG`, `--baseline ARM` | as `render.py` | repeatable |
| `--dpi N` | `150` | raster only |
| `--width IN` / `--height IN` | `7.0` / `4.5` | |
| `--log-x` / `--no-log-x` | `--log-x` | size axis |
| `-v` / `--verbose` | off | |

Output filenames: `<config_id>__<OP>__<scalar>__<kind>.<ext>`.

Exit: 0 success; 1 usage error; 2 unparseable input; 3 matplotlib unavailable.

---

## 5. The merged / normalised intermediate

`reduce.py` produces it; `render.py` and `plots.py` consume it and nothing else. They never read raw result files
and never read Google Benchmark JSON.

### 5.1 Shape

```jsonc
{
  "schema_version": "1.0.0",
  "kind": "eigen-benchmark-comparison-merged",
  "generated_utc": "2026-08-20T23:04:11Z",
  "ops_toml_sha256": "…",
  "reducer_version": "1.0.0",
  "baseline": "accelerate",

  "configs": {
    "<config_id>": {
      "machine_config_id": "m4pro",
      "isa_target": "aarch64-neon",
      "compiler": "AppleClang 17.0.0",
      "cxx_flags": ["-O3", "-DNDEBUG"],
      "eigen_commit": "…40 hex…",
      "eigen_commit_short": "e2a2fda17",
      "eigen_dirty": false,
      "threads": 1,
      "cpu_model": "Apple M4 Pro",
      "provenance_refs": ["<run_id>", "…"],
      "provenance_gaps": [ { "field": "/provenance/cpu/turbo_enabled", "reason": "…" } ],
      "notes": ["measured at a 1-minute load average of 3.76, above the 1.00 …"]
    }
  },

  "arms": {
    "eigen":      { "kind": "eigen",     "library_name": "Eigen",            "library_version": "…" },
    "accelerate": { "kind": "reference", "library_name": "Apple Accelerate", "library_version": "macOS 15.6" }
  },

  "cells": [ /* section 5.3 */ ],

  "coverage": { /* section 5.5 */ },
  "conflicts": [ /* section 5.6 */ ]
}
```

`provenance_gaps` and `notes` are the two caveat channels, and a caveat belongs
to exactly one of them. A gap says the run **could not establish** something
(`/provenance/cpu/turbo_enabled` on macOS). A note says the run **established
the condition and proceeded anyway** — the operator's own `--note`, a machine
profile marked `locally_verified = false`, or a load average above the
profile's `max_load_avg` under `--allow-noisy`. They render under separate
headings that make different claims, so writing one caveat to both states it
twice and the second statement is wrong.

Both are properties of the merged **set**, not of whichever contributing run
sorted first: `touch_config` unions them across every run of a configuration,
the same way `eigen_dirty` is OR'd. A configuration with no caveats carries
`[]`, never `null` — absence of caveats is a positive statement, not an unknown.

### 5.2 `config_id`

The merge key, deterministic and readable, `__` separated, lowercase:

```
<machine_config_id>__<isa_target>__<compiler_id_lower><compiler_major>__<eigen_commit_short>__t<threads>
e.g. m4pro__aarch64-neon__appleclang17__e2a2fda17__t1
```

Two result files merge into the same config exactly when this string matches. A differing compiler *minor* version
does not split a config *(arbitrary, fixed — minor bumps rarely move dense kernels, and splitting on them
fragments the store)*; the full compiler version is retained in `configs.<id>.provenance_refs` for audit. Anything
else that differs — machine, ISA, Eigen commit, thread count — splits the config, because comparing across those is
exactly the mistake this harness exists to prevent.

### 5.3 A cell

One entry per `(config_id, op, scalar, shape, threads)`, carrying every arm side by side:

```jsonc
{
  "config_id": "m4pro__aarch64-neon__appleclang17__e2a2fda17__t1",
  "op": "GEMM",
  "op_family": "blas3",
  "scalar": "f64",
  "shape": { "m": 1024, "n": 1024, "k": 1024 },
  "shape_dims": ["m", "n", "k"],
  "shape_group": "medium",
  "size_key": 1024,                    // the plot x-coordinate; see below
  "flops_per_iteration": 2147483648.0,
  "flops_nominal": false,
  "arms": {
    "eigen":      { "state": "measured",
                    "gflops": 118.4, "gflops_mad": 0.9,
                    "time_s": 0.01814, "time_mad_s": 0.00014,
                    "reps": 10, "cv": 0.011, "run_id": "…" },
    "accelerate": { "state": "not_measured",
                    "reason": "reference_routine_absent",
                    "detail": "Accelerate exposes no ?gesdd in the legacy BLAS surface" }
  },
  "ratio": 1.03,
  "ratio_state": "ok"
}
```

- `gflops` is `flop_rate / 1e9`. **This is the only place the division by 1e9 happens.** Raw result files hold
  flops per second (section 1.6); everything downstream of `reduce.py` is in GFLOP/s.
- `time_s` and `time_mad_s` are wall clock (`real_time`), in seconds.
- `size_key` is the plot x-coordinate, defined once so every plot agrees: the product of the shape's values raised
  to `1/len(shape)` — the geometric mean of the dimensions — rounded to the nearest integer. For a square GEMM it is
  the order; for `10000x8x8` it is `86`. *(arbitrary, fixed)* Renderers that want a different abscissa read `shape`.
- `ratio` is `arms.eigen.gflops / arms.<baseline>.gflops`, so `> 1` means Eigen is faster. `null` unless both arms
  are `measured`.

### 5.4 `ratio_state`

| Value | Meaning |
|---|---|
| `ok` | both arms measured, and the difference exceeds the dispersion |
| `inconclusive` | both arms measured, but the intervals `[median − MAD, median + MAD]` overlap. `.agents/benchmarking.md` rule 5: a change smaller than run-to-run variation is not a win or a regression. Renderers show the ratio greyed and never colour it |
| `not_measured` | at least one arm is not `measured`; `ratio` is `null` |
| `no_reference_equivalent` | `ops.<OP>.reference.kind == "none"`; there is no comparison to make, and the table says so rather than implying a missing measurement |

With `--inconclusive-rule none`, `inconclusive` is never produced.

### 5.5 `coverage`

```jsonc
"coverage": {
  "configs": ["…"],
  "ops": { "GEMM": { "measured": 200, "not_measured": 0, "unaccounted": 0, "arms": ["eigen", "accelerate"] } },
  "scalars": { "f64": { "measured": 200, "not_measured": 0 } },
  "totals": { "measured": 200, "not_measured": 0, "unaccounted": 0 },
  "missing_configs": [ { "machine_config_id": "…", "op": "GEMM", "reason": "machine_unavailable" } ]
}
```

`unaccounted` counts cells inside a contribution's `scope` that appeared in neither `measurements` nor
`not_measured`. It is always a bug; a non-zero value must be surfaced, not smoothed over.

### 5.6 `conflicts`

Two contributions producing different numbers for the same cell key:

```jsonc
{ "config_id": "…", "op": "GEMM", "scalar": "f64", "shape": {…}, "arm": "eigen",
  "kept": { "run_id": "…", "gflops": 118.4, "timestamp_utc": "…" },
  "dropped": [ { "run_id": "…", "gflops": 112.1, "timestamp_utc": "…" } ],
  "policy": "latest" }
```

`latest` keeps the newest `provenance.timestamp_utc`. Conflicts are always recorded even when resolved silently,
because a config whose numbers move between runs is information about the machine.

---

## 6. The "not measured" representation, end to end

A combination that produced no timing is stated as such at every stage. It is never zero, never `null` in a numeric
column, and never omitted.

| Stage | Representation |
|---|---|
| C++ | the arm simply does not register; a reference arm absent from the build produces no names |
| `run.py` | diffs the planned cell set (from `scope`) against the names the binary emitted, and writes a `not_measured` entry with a `reason` for every difference |
| result file | `not_measured[]` — `{op, arm, scalar, shape, threads, reason, detail}`, schema `$defs/not_measured_entry` |
| merged intermediate | `cells[].arms.<arm>.state = "not_measured"` with `reason` and `detail`; `ratio` `null`; `ratio_state` `not_measured` or `no_reference_equivalent` |
| Doxygen / markdown | the cell prints `--not-measured-token` (default `n/a`) with a superscript footnote marker; one footnote per distinct reason on the page, text taken from `detail`. For `no_reference_equivalent` the token is an em dash `—` and the footnote is `ops.<OP>.reference.reason` verbatim, so the table states that the comparison *cannot* exist |
| `coverage.md` / `coverage.json` | every not-measured combination is listed with its reason; this is the manifest that makes a partial dataset honest |
| plots | the point is a gap (`float("nan")`), never `0`. A series with any gap gets `(partial)` appended to its legend entry, and the axes are never rescaled as though the gap were a zero |

`reason` values are the enum in `result_schema.json` `$defs/not_measured_entry/properties/reason`. The distinction
that matters most to a reader is `no_reference_equivalent` (the operation has no counterpart — a permanent fact
about the APIs) versus `not_implemented` (nobody has written the benchmark yet — a temporary fact about us).

---

## 7. Machine config contract — `machines/<id>.toml`

`run.py` reads it; nothing else does. Required keys:

```toml
schema_version = "1.0.0"
id = "m4pro"                     # must equal the filename stem, matches ^[a-z0-9][a-z0-9._-]*$
display_name = "Apple M4 Pro (14-core)"
cpu_model = "Apple M4 Pro"
arch = "arm64"
isa_targets = ["aarch64-neon"]   # targets this machine can build and run
default_isa_target = "aarch64-neon"
sockets = 1
cores_per_socket = 14
threads_per_core = 1
smt_enabled = false
numa_nodes = 1
max_load_avg = 1.0               # run.py refuses above this without --allow-noisy

[memory]
benchmark_budget_bytes = 4294967296   # optional; omitted means no ceiling is enforced

[frequency]
governor = ""                    # "" means the platform exposes none; run.py emits a provenance_gaps entry
pinned = false

[isa."aarch64-neon"]
flags = []                            # compiler options that select the instruction set
cmake_options = ["-DEIGEN_BENCH_ISA_TARGET=aarch64-neon"]
notes = ""                            # optional; travels to the page as a run note

[arms.accelerate]
cmake_options = ["-DEIGEN_BENCH_REFERENCE=accelerate"]
version_fallback = "macOS 15.6 (Accelerate)"
thread_env = { VECLIB_MAXIMUM_THREADS = "{threads}" }
```

`{threads}` in a `thread_env` value is substituted with the run's thread count, so a multithreaded run does not
silently measure a single-threaded reference. A literal (`"1"`) is passed verbatim, and is the right spelling
only for a library that must stay sequential regardless of the run.

An ISA target's **`flags` reach the compiler** as a single `-DCMAKE_CXX_FLAGS`; they are what select the
instruction set, and an opt-in backend such as Eigen's SME needs its `-D…` here rather than in
`cmake_options`. For that reason a target may **not** set `CMAKE_CXX_FLAGS` through `cmake_options` as well —
`cmake_options` is appended last and would overwrite `flags`, leaving the run recording compile options its
binary was never built with. `parse_machine_profile` refuses that combination when the profile loads.

An ISA target's optional **`notes`** is prose about what measuring under that target does and does not mean —
"only GEMM has an SME kernel today, so every other operation here is the same code as the NEON target". It
travels through `provenance.run.notes` to the published page, because a caveat that lives only in a TOML
comment is a caveat the reader never sees.

**`[memory].benchmark_budget_bytes`** is optional and bounds one benchmark's operands. It is enforced inside
the binary before the first allocation, so a point too large for the machine becomes a single `out_of_memory`
cell instead of an allocation failure that loses every cell already measured in that invocation. Because the
footprint scales with `sizeof(Scalar)`, one budget yields a different ceiling per scalar without the grid
having to know about scalars. It bounds memory only — run time is bounded by choosing `--scalars` and
`--groups`. Omit it and nothing is enforced.

Every nullable `provenance` field that this file leaves empty obliges `run.py` to write a matching
`provenance_gaps` entry (section 0 of `result_schema.json`). On this machine that is at minimum
`/provenance/cpu/frequency_governor` — macOS exposes no governor — and `/provenance/numa/policy`.

---

## 8. REUSE metadata

`ops.toml` and every `.h`, `.cpp`, `.cmake`, `CMakeLists.txt`, `.py`, and `.sh` file in this tree carries the
two-line inline header in that language's comment syntax:

```
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0
```

`.json` cannot carry a comment and `.md` headers leak into rendered output, so `result_schema.json`, this file, and
every generated `.md` must instead be covered by an annotation block in the repository-root `REUSE.toml`. Only the
agent that owns `REUSE.toml` may add it.

---

## 9. What the build publishes to `run.py`

Section 3.3 fixes what the *binary* reports. This section fixes what *CMake* decides, which `run.py` must read
rather than re-derive.

### 9.1 Selecting the reference

`-DEIGEN_BENCH_REFERENCE=<key>` and `-DEIGEN_BENCH_BLAS_VENDOR=<key>` are aliases; setting both to different
values is a fatal configure error. `auto` searches the vendor table, `none` disables the reference arm, and a
`BLA_VENDOR` spelling (`Apple`, `FLAME`, `Generic`, ...) resolves to its arm key. With no reference configured the
Eigen arm still builds and registers — a developer without a BLAS is never blocked. *(arbitrary, fixed)*

The vendor table lives in `vendors.cmake`, one row per arm. `eigenblas` is deliberately absent: Eigen's own
`blas/` shim exports the Fortran ABI and no `cblas.h`, so it needs an arm decision in the C++ layer rather than
another table row.

Adding a vendor is one `eigen_bench_declare_vendor()` call and nothing else, including for `auto`: the candidate
order is **derived from the registry**, ranked by each row's optional `AUTO_PRIORITY` (lower first, default 500)
and then declaration order. `EIGEN_BENCH_VENDOR_AUTO_ORDER` remains as an override that is tried first, and every
declared key it omits is still tried after it — a hand-maintained list of keys would silently leave a new row out
of auto-detection, and being a CACHE entry it would also stay stale in an already-configured build tree.

Two selection rules exist to stop a measurement being attributed to the wrong library:

- A candidate is linked against the `BLAS_LIBRARIES` **that candidate's** `find_package(BLAS)` returned.
  `FindBLAS` creates `BLAS::BLAS` on its first success and never revises it, so the shared target is adopted only
  when it still matches; otherwise the libraries are linked directly, with `BLAS_LINKER_FLAGS` restored by hand.
  Linking a rejected earlier candidate while every provenance field named the selected one is undetectable
  downstream. *(reproduced before the guard existed.)*
- A missing CBLAS header is **never** a reason to reject or skip a vendor. The comparison arm calls the Fortran
  BLAS directly and includes no vendor header (section 9.2); only `Tuning/bench_blas_gemm.cpp` needs one, and that
  target is skipped when it is absent. Gating selection on it refused usable installations — NVPL resolved through
  its config package, any module/spack/conda MKL whose headers ship separately — for a header nothing in
  `benchmarks/comparison/` includes.

### 9.2 The macros CMake defines

`EIGEN_BENCH_REFERENCE_ARM=<key>` is a **bare token** (section 3.2) and its absence is what compiles the reference
arm out. Everything else is a C string literal: `EIGEN_BENCH_REFERENCE_LIBRARY_NAME`,
`EIGEN_BENCH_REFERENCE_VERSION_FALLBACK`, `EIGEN_BENCH_REFERENCE_PATH` (the `reference_library_path` context key),
`EIGEN_BENCH_REFERENCE_INTERFACE`, `EIGEN_BENCH_REFERENCE_THREADING`, `EIGEN_BENCH_EIGEN_COMMIT`,
`EIGEN_BENCH_EIGEN_DIRTY`, `EIGEN_BENCH_COMPILER_ID`, `EIGEN_BENCH_COMPILER_VERSION`, `EIGEN_BENCH_CXX_STANDARD`,
`EIGEN_BENCH_CXX_FLAGS`, `EIGEN_BENCH_ISA_TARGET`, `EIGEN_BENCH_OPS_TOML_SHA256`.

`EIGEN_BENCH_THREAD_ENV_VARS` is **not** currently defined by CMake: `bench_compare.h` carries its own
fallback list, which must be kept in step with `run.py`'s `THREAD_COUNT_ENV_VARS` by hand. The data to
generate it already exists as the per-vendor `THREAD_ENV` field in `vendors.cmake`; wiring that up would
make the two lists one.

Two are not string literals and both are safety-critical:

- `EIGEN_BENCH_REFERENCE_ILP64` is defined iff the vendor uses 64-bit BLAS integers. `eigen_bench::BlasInt` is an alias for
  `Eigen::BlasIndex`, so the width comes from Eigen's own `EIGEN_64BIT_BLAS` rather than a private switch.
  CMake does not yet define `EIGEN_64BIT_BLAS` alongside this macro; a `static_assert` in `bench_compare.h`
  turns that disagreement into a build error rather than silent argument corruption. It is derived from the table row that
  selected the library, never assumed.
- `EIGEN_BENCH_HAVE_<UPPERCASED_SYMBOL>` (`..._OPENBLAS_GET_CONFIG`, `..._MKL_GET_VERSION_STRING`,
  `..._BLI_INFO_GET_VERSION_STR`, `..._ARMPLVERSION`, `..._NVPL_BLAS_GET_VERSION`) is defined only after a check
  that compiles **and links** the symbol. Defining one on a guess turns a missing version query into a link
  failure. `accelerate` and `netlib` have no query and fall through to `EIGEN_BENCH_REFERENCE_VERSION_FALLBACK`.

The reference arm calls the Fortran BLAS (`dgemm_`) directly, so the comparison targets need no `cblas.h`. The
table still resolves a CBLAS header per vendor because `Tuning/bench_blas_gemm.cpp` includes one, and reports it in
`vendor_info.json`; it is advisory, never a gate on vendor selection (section 9.1). On macOS that header is inside
the framework (`vecLib.framework/Headers/cblas.h`), not on the default include path;
`cmake/FindAccelerate.cmake` is a different module, for the Accelerate *sparse* backend, and cannot be used here.

### 9.3 `vendor_info.json`

Configuring writes `<build-dir>/comparison/vendor_info.json` (`vendor_info-$<CONFIG>.json` under a multi-config
generator) with `schema_version` `"1.0.0"`: the build's own record of what it decided, for a reader or a future
consumer. `run.py` does **not** read it today — it re-derives the same facts from `CMakeCache.txt`,
`compile_commands.json`, the binary's own context keys, and a search for the executable — so the two can in
principle disagree and nothing cross-checks them. It
carries `build_target` (`"bench_comparison_all"`, the aggregate target — `bench_comparison` would collide with an
executable named after a source stem), the compiler id/version/path, `cxx_standard`, `cxx_flags`, `isa_target`,
`eigen_commit`, `eigen_dirty`, `ops_toml_sha256`, the `arms` list, a `reference` block (`available`, `arm`,
`library_name`, `library_version`, `version_source`, `library_path`, `libraries`, `include_dirs`, `interface`,
`threading_model`, `thread_env`, `provides`, `notes`), and one `targets` entry per built binary giving its
`target`, absolute `executable`, and the `ops` it carries.

One executable is built per `bench_*.cpp` in this directory, named after the source stem — the same rule `run.py`
applies to an op's `source` key, so several ops may share one binary. An op marked `implemented` whose source is
absent is a configure-time warning naming the target `run.py` will fail to find. *(arbitrary, fixed)*
