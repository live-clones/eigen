# MR !2903 comparison harness — fleet check

Date: 2026-08-22. Branch `benchmark-comparison-harness` @ `9e6620ec5` (== `origin`, == MR head).

Question asked: do the benchmarks in !2903 work on the machines we have access to?

## Summary

| Host | Arch / ISA | Reference BLAS | Harness status |
|---|---|---|---|
| local M4 | arm64 macOS, NEON + SME | Accelerate | already verified in the MR |
| AVX-512 Zen 4 box | x86_64, AVX-512 | OpenBLAS 0.3.32 | **verified**, full 724-cell run |
| 24-core Zen 2 box | x86_64, AVX2 | OpenBLAS 0.3.26 + "netlib" | **verified**; exposed the netlib/alternatives bug |
| 72-core Neoverse V2 | aarch64, NEON + SVE2 | OpenBLAS 0.3.26 | **verified**, both ISA targets, full pipeline |
| 20-core Cortex-X925 | aarch64, NEON + SVE2 | none installable (no sudo) | **verified** Eigen-only / no-reference path |
| 2 busy Zen 2 boxes | x86_64, AVX2 | OpenBLAS + netlib | not run: sustained load 8-10, would fail the noise guard legitimately |
| Arm embedded board | aarch64 NEON | none, no sudo, 19 GB disk | not attempted (see below) |

Verdict: **the harness works on every machine it could be run on** — three new machines across two
architectures, four new ISA/arm configurations, and the full `run -> reduce -> render` pipeline.
Three real bugs surfaced, described below; none of them stops it working, all three affect what a
published page would say.

## Portability facts worth keeping

- **`run.py` needs only the standard library plus `tomllib`, i.e. Python >= 3.11.** `jsonschema` is a
  deferred import used for schema validation; `matplotlib` is needed only by `plots.py`. Every Linux
  host in the fleet has Python 3.12+ and `jsonschema`; only two have `matplotlib`, and only two have
  `pytest`. So the measurement path runs everywhere, but `plots.py` and the pytest suite do not.
  This is a good property and worth stating in the harness README: a measuring host needs far less
  than a developing host.
- `run.py` configures `benchmarks/` directly, not the whole Eigen tree, so a measuring host needs no
  full Eigen configure and no test dependencies.
- Google Benchmark is pulled by FetchContent, so a measuring host needs outbound GitHub access on the
  first configure. Every Linux host in the fleet has it.
- `ninja` is absent on the 72-core Arm server; the machine profile's `[build].generator` field already
  covers this (set it to `Unix Makefiles` there), so no code change is needed. Good sign for the
  "adding a machine is a data change" design goal.

## Confirmed working on the AVX-512 x86 box

The `zen4-7800x3d` profile — shipped in the MR as `locally_verified = false`, never run — worked
**unmodified** on the machine it describes. Nothing in the profile had to be corrected:

- CPU model probed matches the profile string exactly; `cores_per_socket = 8`, `threads_per_core = 2`,
  `smt_enabled = true`, `numa_nodes = 1` all confirmed.
- `[frequency].governor = "performance"` confirmed by probe.
- `[pinning] tool = "taskset"`, `cpu_list = "0-7"` — the assumption written in the profile's `notes`
  (logical CPUs 0-7 are one thread of each core) is **correct** on this part: the probe reports
  L1d/L2 `num_sharing = 2` and L3 `num_sharing = 16` over 8 cores / 16 threads.
- `x86-64-avx512` ISA target configures and builds with gcc 15.2.
- `find_package(BLAS)` with `BLA_VENDOR OpenBLAS` resolves, and the runtime version query works —
  the result file carries `OpenBLAS 0.3.32 NO_LAPACKE DYNAMIC_ARCH NO_AFFINITY Cooperlake
  MAX_THREADS=128`, i.e. the `version_fallback` in the profile was correctly *not* used.
  Note the vendor row declares `PROVIDES ... lapacke` while this build is `NO_LAPACKE`; harmless
  today because no measured op needs LAPACKE, but the declaration is not true of this build.
- Both arms measured, validation passed, schema-valid result file written.

The load-average guard also demonstrably works: a first attempt was **refused** at 0.87 > the
profile's 0.5 (the load was my own `git clone` decaying), with a correct actionable message.

## BUG: the `--allow-noisy` caveat is asserted on runs that never passed `--allow-noisy`

**This is the one substantive finding.** It sits inside the MR's own "honesty machinery", so it is
worth fixing before publication rather than after.

`run.py:1744` builds the noise caveat from the **peak** of the before- and after-run load averages:

```python
peak_load = max((value[0] for value in (inputs.load_avg_before, inputs.load_avg_after) if value), default=None)
if peak_load is not None and peak_load > machine.max_load_avg:
    note = (... "measurements and the run was allowed to proceed with --allow-noisy")
```

but the **guard** at `run.py:2356` consults only the *before* value:

```python
if host.load_avg and host.load_avg[0] > machine.max_load_avg and not args.allow_noisy:
```

Two consequences, both bad on a published page:

1. **The note states something false.** On the observed run: `load_avg_before = 0.37` (guard passed
   cleanly, no flag needed), `load_avg_after = 0.81`, `max_load_avg = 0.5`. The note fired and said
   *"the run was allowed to proceed with --allow-noisy"* — while `provenance.harness.argv` in the very
   same document is `["--machine","zen4-7800x3d","--arms","openblas","--ops","GEMM","--scalars","f64",
   "--groups","small","--repetitions","3","--min-time","0.05s","-j","16"]`, with no such flag. The
   result file contradicts itself, and it does so by attributing an override to the operator.

2. **The trigger is self-inflicted, so the caveat fires on clean runs.** `load_avg_after` is sampled
   immediately after the harness's own `cmake --build -j16` and its own benchmark process. A 1-minute
   load average taken at that moment reflects the harness, not competing work. On a strict profile
   (`max_load_avg = 0.5`) essentially *every* run on an otherwise idle machine will carry a
   "the machine was noisy" caveat — which devalues the caveat exactly where it is supposed to carry
   weight, and is the opposite of the `test_a_load_average_above_the_profile_is_stated_with_the_numbers`
   docstring's intent ("the unbreached case stays quiet, or the note is decoration rather than a
   caveat").

**Why the test suite does not catch it.** `tests/test_run.py:804` forces `max_load_avg = 0.0` so the
note always fires, and its `stub_run` helper (`tests/test_run.py:498`) *always* passes `--allow-noisy`.
So no test ever exercises the combination that occurs in practice — guard not breached, flag not
passed, note fires anyway. A regression test needs a profile whose threshold sits between the before
and after readings, and must assert that the note's wording is reachable only when
`args.allow_noisy` is true.

**Suggested fix.** Separate the two facts the note currently conflates:
- whether the *threshold was breached at guard time* (that is what `--allow-noisy` gates), and
- what the load actually was (already recorded verbatim in `run.load_avg_before` / `load_avg_after`).

i.e. gate the note on `args.allow_noisy` and phrase the after-run reading, if it is reported at all,
as the harness's own load rather than as competing work.

### The `--allow-noisy` bug reproduces on three machines

Confirmed independently on three hosts of two architectures, every one of them with the guard
passing cleanly and no flag on the command line:

| Host | `load_avg_before` | `load_avg_after` | `max_load_avg` | `--allow-noisy` in argv? | note emitted? |
|---|---|---|---|---|---|
| AVX-512 Zen 4 | 0.37 | 0.81 | 0.5 | no | yes, falsely |
| 72-core Neoverse V2 | **0.11** | 0.65 | 0.5 | no | yes, falsely |
| 20-core Cortex-X925 | 0.25 | 0.57 | 0.5 | no | yes, falsely |

The Neoverse V2 row is the clearest: a 1-minute load average of 0.11 on an idle 72-core server is
about as quiet as a machine ever gets, and the run is still stamped "competing work may have
displaced these measurements and the run was allowed to proceed with --allow-noisy".

## BUG: `run.py` refuses to run a second time, because of a CWD-dependent path

`run.py` has a guard that refuses to measure from a dirty worktree. It also has a helper,
`output_pathspecs` (`run.py:1320`), whose docstring anticipates this exact failure:

> ... Eigen's .gitignore covers `*.build*` but not `build*/`, so without this the harness's own
> default --build-dir sits in the worktree as untracked and trips the guard on the very first run.
> The results directory is the same story one step later: **the first successful measurement made the
> second one refuse.**

The helper does not work, because the build directory is resolved against two different bases:

```python
# run.py:2342 -- the exclusion pathspec, args.build_dir resolved against the CWD
git_facts = probe_git(root, Path(args.build_dir), Path(args.results_dir))

# run.py:2469 -- where the build directory is ACTUALLY created, resolved against the repo root
build_dir = make_build_dir(Path(args.build_dir), isa_target, arm_key or "eigen")
if not build_dir.is_absolute():
    build_dir = root / build_dir
```

`--build-dir` defaults to the **relative** string `"build-comparison"`. `output_pathspecs` calls
`.resolve()` on it, which resolves against the process's CWD, so it excludes
`<cwd>/build-comparison` while the build is created at `<root>/build-comparison`. Whenever
CWD != repo root the exclusion misses and the guard trips. That is the normal case, since `run.py`
lives in `benchmarks/comparison/` and that is where you naturally invoke it from.

`--results-dir` is unaffected: its default is *absolute* (`str(comparison_dir / "results")`), and
`benchmarks/comparison/results/` is additionally in `.gitignore`. Only `--build-dir` is relative.

Isolated on the AVX-512 host, three real runs differing only in CWD and `--build-dir` spelling:

| | invocation | result |
|---|---|---|
| A | from `benchmarks/comparison/`, default `--build-dir` | **refused**: "the Eigen worktree has uncommitted changes" |
| B | from the repo root, same default `--build-dir` | wrote the result file |
| C | from `benchmarks/comparison/`, **absolute** `--build-dir` | wrote the result file |

`git status --porcelain --untracked-files=no` was empty throughout, i.e. the worktree was clean by
the definition `CMakeLists.txt` uses for `EIGEN_BENCH_EIGEN_DIRTY`. So the C++ side recorded
`dirty: false` for exactly the tree the Python side called dirty — the two dirty definitions
disagree as well.

Note `--dry-run` returns *before* the dirty check, so no dry run reveals this.

**Fix** — one line, mirroring what `run.py:2469-2471` already does:

```python
build_dir_arg = Path(args.build_dir)
if not build_dir_arg.is_absolute():
    build_dir_arg = root / build_dir_arg
git_facts = probe_git(root, build_dir_arg, Path(args.results_dir))
```

**The fix is verified.** Applied on the Zen 2 host and committed (so the patch itself did not make
the tree dirty — which it does if you test it uncommitted, and is worth knowing before you conclude
the fix failed):

- from `benchmarks/comparison/` with the default relative `--build-dir`: **now succeeds**, where it
  was refused before;
- a modified tracked file (`echo // scratch >> Eigen/Core`): **still refused**;
- an untracked stray `benchmarks/comparison/stray.cpp`: **still refused**.

So the guard keeps exactly the coverage its docstring claims, and stops firing on the harness's own
output. `output_pathspecs` itself needed no change — it was always correct, it was just being handed
a path resolved against the wrong base.

**Workaround until then:** invoke `run.py` from the repository root, or pass an absolute
`--build-dir`. Do *not* reach for `--allow-dirty`, which is what the guard's docstring warns against.

## BUG: the `netlib` arm silently measures OpenBLAS on Debian/Ubuntu

The most consequential finding, because it produces a plausible published number that is wrong about
which library it measured.

The `netlib` vendor row uses `BLA_VENDOR Generic`, which finds `/usr/lib/<triplet>/libblas.so`. On
Debian and Ubuntu that path is not netlib — it is an **update-alternatives symlink**, and installing
`libopenblas-dev` repoints it at OpenBLAS. Checked on all three Debian/Ubuntu hosts tried; all three
identical:

```
/usr/lib/<triplet>/libblas.so   -> /usr/lib/<triplet>/openblas-pthread/libblas.so.3
/usr/lib/<triplet>/liblapack.so -> /usr/lib/<triplet>/openblas-pthread/liblapack.so.3
update-alternatives --query libblas.so-<triplet>:  Value: .../openblas-pthread/libblas.so
```

Confirmed end to end by building and running the `netlib` arm. What the result file says:

```json
"netlib": {
  "kind": "reference",
  "library_name": "Netlib reference BLAS",
  "library_path": "/usr/lib/x86_64-linux-gnu/libblas.so",
  "library_version": "unknown"
}
```

What `ldd` says about the very binary that produced those numbers:

```
libblas.so.3    => /lib/x86_64-linux-gnu/libblas.so.3
libopenblas.so.0 => /lib/x86_64-linux-gnu/libopenblas.so.0
```

Nothing in the harness can catch it:

- `library_path` is recorded as the alternatives symlink and never resolved, so it does not name the
  real vendor.
- `netlib` is the one vendor row in `vendors.cmake` with **no `VERSION_RUNTIME_SYMBOL`** (every other
  vendor has one: `openblas_get_config`, `mkl_get_version_string`, `bli_info_get_version_str`,
  `armplversion`, `nvpl_blas_get_version`). It falls back to `VERSION_PKGCONFIG cblas`, which was not
  present, so `library_version` is `"unknown"` — the only honest signal, and a weak one.

Why it matters more than a mislabelled column usually would: the `netlib` arm is *documented* as the
floor. `zen4-7800x3d.toml` says of it — "The unoptimised reference implementation. Useful as a floor
and as a correctness oracle, not as a competitive arm; a renderer should not read 'Eigen is 30x
netlib' as a tuning result." A reader who sees Eigen at ~1.1x a column headed "Netlib reference
BLAS/LAPACK" concludes Eigen is barely faster than unoptimised reference code, when the column is in
fact a tuned OpenBLAS. The error flatters nobody and misleads specifically.

**Two cheap fixes, both in the harness's existing vocabulary:**

1. Record the **resolved** library path (`realpath`) next to the declared one. The resolved target
   `.../openblas-pthread/libblas.so.3` names the vendor outright, and the reducer/renderer already
   have a place to put a caveat.
2. Give `netlib` a **negative** runtime check: link-check the *other* vendors' version symbols
   against the selected library and refuse (or record a `run.notes` caveat) when one resolves. A
   generic BLAS that answers `openblas_get_config()` is not netlib. This is the same
   `check_cxx_source_compiles` machinery `vendors.cmake` already runs for the positive case.

## Google Benchmark: `REQUIRED` but unversioned, and not fetched

`benchmarks/CMakeLists.txt:36` is `find_package(benchmark REQUIRED)` — **no version constraint and no
FetchContent fallback**. So:

- A measuring host must have Google Benchmark installed already; the harness will not fetch it. Worth
  saying out loud in the harness README, because it is the one non-obvious prerequisite (the rest is
  a compiler, CMake and a BLAS).
- CONTRACTS.md pins its guarantees to a version the build does not require: the name grammar
  (§1.2), the counter behaviour (§204), `AddCustomContext` (§393) and the custom-`main` interaction
  (§461) are each "verified against 1.9.5", but nothing stops the build resolving 1.6.0.

**Tested:** Ubuntu 24.04 ships **1.8.3**, and the harness works correctly against it — GEMM
registered, both arms measured, validation passed, schema-valid file written. So 1.8.3 is a
sufficient floor in practice; it is just not declared. Suggest `find_package(benchmark 1.8 REQUIRED)`
(or whatever is verified) so the failure is a clear CMake message rather than a subtle behaviour
difference in the JSON.

## `cpu.model` is silently unverifiable on aarch64 Linux

`run.py:1532` is `cpu_model = host.cpu_model or machine.cpu_model`, and `probe_host()` reads the
model from `/proc/cpuinfo`'s `model name` / `cpu model` line (`run.py:~1244`).

**aarch64 Linux `/proc/cpuinfo` has no such line.** On the 72-core Arm server it carries only:

```
CPU implementer : 0x41
CPU part        : 0xd4f
```

so `host.cpu_model` is `None` and the profile's declared string is published as `provenance.cpu.model`
with **no `provenance_gaps` entry** saying it was never confirmed, and the `cpu_model` mismatch
warning at `run.py:2366` can never fire on the whole architecture family. macOS is unaffected
(`sysctl machdep.cpu.brand_string`), which is why the M4 run did not expose it.

Note the inconsistency: `frequency_governor` immediately below uses the identical `host.X or
machine.X` shape and *does* record a gap when the value could not be established. `cpu.vendor`
records a gap too. `cpu.model` is the one that does not.

Fix is either direction — record a gap when `host.cpu_model is None`, or probe it properly on
aarch64 (`lscpu` reports `Model name: Neoverse-V2`; or decode implementer/part from the MIDR).

## Smaller observations

- **`max_load_avg` is hard to satisfy in practice, because the harness's own work trips it.** The
  guard samples the load *before* the run, then `run.py` does a `cmake --build --parallel N` and
  measures immediately afterwards. Back-to-back invocations are routinely refused because the
  previous invocation's build is still in the 1-minute average — I hit this on two separate hosts and
  had to poll `/proc/loadavg` and wait ~2 minutes between runs. Worth either sampling after the build
  and before the measurement, or documenting the wait.
- **`provenance.hostname` is the only place a host name appears, and it never reaches rendered
  output** — `render.py`, `reduce.py` and `plots.py` contain no reference to it, and no golden file
  carries one. Good: result files can come from named internal machines without the name reaching a
  published page. (Result files are gitignored anyway.)
- **`PROVIDES ... lapacke` is wrong for the packaged OpenBLAS.** Both distro builds encountered report
  `NO_LAPACKE` in `openblas_get_config()` (0.3.32 on the AVX-512 host, 0.3.26 on the others), while
  `vendors.cmake`'s openblas row declares `PROVIDES blas cblas lapack lapacke`. Harmless today —
  nothing measured needs LAPACKE — but the declaration will be false the moment an op does, and
  `EIGEN_BENCH_REFERENCE_FAMILIES` is derived from it.
- **`run.py` needs only the standard library plus `tomllib` (Python >= 3.11).** `jsonschema` is a
  deferred import, `matplotlib` is needed only by `plots.py`. Every Linux host in the fleet satisfies
  the measuring requirement; only two have `matplotlib` and only two have `pytest`. A measuring host
  needs much less than a developing host — worth stating.
- **The "adding a machine is a data change" claim holds.** Two new profiles were added for hosts of a
  kind the harness had never seen (a 24-core Zen 2 x86 part, and a 72-core aarch64 server with 9 NUMA
  nodes, no ninja, and SVE2). Neither needed any change to Python, CMake or C++. The pytest suite even
  picked them up automatically: 352 -> 354 passing, the two extra being the per-profile parametrised
  tests.
- **`[build].generator` earns its keep.** The 72-core Arm server has no ninja; setting
  `generator = "Unix Makefiles"` in the profile was the entire fix.
- The `numactl` pinning path in `plan_pinning` had no profile exercising it (both shipped profiles use
  `taskset` or `none`). The new Arm profile uses it, so that branch is now covered by a real run.

## Eigen finding (not a harness bug): the SVE backend is ~2x slower than NEON

Surfaced by the harness on the 72-core Neoverse V2, and worth pursuing independently of this MR.

`EIGEN_ARM64_USE_SVE` with `-march=armv9-a+sve2 -msve-vector-bits=128` is **slower than the default
NEON path on every operation measured**, most dramatically for GEMM. Because this part's SVE vector
length is 128 bits — the same width as NEON — this is a clean control: the two configurations differ
in code generation only, not in vector width, so the gap cannot be explained by width.

`double`, single-threaded, pinned to node 0, same binary options otherwise (GFLOP/s):

| op | size | NEON | SVE | SVE / NEON |
|---|---|---|---|---|
| GEMM | 128^3 | 39.06 | 18.86 | **0.48x** |
| GEMM | 96^3 | 38.59 | 18.77 | 0.49x |
| GEMM | 64^3 | 36.04 | 18.03 | 0.50x |
| GEMV | m=n=128 | 19.82 | 15.48 | 0.78x |
| POTRF | n=128 | 15.31 | 11.07 | 0.72x |

Against the OpenBLAS arm the sign flips, which is what makes it publishable-relevant rather than a
curiosity: Eigen NEON is **x1.12** OpenBLAS at GEMM 128^3, Eigen SVE is **x0.54**. So the same
library on the same hardware either beats or loses to half of the reference depending on a build
macro, and the ISA target is exactly the axis this harness was built to expose.

Caveats: gcc 13.3, OpenBLAS 0.3.26, and the OpenBLAS arm here dispatches to its *Neoverse V1* kernels
on this V2 part (see the profile), so the OpenBLAS column is not OpenBLAS at its best either. The
NEON-vs-SVE comparison is unaffected by that — both Eigen columns face the same reference.

Numbers above are from the smoke configuration (2 repetitions, 0.05 s min-time, `small` group). A
full-fidelity run at default settings across all shape groups was launched to confirm; the ratios
were stable to the third digit across all six shapes measured, so the effect is far outside noise.

## What was not done, and why

- **The two busy Zen 2 boxes** were left alone: both were carrying a sustained load average of 8-10
  from other work. The harness would have refused them, correctly, and forcing it with
  `--allow-noisy` would have produced exactly the unreproducible numbers the guard exists to prevent.
  Nothing about them differs from the idle Zen 2 box that was verified, apart from the GPUs.
- **The Arm embedded board** was not attempted: no BLAS development package, no sudo to install one,
  no Google Benchmark, and 19 GB of free disk. It would have been a build check only, duplicating
  what the Cortex-X925 host already established for aarch64 Eigen-only builds, at a much higher cost
  (8 cores clamped to a 30 W power mode).
- **No LAPACK-specific vendor was exercised beyond OpenBLAS's built-in one.** oneMKL 2026.1 is
  installed on the AVX-512 host and its arm is declared in that profile, but it was not run; MKL on an
  AMD part dispatches conservatively, which the profile already documents, so it is a configuration
  question rather than a portability one.

## Reproducing

Each host has a clean clone of the MR branch on local (not network-mounted) disk, with its machine
profile committed as one local commit on top of the MR head so the worktree is clean and results are
not marked unreproducible.

**Invoke `run.py` from the repository root** until the `--build-dir` bug above is fixed:

```sh
cd ~/local/repos/eigen-bench
python3 benchmarks/comparison/run.py --machine <id> --arms openblas --ops all -j <N>
```

Between back-to-back runs, wait for the 1-minute load average to fall below the profile's
`max_load_avg` or the next run is refused:

```sh
while [ "$(echo "$(cut -d' ' -f1 /proc/loadavg) < 0.35" | bc -l)" != 1 ]; do sleep 10; done
```

The Cortex-X925 host needs `export CMAKE_PREFIX_PATH=$HOME/local/gbench` (Google Benchmark 1.9.5
built from source there, since that account cannot install the distro package).

## New machine profiles

Three written and verified against the hardware they name, all with `locally_verified = true`, all
using generic hardware descriptions with no host names or internal identifiers:

- `machines/zen2-3960x.toml` — 24C/48T Zen 2 x86, AVX2, OpenBLAS
- `machines/neoverse-v2-72c.toml` — 72-core aarch64, NEON + SVE2, OpenBLAS, `Unix Makefiles`
  generator, `numactl` pinning
- `machines/cortex-x925-20c.toml` — 20-core heterogeneous aarch64, NEON + SVE2, no reference arm,
  `taskset` pinned to the performance cluster

They are in the working tree only, not committed to the branch. `pytest` picks them up automatically
(352 -> 354 passing).

## Cross-fleet merge: the incremental/partial claims hold

Merged the result files from four machines, three architectures and three reference arms into one
dataset and rendered it, on the M4 (the only host here with matplotlib):

- 5 configurations, **114 measured arm cells, 0 not-measured, 0 unaccounted**.
- `reduce.py` correctly **refused** to pick a baseline when the input carried two reference arms
  (`netlib`, `openblas`) and asked for `--baseline`, rather than silently choosing.
- `render.py --baseline eigen` correctly refused on the Eigen-only dataset: *"was given but this
  document records no baseline, so it has no ratios to label"*.
- The Eigen-only machine merged in without producing a single zero or a dropped row, and its
  configuration appears in the coverage manifest with its own caveat block.
- `plots.py` produced 11 SVGs across all five configurations.

This is the strongest evidence for the MR's central design claim, and it holds: an aarch64 Eigen-only
run, an x86 OpenBLAS run and an x86 "netlib" run from three different Eigen commits merged into one
self-describing document.

**One consequence worth noting for the `--allow-noisy` bug:** the false caveat does not stay in the
result JSON. It propagates through `reduce.py` into the rendered `coverage.md`, where every one of
the five configurations now carries *"was measured under the following stated conditions: measured at
a 1-minute load average of ... and the run was allowed to proceed with --allow-noisy"*. That is the
published page saying the operator overrode the quiet-machine guard five times. None of them did.

## Follow-up: `zen4-7800x3d` ships without a memory budget

`[memory].benchmark_budget_bytes` is optional, and `apple-m4.toml` sets it with a careful measured
justification. `zen4-7800x3d.toml` sets none, so nothing bounds the operands — and the default
`--ops all` grid includes GEMM at `m:12288` and `m:16384`. A square `double` GEMM at n=16384 holds
3n² operands = 6.0 GiB, and `complex<double>` at the same shape holds 24 GiB.

Consequence for anyone picking that profile up: `run.py --machine zen4-7800x3d --ops all` runs for
hours and allocates in the tens of GiB, with no `out_of_memory` skips to stop it, where the same
command on `apple-m4` skips those cells explicitly. The two profiles in the tree behave very
differently for the same command line and nothing says so.

Worth setting a budget in that profile before anyone runs it unattended. The mechanism is already
there and already documented — it just is not used.

## Fixes committed

Three commits on `benchmark-comparison-harness`, each with a regression test verified to fail
against the unfixed code (the standard the branch already sets for itself):

| Commit | Fixes |
|---|---|
| `0d6979ca5` | the forged `--allow-noisy` caveat |
| `1711b8e5c` | `--build-dir` resolved against two different bases |
| `4204f1e19` | a "generic" reference BLAS that is really a named vendor |

Verification for each:

- **allow-noisy** — unit test on `assemble_provenance` with `load_avg_before` 0.37 and
  `load_avg_after` 0.81 against a 0.50 threshold. Fails against the unfixed code with the exact
  false note; passes after. The genuine breach case is asserted too, so the caveat is not simply
  suppressed.
- **build-dir** — two tests: one on the new `resolve_output_dir` helper, and one that monkeypatches
  `probe_git` and checks what `main()` actually passes it. Only the second catches a call site that
  resolves the path itself again, which is what the defect was; confirmed by reverting just the call
  site and watching that test fail while the helper test still passes.
- **generic vendor** — verified against the real affected configuration rather than a fixture, since
  that is the only place the behaviour exists: netlib against Debian's alternatives is now refused
  with the resolved path named; netlib pointed at a genuine library at its own path still
  configures; openblas is untouched; and Accelerate on macOS — which also has no runtime version
  query and so takes the same path — resolves through the SDK and configures as before, then builds.
  A structural test pins the coupling the check depends on.

Full suite: **360 passing** (352 on the MR head, +4 new tests, +4 from the three added machine
profiles being picked up automatically). `codespell` clean over the whole tree.

Not committed: the three machine profiles and this document, which are in the working tree only.
