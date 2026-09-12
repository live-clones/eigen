# Cross-library comparison benchmarks

Eigen against a BLAS/LAPACK library, head to head: one Google Benchmark binary per operation, each
carrying an Eigen arm and a library arm that run the same operation on the same operands in the same
process. The library's result is validated against Eigen's before it is timed, both arms report the
same nominal flop count (`benchmarks/bench_common.h`), and ratios are formed within one run. The
pages under <https://libeigen.gitlab.io/benchmarks/> are drawn from the result files this directory
produces.

Operations: `bench_gemm_compare.cpp` (GEMM), `bench_gemv_compare.cpp` (GEMV), `bench_potrf_compare.cpp`
(POTRF). Benchmark names are `OP/arm/scalar/dim:value/...`, for example `GEMM/openblas/f64/m:256/n:256/k:256`.

## Producing the data for a new machine

1. **Install the libraries** to compare against. Google Benchmark must be findable by CMake.

2. **Write a machine profile**, `machines/<id>.toml`, copying the closest existing one. The minimum:

   ```toml
   id = "zen5-ai-max-395"                # equals the file name
   display_name = "AMD Ryzen AI MAX+ 395 (16C/32T, Zen 5), WSL2"
   cpu_model = "AMD RYZEN AI MAX+ 395 w/ Radeon 8060S"   # /proc/cpuinfo verbatim; probed if absent
   max_load_avg = 1.0                    # run.py refuses to measure above this without --allow-noisy
   default_isa = "x86-64-avx512"

   [pinning]
   cpu_list = "0,2,4,6"                  # taskset list, one logical CPU per physical core; omit on macOS

   [isa."x86-64-avx512"]
   flags = ["-mavx512dq", "-mfma"]       # becomes CMAKE_CXX_FLAGS; the key is recorded as the ISA target

   [arms.openblas]
   cmake_options = ["-DEIGEN_BENCH_REFERENCE=openblas"]
   thread_env = { OPENBLAS_NUM_THREADS = "{threads}" }
   ```

   `EIGEN_BENCH_REFERENCE` names a row of `vendors.cmake` (`openblas`, `mkl`, `mkl_bypass`, `aocl`,
   `blis`, `armpl`, `nvpl`, `accelerate`, `netlib`). Where CMake's FindBLAS cannot find the library,
   name it: `-DBLAS_LIBRARIES=<path>`, `-DLAPACK_LIBRARIES=<path>[;<path>...]`, and
   `-DCMAKE_LIBRARY_PATH=<dir>` for an install layout no default prefix covers. A library with no
   version query (Accelerate, netlib) gets `-DEIGEN_BENCH_REFERENCE_VERSION="..."`. The existing
   profiles carry the exact lines each library needed on its host.

3. **Run**, one arm at a time, on an otherwise idle machine:

   ```bash
   cd benchmarks/comparison
   python3 run.py --machine <id> --arms openblas --scalars f32,f64 --exclude 'm:(6144|8192|12288|16384)/'
   ```

   Each invocation configures and builds `benchmarks/` into `build-comparison/<isa>__<arm>`, runs every
   `bench_*_compare` binary with every library's thread count set to `--threads` (default 1), and
   writes `results/<id>/<eigen-commit>/<id>-<isa>-<arm>-<commit>-<time>.json`. `--ops`, `--scalars`,
   `--filter` and `--exclude` narrow a run; `--isa` selects other `[isa.*]` targets from the profile;
   `--repetitions` (10) and `--min-time` (0.2s) go to Google Benchmark. The script refuses a worktree
   with modified tracked files and a loaded machine, so that every file records the commit it
   measured and conditions it was measured under; `--note` adds free text to the record. The largest
   sizes in the GEMM grid take minutes per repetition, hence the `--exclude` above.

4. **Make the page.** The website repository, <https://gitlab.com/libeigen/libeigen.gitlab.io>, has
   `scripts/benchmarks/plot_comparison.py`, which takes the result files and writes the charts, the
   per-shape tables and a ratio summary; a page is that plus a conditions block and observations,
   following the existing pages under `content/en/benchmarks/`. Commit the result files beside the
   charts so the page can be regenerated.

## What a result file holds

`machine` (id, CPU model, OS, kernel), `build` (ISA target, compiler and flags, Eigen commit and
version, whether the worktree was dirty), `arms` (Eigen, and the library with the version string it
reported at run time, the file it was linked from and where its LAPACK came from), `run` (times,
thread caps, pinning, load before and after, notes) and `measurements`: one entry per benchmark name
with the shape and the per-repetition flop rates in flops per second.

## Adding an operation

Copy one of the `bench_*_compare.cpp` files: declare the Fortran routine, write the Eigen and the
reference kernels, validate the reference result against Eigen outside the timed loop, report the
flop count through `eigen_bench::setFlopRate` with a formula from `bench_common.h`, and register the
grid with `REGISTER_COMPARISON_POINT`, which registers both arms of each point adjacently so a ratio is
never formed across minutes of drift. A source whose reference is a LAPACK routine carries the
`// EIGEN_BENCH_REFERENCE_FAMILY: lapack` marker so it is built Eigen-only against a library with no
LAPACK. `ctest` runs one iteration of every tiny cell, which exercises the validation.
