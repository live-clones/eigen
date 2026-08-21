#!/usr/bin/env python3
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

"""Regenerate the committed fixtures and golden files.

    python3 benchmarks/comparison/tests/regenerate.py            # everything
    python3 benchmarks/comparison/tests/regenerate.py --fixtures  # synthetic inputs
    python3 benchmarks/comparison/tests/regenerate.py --goldens   # rendered output

This is the ONLY sanctioned way to rewrite `tests/fixtures/`.  Everything it
writes is deterministic -- no clock, no hostname, no filesystem paths -- so a
regeneration diff shows a behaviour change and nothing else, which is the whole
point of keeping golden files under review.

`--goldens` shells out to `render.py` and `plots.py`; it is a no-op with a
warning while those do not exist yet.  Golden files that are checked in without
having been generated are a hand-written specification of the intended output:
the first real regeneration diff is the review of the renderer against it.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness_support as support  # noqa: E402

FIXTURES = support.FIXTURES
# Exactly the `small` group of shape_families.square3, so a contribution's scope
# is expressible as a group name and nothing is left `unaccounted`.
SIZES = [24, 32, 48, 64, 96, 128]
GROUP = "small"
GROUP_OF = {n: GROUP for n in SIZES}
EIGEN_GFLOPS = {24: 8.0, 32: 14.0, 48: 28.0, 64: 45.0, 96: 78.0, 128: 96.0}
REFERENCE_GFLOPS = {24: 5.0, 32: 10.0, 48: 30.0, 64: 60.0, 96: 90.0, 128: 95.0}
# 128 and 96 are the deliberate inconclusive points: 96 +/- 1 against 95 +/- 1,
# and 78 +/- 6 against 90 +/- 6. Everywhere else the intervals are disjoint.
MAD = {24: 0.5, 32: 0.5, 48: 0.5, 64: 0.5, 96: 6.0, 128: 1.0}


def square(n):
    return {"m": n, "n": n, "k": n}


def gemm_measurements(arm, gflops_of, sizes=SIZES, scalar="f64", threads=1, repetitions=10):
    return [
        support.make_measurement(
            arm=arm,
            scalar=scalar,
            shape=square(n),
            gflops=gflops_of[n],
            gflops_mad=MAD[n],
            shape_group=GROUP_OF[n],
            threads=threads,
            repetitions=repetitions,
        )
        for n in sizes
    ]


# --------------------------------------------------------------------------
# Result files
# --------------------------------------------------------------------------


def write_results():
    out = FIXTURES / "results"

    # The ordinary case: both arms, whole grid, nothing missing.
    support.write_json(
        out / "gemm_eigen_accelerate.json",
        support.make_result(
            run_id="m4pro-neon-accelerate-e2a2fda17-20260801T120000Z",
            measurements=gemm_measurements("eigen", EIGEN_GFLOPS)
            + gemm_measurements("accelerate", REFERENCE_GFLOPS),
        ),
    )

    # The single smallest document the schema accepts, for the schema tests.
    support.write_json(
        out / "minimal_valid.json",
        support.make_result(
            run_id="minimal-e2a2fda17-20260801T120000Z",
            measurements=[support.make_measurement(gflops=96.0)],
            scope={"ops": ["GEMM"], "arms": ["eigen"], "scalars": ["f64"], "threads": [1]},
            provenance=support.make_provenance(
                arms={"eigen": {"kind": "eigen", "library_name": "Eigen", "library_version": "5.0.1-master"}}
            ),
        ),
    )

    # Partial matrix: the reference arm covers two of the five shapes.
    covered = [24, 32]
    support.write_json(
        out / "gemm_partial_shapes.json",
        support.make_result(
            run_id="m4pro-neon-accelerate-e2a2fda17-20260803T120000Z",
            provenance=support.make_provenance(timestamp="2026-08-03T12:00:00Z"),
            measurements=gemm_measurements("eigen", EIGEN_GFLOPS)
            + gemm_measurements("accelerate", REFERENCE_GFLOPS, sizes=covered),
            not_measured=[
                support.make_not_measured(
                    shape=square(n),
                    reason="shape_unsupported",
                    detail=f"Reference arm skipped m=n=k={n} for this contribution",
                )
                for n in SIZES
                if n not in covered
            ],
        ),
    )

    # The reference library is not present at all: every reference cell is an
    # explicit negative, never an absence.
    support.write_json(
        out / "gemm_reference_unavailable.json",
        support.make_result(
            run_id="m4pro-neon-noref-e2a2fda17-20260804T120000Z",
            provenance=support.make_provenance(timestamp="2026-08-04T12:00:00Z"),
            measurements=gemm_measurements("eigen", EIGEN_GFLOPS),
            not_measured=[
                support.make_not_measured(
                    scalar=None,
                    shape=None,
                    reason="reference_library_unavailable",
                    detail="No BLAS was configured for this build",
                )
            ],
        ),
    )

    # Same cell keys as gemm_eigen_accelerate.json, newer timestamp, different
    # numbers: the duplicate/newest-wins case.
    faster = {n: value * 1.10 for n, value in EIGEN_GFLOPS.items()}
    support.write_json(
        out / "gemm_rerun_newer.json",
        support.make_result(
            run_id="m4pro-neon-accelerate-e2a2fda17-20260805T120000Z",
            provenance=support.make_provenance(timestamp="2026-08-05T12:00:00Z"),
            measurements=gemm_measurements("eigen", faster)
            + gemm_measurements("accelerate", REFERENCE_GFLOPS),
        ),
    )

    # One repetition: no dispersion is available, so mad is 0 and count is 1.
    single = support.make_result(
        run_id="m4pro-neon-accelerate-e2a2fda17-20260806T120000Z",
        provenance=support.make_provenance(timestamp="2026-08-06T12:00:00Z", repetitions=1),
        measurements=gemm_measurements("eigen", EIGEN_GFLOPS, repetitions=1)
        + gemm_measurements("accelerate", REFERENCE_GFLOPS, repetitions=1),
    )
    for measurement in single["measurements"]:
        for stat in measurement["stats"].values():
            stat.update(mad=0.0, count=1, stddev=0.0, cv=None, min=stat["median"], max=stat["median"])
    support.write_json(out / "gemm_single_repetition.json", single)

    # A machine that measured nothing at all: it still has to appear.
    support.write_json(
        out / "zero_ops_measured.json",
        support.make_result(
            run_id="m4max-neon-accelerate-e2a2fda17-20260807T120000Z",
            provenance=support.make_provenance(machine="m4max", timestamp="2026-08-07T12:00:00Z"),
            measurements=[],
            not_measured=[
                support.make_not_measured(arm=arm, scalar=None, shape=None, reason="build_failed", detail="Link failed: no BLAS")
                for arm in ("eigen", "accelerate")
            ],
        ),
    )

    # Schema-valid name whose dimension ORDER contradicts ops.toml.  Section 1.3
    # calls this a hard error: the C++ and the registry have diverged and every
    # number in the file is suspect.
    transposed = support.make_measurement(shape={"n": 512, "m": 512, "k": 512}, shape_dims=("n", "m", "k"), gflops=96.0)
    support.write_json(
        out / "gemm_bad_dim_order.json",
        support.make_result(
            run_id="m4pro-neon-baddims-e2a2fda17-20260808T120000Z",
            provenance=support.make_provenance(timestamp="2026-08-08T12:00:00Z"),
            measurements=[transposed],
            scope={
                "ops": ["GEMM"],
                "arms": ["eigen"],
                "scalars": ["f64"],
                "threads": [1],
                "shape_groups": {"GEMM": ["medium"]},
            },
        ),
    )

    # An op that ops.toml has never heard of.
    unknown = support.make_measurement(gflops=96.0)
    unknown["op"] = "NOSUCHOP"
    unknown["name"] = unknown["name"].replace("GEMM/", "NOSUCHOP/", 1)
    support.write_json(
        out / "unknown_op.json",
        support.make_result(
            run_id="m4pro-neon-unknownop-e2a2fda17-20260809T120000Z",
            provenance=support.make_provenance(timestamp="2026-08-09T12:00:00Z"),
            measurements=[unknown],
            scope={"ops": ["NOSUCHOP"], "arms": ["eigen"], "scalars": ["f64"], "threads": [1]},
        ),
    )

    # Identical timings recorded from binaries that reported different raw time
    # units, on two machines so they do not collide.  Every time in a result file
    # is already seconds; a reducer that re-applies the unit produces a 1000x
    # error that looks entirely plausible.
    for machine, unit, day in (("m4pro", "ns", "10"), ("m4max", "us", "11")):
        support.write_json(
            out / f"gemm_time_unit_{unit}.json",
            support.make_result(
                run_id=f"{machine}-neon-unit{unit}-e2a2fda17-202608{day}T120000Z",
                provenance=support.make_provenance(machine=machine, timestamp=f"2026-08-{day}T12:00:00Z"),
                measurements=[
                    support.make_measurement(
                        arm=arm,
                        shape=square(n),
                        gflops=(EIGEN_GFLOPS if arm == "eigen" else REFERENCE_GFLOPS)[n],
                        gflops_mad=MAD[n],
                        shape_group=GROUP,
                        source_time_unit=unit,
                    )
                    for arm in ("eigen", "accelerate")
                    for n in SIZES
                ],
            ),
        )

    # A NaN that reached the file.  Nothing downstream may render it as a number.
    nan_doc = support.make_result(
        run_id="m4pro-neon-nan-e2a2fda17-20260812T120000Z",
        provenance=support.make_provenance(timestamp="2026-08-12T12:00:00Z"),
        measurements=gemm_measurements("eigen", EIGEN_GFLOPS),
        scope={
            "ops": ["GEMM"],
            "arms": ["eigen"],
            "scalars": ["f64"],
            "threads": [1],
            "shape_groups": {"GEMM": [GROUP]},
        },
    )
    for stat in nan_doc["measurements"][2]["stats"].values():
        stat["median"] = float("nan")
    (FIXTURES / "results" / "gemm_nan_row.json").write_text(
        json.dumps(nan_doc, indent=2, sort_keys=True, allow_nan=True) + "\n"
    )


# --------------------------------------------------------------------------
# Merged intermediates
# --------------------------------------------------------------------------


def gemm_cells():
    cells = []
    for n in SIZES:
        flops = support.gemm_flops(n, n, n)
        eigen = support.merged_arm_measured(EIGEN_GFLOPS[n], MAD[n], flops)
        if n == SIZES[-1]:
            reference = support.merged_arm_absent(
                "reference_routine_absent",
                f"The reference build in this contribution did not register m=n=k={n}",
            )
            ratio, state = None, "not_measured"
        else:
            reference = support.merged_arm_measured(REFERENCE_GFLOPS[n], MAD[n], flops)
            ratio = EIGEN_GFLOPS[n] / REFERENCE_GFLOPS[n]
            lo_e, hi_e = EIGEN_GFLOPS[n] - MAD[n], EIGEN_GFLOPS[n] + MAD[n]
            lo_r, hi_r = REFERENCE_GFLOPS[n] - MAD[n], REFERENCE_GFLOPS[n] + MAD[n]
            state = "inconclusive" if lo_e <= hi_r and lo_r <= hi_e else "ok"
        cells.append(
            support.make_cell(
                shape=square(n),
                shape_group=GROUP_OF[n],
                arms={"eigen": eigen, "accelerate": reference},
                ratio=ratio,
                ratio_state=state,
            )
        )
    # An Eigen-only operation: the reference cannot exist, which is a different
    # statement from "was not measured".
    reason = support.load_ops()["ops"]["FULLPIVLU"]["reference"]["reason"]
    cells.append(
        support.make_cell(
            op="FULLPIVLU",
            op_family="eigen-only",
            shape={"m": 512, "n": 512},
            shape_dims=("m", "n"),
            arms={
                "eigen": support.merged_arm_measured(31.5, 0.4, 2 * 512 * 512 * 512 - 2 * 512**3 / 3),
                "accelerate": support.merged_arm_absent("no_reference_equivalent", reason),
            },
            flops=2 * 512 * 512 * 512 - 2 * 512**3 / 3,
            nominal=True,
            ratio=None,
            ratio_state="no_reference_equivalent",
        )
    )
    return cells


def write_merged():
    out = FIXTURES / "merged"
    support.write_json(out / "gemm_merged.json", support.make_merged(gemm_cells()))

    # The one-cell dataset: every other point of the grid must render as
    # "not measured", and adding a second cell must not disturb this one.
    n = 24
    flops = support.gemm_flops(n, n, n)
    one = support.make_cell(
        shape=square(n),
        shape_group=GROUP,
        arms={
            "eigen": support.merged_arm_measured(EIGEN_GFLOPS[n], MAD[n], flops),
            "accelerate": support.merged_arm_measured(REFERENCE_GFLOPS[n], MAD[n], flops),
        },
        ratio=EIGEN_GFLOPS[n] / REFERENCE_GFLOPS[n],
        ratio_state="ok",
    )
    support.write_json(out / "one_cell.json", support.make_merged([one]))


# --------------------------------------------------------------------------
# Canned Google Benchmark output
# --------------------------------------------------------------------------


def gbench_rows(name, gflops, flops, repetitions=3, family_index=0, instance=0, threads=1):
    """One benchmark's iteration rows plus the four aggregate rows 1.9.5 emits."""
    rates = [gflops * 1e9 * factor for factor in (0.98, 1.00, 1.02)][:repetitions]
    rows = []
    for index, rate in enumerate(rates):
        seconds = flops / rate
        rows.append(
            {
                "name": name,
                "family_index": family_index,
                "per_family_instance_index": instance,
                "run_name": name,
                "run_type": "iteration",
                "repetitions": repetitions,
                "repetition_index": index,
                "threads": threads,
                "iterations": 512,
                "real_time": seconds * 1e9,
                "cpu_time": seconds * 1e9 * 0.99,
                "time_unit": "ns",
                "GFLOPS": rate,
            }
        )
    mean_rate = sum(rates) / len(rates)
    ordered = sorted(rates)
    median_rate = ordered[len(ordered) // 2]
    aggregates = {
        "mean": mean_rate,
        "median": median_rate,
        "stddev": (sum((r - mean_rate) ** 2 for r in rates) / (len(rates) - 1)) ** 0.5,
        "cv": 0.02,
    }
    for aggregate, value in aggregates.items():
        seconds = flops / (value if aggregate in ("mean", "median") and value else mean_rate)
        rows.append(
            {
                "name": f"{name}_{aggregate}",
                "family_index": family_index,
                "per_family_instance_index": instance,
                "run_name": name,
                "run_type": "aggregate",
                "repetitions": repetitions,
                "threads": threads,
                "aggregate_name": aggregate,
                "aggregate_unit": "percentage" if aggregate == "cv" else "time",
                "iterations": 512,
                "real_time": seconds * 1e9,
                "cpu_time": seconds * 1e9 * 0.99,
                "time_unit": "ns",
                "GFLOPS": value,
            }
        )
    return rows


def write_raw():
    context = {
        "date": "2026-08-01T05:00:00-07:00",
        "host_name": "test-host",
        "executable": "./bench_gemm_compare",
        "num_cpus": 14,
        "mhz_per_cpu": 4400,
        "cpu_scaling_enabled": False,
        "caches": [{"type": "Data", "level": 1, "size": 131072, "num_sharing": 1}],
        "load_avg": [0.31, 0.29, 0.30],
        "library_version": "v1.9.5",
        "library_build_type": "release",
        "json_schema_version": 1,
        "eigen_bench.schema_version": "1.0.0",
        "eigen_bench.reference_arm": "accelerate",
        "eigen_bench.reference_library_name": "Apple Accelerate",
        "eigen_bench.reference_library_version": "macOS 15.6",
        "eigen_bench.reference_library_path": "/System/Library/Frameworks/Accelerate.framework/Accelerate",
        "eigen_bench.reference_interface": "lp64",
        "eigen_bench.reference_threading": "gcd",
        "eigen_bench.eigen_commit": support.EIGEN_COMMIT,
        "eigen_bench.eigen_dirty": "false",
        "eigen_bench.compiler_id": "AppleClang",
        "eigen_bench.compiler_version": "17.0.0.17000013",
        "eigen_bench.cxx_standard": "17",
        "eigen_bench.cxx_flags": "-O3 -DNDEBUG",
        "eigen_bench.isa_target": "aarch64-neon",
        "eigen_bench.eigen_nb_threads": "1",
        "eigen_bench.thread_env": '{"VECLIB_MAXIMUM_THREADS": "1", "OMP_NUM_THREADS": "1"}',
        "eigen_bench.ops_toml_sha256": support.DUMMY_SHA256,
    }
    rows = []
    family = 0
    for n in (64, 128):
        flops = support.gemm_flops(n, n, n)
        for instance, (arm, table) in enumerate((("eigen", EIGEN_GFLOPS), ("accelerate", REFERENCE_GFLOPS))):
            rows.extend(
                gbench_rows(
                    f"GEMM/{arm}/f64/m:{n}/n:{n}/k:{n}",
                    table[n],
                    flops,
                    family_index=family,
                    instance=instance,
                )
            )
        family += 1
    # A row the binary could not run.  It must become an explicit negative, not
    # a measurement with a zero in it.
    rows.append(
        {
            "name": "GEMM/accelerate/f64/m:256/n:256/k:256",
            "family_index": family,
            "per_family_instance_index": 1,
            "run_name": "GEMM/accelerate/f64/m:256/n:256/k:256",
            "run_type": "iteration",
            "repetitions": 3,
            "repetition_index": 0,
            "threads": 1,
            "error_occurred": True,
            "error_message": "reference routine returned a nonzero info",
            "iterations": 0,
            "real_time": 0.0,
            "cpu_time": 0.0,
            "time_unit": "ns",
        }
    )
    support.write_json(FIXTURES / "raw" / "gbench_gemm.json", {"context": context, "benchmarks": rows}, sort_keys=False)



def write_listing(build_dir=None):
    """Capture `--benchmark_list_tests` from a real binary.

    This fixture is the *registered* side of the reconciliation in
    `test_ops_registry.py`: it answers "what did the C++ actually register",
    which is exactly the question `ops.toml` cannot answer about itself.  A
    synthesised listing would make that test compare `ops.toml` against a
    restatement of `ops.toml` and pass while covering nothing, so this refuses
    to invent one -- an out-of-date capture is recoverable, a fabricated one is
    a silently empty test.
    """
    target = FIXTURES / "raw" / "benchmark_list_tests.txt"
    exe = None
    roots = [Path(build_dir)] if build_dir else [support.REPO_ROOT / "build-bench", support.REPO_ROOT / "build"]
    for root in roots:
        if root.is_dir():
            exe = next(iter(sorted(root.rglob("bench_gemm_compare"))), None)
            if exe:
                break
    if exe is None:
        searched = ", ".join(str(r) for r in roots)
        print(
            f"no bench_gemm_compare binary found (looked in {searched}); leaving\n"
            f"  {target.relative_to(support.REPO_ROOT)}\n"
            "unchanged.  To refresh it, build the comparison target against a real\n"
            "reference BLAS and re-run with --build-dir:\n"
            "  cmake -G Ninja -S benchmarks -B build-bench -DCMAKE_BUILD_TYPE=Release\n"
            "  cmake --build build-bench\n"
            "  python3 benchmarks/comparison/tests/regenerate.py --fixtures --build-dir build-bench",
            file=sys.stderr,
        )
        return
    proc = subprocess.run([str(exe), "--benchmark_list_tests=true"], capture_output=True, text=True, check=True)
    # Registration order, not sorted: the interleaving of the eigen and reference
    # arms is exactly what this capture is evidence of.
    names = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not names:
        print(f"{exe} listed no benchmarks; leaving {target.name} unchanged", file=sys.stderr)
        return
    previous = target.read_text().splitlines() if target.exists() else []
    if previous and len(names) < len(previous):
        print(
            f"refusing to shrink {target.name} from {len(previous)} to {len(names)} names.\n"
            "That normally means the binary was built without a reference arm, or with a\n"
            "filter applied.  Delete the file first if the reduction is intended.",
            file=sys.stderr,
        )
        return
    target.write_text("".join(name + "\n" for name in names))
    print(f"captured {len(names)} benchmark names from {exe}", file=sys.stderr)


# --------------------------------------------------------------------------
# Golden files
# --------------------------------------------------------------------------


def write_goldens():
    merged = FIXTURES / "merged" / "gemm_merged.json"
    golden = FIXTURES / "golden"
    golden.mkdir(parents=True, exist_ok=True)
    if not support.have_script("render.py"):
        print("render.py does not exist yet; leaving the hand-written goldens alone", file=sys.stderr)
        return
    for fmt, filename in (
        ("doxygen", "comparison_tables.dox"),
        ("markdown", "comparison_tables.md"),
    ):
        proc = subprocess.run(
            [sys.executable, str(support.script_path("render.py")), "--format", fmt, "--out", "-", str(merged)],
            capture_output=True,
            text=True,
            cwd=str(support.REPO_ROOT),
            check=True,
        )
        (golden / filename).write_text(proc.stdout)
    subprocess.run(
        [
            sys.executable,
            str(support.script_path("render.py")),
            "--format",
            "coverage",
            "--out-dir",
            str(golden),
            str(merged),
        ],
        check=True,
        cwd=str(support.REPO_ROOT),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixtures", action="store_true", help="regenerate the synthetic inputs")
    parser.add_argument("--goldens", action="store_true", help="regenerate the rendered goldens")
    parser.add_argument("--build-dir", help="build tree holding bench_gemm_compare, for the listing capture")
    args = parser.parse_args()
    everything = not (args.fixtures or args.goldens)
    if everything or args.fixtures:
        write_results()
        write_merged()
        write_raw()
        write_listing(args.build_dir)
    if everything or args.goldens:
        write_goldens()
    return 0


if __name__ == "__main__":
    sys.exit(main())
