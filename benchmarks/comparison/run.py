#!/usr/bin/env python3
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0
"""Build and run the comparison benchmarks for one machine, one result file per (ISA target, arm).

    run.py --machine ID [--arms LIST] [--isa LIST] [--ops LIST] [--scalars LIST]
           [--filter REGEX] [--exclude REGEX] [--repetitions N] [--min-time SPEC]

A machine profile is machines/<ID>.toml (see README.md). For every requested ISA
target and reference arm the script configures and builds benchmarks/ into
<build-dir>/<isa>__<arm>, runs each bench_*_compare binary with every library's
thread count pinned to --threads, and writes the per-repetition rates together
with the conditions of the run to <results-dir>/<ID>/<eigen-commit>/. The
website's chart script draws pages from those files.

Python 3.11 or newer (tomllib).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RESULT_SCHEMA = "eigen-comparison-run/2"

# Every library's thread cap is set on every run, so an arm that is not the
# one linked cannot race a multithreaded default against single-threaded Eigen.
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
    "NVPL_BLAS_NUM_THREADS", "ARMPL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "ACCELERATE_MAXIMUM_THREADS",
)
FIXED_ENV = {"MKL_DYNAMIC": "FALSE", "OMP_DYNAMIC": "FALSE"}

# Benchmark names are "OP/arm/scalar/dim:value[/dim:value...]" (bench_compare.h).
NAME_RE = re.compile(r"^(?P<op>[A-Z0-9]+)/(?P<arm>[a-z][a-z0-9_]*)/(?P<scalar>[a-z0-9]+)(?P<dims>(?:/[a-z]+:\d+)+)$")


def fail(message: str, code: int = 1) -> None:
    print(f"run.py: error: {message}", file=sys.stderr)
    sys.exit(code)


def csv(values: list[str] | None) -> list[str]:
    out: list[str] = []
    for value in values or []:
        out += [v.strip() for v in value.split(",") if v.strip()]
    return out


# --------------------------------------------------------------------------
# Machine profile and host facts
# --------------------------------------------------------------------------


def load_profile(machines_dir: Path, machine_id: str) -> dict:
    path = machines_dir / f"{machine_id}.toml"
    if not path.is_file():
        fail(f"no machine profile {path}")
    profile = tomllib.load(open(path, "rb"))
    for key in ("id", "display_name", "isa", "default_isa", "arms"):
        if key not in profile:
            fail(f"{path}: missing required key '{key}'")
    if profile["id"] != machine_id:
        fail(f"{path}: id {profile['id']!r} does not match the file name")
    if profile["default_isa"] not in profile["isa"]:
        fail(f"{path}: default_isa {profile['default_isa']!r} is not an [isa.*] table")
    return profile


def cpu_model() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    if sys.platform == "darwin":
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        if out.returncode == 0:
            return out.stdout.strip()
    return platform.processor() or platform.machine()


def os_release() -> str:
    try:
        for line in open("/etc/os-release"):
            if line.startswith("PRETTY_NAME="):
                return line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    return platform.platform()


def git(*args: str) -> str:
    out = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else ""


def eigen_version() -> str:
    text = (REPO / "Eigen" / "Version").read_text(errors="replace")
    m = re.search(r'#define EIGEN_VERSION_STRING "([^"]+)"', text)
    return m.group(1) if m else ""


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------


def configure_and_build(build_dir: Path, profile: dict, isa: str, arm: dict, jobs: int) -> None:
    generator = profile.get("build", {}).get("generator") or ("Ninja" if shutil.which("ninja") else "Unix Makefiles")
    command = [
        "cmake", "-S", str(REPO / "benchmarks"), "-B", str(build_dir), "-G", generator,
        "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CXX_STANDARD=17",
        "-DCMAKE_CXX_FLAGS=" + " ".join(profile["isa"][isa].get("flags", [])),
        f"-DEIGEN_BENCH_ISA_TARGET={isa}",
        *arm.get("cmake_options", []),
    ]
    if subprocess.run(command).returncode != 0:
        fail("configure failed", 5)
    if subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", "bench_comparison_all", "--parallel", str(jobs)]
    ).returncode != 0:
        fail("build failed", 5)


def vendor_info(build_dir: Path) -> dict:
    path = build_dir / "comparison" / "vendor_info.json"
    if not path.is_file():
        fail(f"{path} missing; was the tree configured by this script?", 5)
    return json.load(open(path))


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------


def benchmark_env(threads: int, arm: dict) -> dict[str, str]:
    env = dict(os.environ)
    for var in THREAD_ENV_VARS:
        env[var] = str(threads)
    env.update(FIXED_ENV)
    for var, value in arm.get("thread_env", {}).items():
        env[var] = str(value).replace("{threads}", str(threads))
    return env


def list_benchmarks(binary: Path, pattern: str, env: dict[str, str]) -> list[str]:
    out = subprocess.run(
        [str(binary), "--benchmark_list_tests=true", f"--benchmark_filter={pattern}"],
        capture_output=True, text=True, env=env,
    )
    return [line.strip() for line in out.stdout.splitlines() if NAME_RE.match(line.strip())]


def run_binary(binary: Path, names: list[str], args: argparse.Namespace, env: dict[str, str],
               pin: list[str], out_json: Path) -> int:
    command = [
        *pin, str(binary),
        f"--benchmark_filter=^({'|'.join(names)})$",
        f"--benchmark_repetitions={args.repetitions}",
        f"--benchmark_min_time={args.min_time}",
        f"--benchmark_out={out_json}", "--benchmark_out_format=json",
        "--benchmark_report_aggregates_only=false",
    ]
    return subprocess.run(command, env=env, stdout=subprocess.DEVNULL).returncode


def parse_benchmark_json(path: Path) -> tuple[dict, list[dict]]:
    """Google Benchmark's JSON -> (context, per-name measurements with one sample per repetition)."""
    data = json.load(open(path))
    by_name: dict[str, dict] = {}
    for row in data.get("benchmarks", []):
        if row.get("run_type", "iteration") != "iteration":
            continue
        name = row.get("run_name") or row["name"]
        m = NAME_RE.match(name)
        if not m:
            continue
        cell = by_name.setdefault(name, {
            "name": name, "op": m["op"], "arm": m["arm"], "scalar": m["scalar"],
            "shape": {}, "shape_dims": [], "samples": {"flop_rate": [], "real_time_s": []},
            "iterations": [], "error": None,
        })
        if not cell["shape_dims"]:
            for item in m["dims"].strip("/").split("/"):
                dim, value = item.split(":")
                cell["shape_dims"].append(dim)
                cell["shape"][dim] = int(value)
        if row.get("error_occurred"):
            cell["error"] = row.get("error_message", "error")
            continue
        unit = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}[row.get("time_unit", "ns")]
        cell["samples"]["real_time_s"].append(row["real_time"] * unit)
        cell["samples"]["flop_rate"].append(row.get("GFLOPS"))  # flops per second despite the name
        cell["iterations"].append(row.get("iterations"))
    return data.get("context", {}), list(by_name.values())


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--machine", required=True, metavar="ID", help="machines/<ID>.toml")
    ap.add_argument("--arms", action="append", metavar="LIST", help="reference arms from the profile; default: all")
    ap.add_argument("--isa", action="append", metavar="LIST", help="ISA targets from the profile; default: default_isa")
    ap.add_argument("--ops", action="append", metavar="LIST", help="operations, e.g. GEMM,POTRF; default: all built")
    ap.add_argument("--scalars", action="append", metavar="LIST", help="default: f64")
    ap.add_argument("--filter", default=None, metavar="REGEX", help="keep only benchmark names matching")
    ap.add_argument("--exclude", default=None, metavar="REGEX", help="drop benchmark names matching")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--repetitions", type=int, default=10)
    ap.add_argument("--min-time", default="0.2s", metavar="SPEC", help="--benchmark_min_time")
    ap.add_argument("--build-dir", default=str(REPO / "build-comparison"), metavar="DIR")
    ap.add_argument("--results-dir", default=str(HERE / "results"), metavar="DIR")
    ap.add_argument("--machines-dir", default=str(HERE / "machines"), metavar="DIR")
    ap.add_argument("--no-build", action="store_true", help="reuse the configured and built tree")
    ap.add_argument("-j", dest="jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--note", default=None, metavar="TEXT", help="recorded in the result's run.notes")
    ap.add_argument("--allow-dirty", action="store_true", help="measure a worktree with modified tracked files")
    ap.add_argument("--allow-noisy", action="store_true", help="measure above the profile's max_load_avg")
    ap.add_argument("--dry-run", action="store_true", help="build and list the cells, measure nothing")
    args = ap.parse_args()

    profile = load_profile(Path(args.machines_dir), args.machine)
    arms = csv(args.arms) or list(profile["arms"])
    for arm in arms:
        if arm not in profile["arms"]:
            fail(f"profile {args.machine} has no [arms.{arm}]", 2)
    isas = csv(args.isa) or [profile["default_isa"]]
    for isa in isas:
        if isa not in profile["isa"]:
            fail(f"profile {args.machine} has no [isa.\"{isa}\"]", 2)
    scalars = csv(args.scalars) or ["f64"]
    ops = [op.upper() for op in csv(args.ops)]

    commit = git("rev-parse", "HEAD")
    version = eigen_version()
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    if dirty and not args.allow_dirty:
        fail("the Eigen worktree has modified tracked files; commit them or pass --allow-dirty", 3)
    load = os.getloadavg()[0]
    if load > float(profile.get("max_load_avg", 1.0)) and not args.allow_noisy:
        fail(f"1-minute load average {load:.2f} exceeds max_load_avg; wait or pass --allow-noisy", 4)

    cpu_list = profile.get("pinning", {}).get("cpu_list")
    pin = ["taskset", "-c", str(cpu_list)] if cpu_list and shutil.which("taskset") else []
    results_root = Path(args.results_dir) / args.machine / commit[:9]
    written = []

    for isa in isas:
        for arm_key in arms:
            arm = profile["arms"][arm_key]
            build_dir = Path(args.build_dir) / f"{isa}__{arm_key}"
            if not args.no_build:
                configure_and_build(build_dir, profile, isa, arm, args.jobs)
            info = vendor_info(build_dir)
            if info.get("arm") != arm_key:
                fail(f"{build_dir} was configured for arm {info.get('arm')!r}, not {arm_key!r}", 2)
            binaries = sorted(p for p in (build_dir / "comparison").glob("bench_*_compare") if os.access(p, os.X_OK))
            env = benchmark_env(args.threads, arm)
            started = dt.datetime.now(dt.timezone.utc)
            load_before = os.getloadavg()
            context: dict = {}
            measurements: list[dict] = []
            failures: list[str] = []
            for binary in binaries:
                op = binary.name[len("bench_"):-len("_compare")].upper()
                if ops and op not in ops:
                    continue
                pattern = f"^{op}/(eigen|{arm_key})/({'|'.join(scalars)})/"
                names = list_benchmarks(binary, pattern, env)
                if args.filter:
                    names = [n for n in names if re.search(args.filter, n)]
                if args.exclude:
                    names = [n for n in names if not re.search(args.exclude, n)]
                if not names:
                    continue
                print(f"run.py: {isa}/{arm_key}: {binary.name}: {len(names)} cells", file=sys.stderr)
                if args.dry_run:
                    for name in names:
                        print(name)
                    continue
                out_json = build_dir / "comparison" / f"{binary.name}.{isa}.{arm_key}.json"
                status = run_binary(binary, names, args, env, pin, out_json)
                if status != 0:
                    failures.append(f"{binary.name} exited {status}")
                if out_json.is_file():
                    ctx, cells = parse_benchmark_json(out_json)
                    context = context or ctx
                    measurements += cells
            if args.dry_run:
                continue
            finished = dt.datetime.now(dt.timezone.utc)
            errors = [m["name"] for m in measurements if m["error"]]
            if not measurements:
                fail(f"{isa}/{arm_key}: nothing was measured ({'; '.join(failures) or 'no matching cells'})", 6)

            def ctx(key: str) -> str:
                return str(context.get(f"eigen_bench.{key}", ""))

            result = {
                "schema": RESULT_SCHEMA,
                "machine": {
                    "id": profile["id"], "display_name": profile["display_name"],
                    "cpu_model": profile.get("cpu_model") or cpu_model(), "logical_cpus": os.cpu_count(),
                    "os": os_release(), "kernel": platform.release(), "notes": profile.get("notes", "").strip(),
                },
                "build": {
                    "isa_target": ctx("isa_target") or isa,
                    "compiler": f"{ctx('compiler_id')} {ctx('compiler_version')}".strip(),
                    "cxx_flags": ctx("cxx_flags"), "cxx_standard": ctx("cxx_standard"),
                    "eigen_commit": commit, "eigen_dirty": dirty, "eigen_version": version,
                },
                "arms": [
                    {"key": "eigen", "library_name": "Eigen", "library_version": f"{version} ({commit[:9]})"},
                    {
                        "key": arm_key,
                        "library_name": ctx("reference_library_name") or info.get("library_name") or arm_key,
                        "library_version": ctx("reference_library_version") or info.get("library_version_fallback", ""),
                        "library_path": info.get("library_path", ""),
                        "libraries": info.get("libraries", []),
                        "lapack": info.get("lapack", {}),
                        "notes": info.get("notes", ""),
                    },
                ],
                "run": {
                    "started_utc": started.isoformat(timespec="seconds"),
                    "finished_utc": finished.isoformat(timespec="seconds"),
                    "duration_s": round((finished - started).total_seconds(), 1),
                    "threads": args.threads, "repetitions": args.repetitions, "min_time": args.min_time,
                    "pinning": " ".join(pin) or None,
                    "env": {k: env[k] for k in sorted(env) if k in THREAD_ENV_VARS or k in FIXED_ENV
                            or k in arm.get("thread_env", {})},
                    "load_avg_before": load_before, "load_avg_after": os.getloadavg(),
                    "allow_dirty": args.allow_dirty, "allow_noisy": args.allow_noisy,
                    "failures": failures, "errored_cells": errors,
                    "notes": args.note, "argv": sys.argv[1:],
                },
                "measurements": measurements,
            }
            results_root.mkdir(parents=True, exist_ok=True)
            stamp = started.strftime("%Y%m%dT%H%M%SZ")
            out = results_root / f"{args.machine}-{isa}-{arm_key}-{commit[:9]}-{stamp}.json"
            out.write_text(json.dumps(result, indent=1) + "\n")
            written.append(out)
            print(f"run.py: wrote {out} ({len(measurements)} cells"
                  + (f", {len(errors)} errored" if errors else "") + ")", file=sys.stderr)
            for message in failures:
                print(f"run.py: warning: {isa}/{arm_key}: {message}", file=sys.stderr)
    return 0 if written or args.dry_run else 1


if __name__ == "__main__":
    sys.exit(main())
