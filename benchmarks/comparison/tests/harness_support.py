# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

"""Shared helpers for the comparison-harness test suite.

Three jobs:

* locate `benchmarks/comparison/{ops.toml,result_schema.json,run.py,...}` and the
  committed fixtures without depending on the caller's working directory;
* build synthetic result and merged documents, so a test can state the case it
  cares about ("one repetition", "reference library missing") in one call;
* reach the implementations in a way that survives the drift that is expected
  while `run.py`/`reduce.py`/`render.py`/`plots.py` are written in parallel with
  these tests -- `resolve_callable` fails with the list of names it looked for
  rather than silently skipping, and `require_script` skips with the exact path
  that is missing.

Nothing here talks to the network, builds anything, or writes outside `tmp_path`
except `regenerate.py`, which is the single documented fixture/golden refresh.
"""

import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
COMPARISON_DIR = TESTS_DIR.parent
BENCHMARKS_DIR = COMPARISON_DIR.parent
REPO_ROOT = BENCHMARKS_DIR.parent
FIXTURES = TESTS_DIR / "fixtures"

OPS_TOML = COMPARISON_DIR / "ops.toml"
RESULT_SCHEMA = COMPARISON_DIR / "result_schema.json"

if str(COMPARISON_DIR) not in sys.path:
    sys.path.insert(0, str(COMPARISON_DIR))

IMPLEMENTATIONS = ("run.py", "reduce.py", "render.py", "plots.py")

# CONTRACTS.md section 1.3.
RESERVED_DIMS = {
    "repeats",
    "iterations",
    "min_time",
    "min_warmup_time",
    "manual_time",
    "real_time",
    "big_o",
    "rms",
    "process_time",
}

# result_schema.json provenance.threading.env: "At minimum probe ...".  A vendor
# whose thread count is left to the library's own default silently turns a
# single-threaded comparison into a multi-threaded one, which is the single
# most expensive scripting bug this harness can have.
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NVPL_BLAS_NUM_THREADS",
    "ARMPL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "GOTO_NUM_THREADS",
    "EIGEN_NB_THREADS",
)
AFFINITY_ENV_VARS = ("OMP_PROC_BIND", "OMP_PLACES", "KMP_AFFINITY")

DUMMY_SHA256 = "0123456789abcdef" * 4
EIGEN_COMMIT = "e2a2fda17c0b1d3e4f5a6b7c8d9e0f1a2b3c4d5e"
EIGEN_COMMIT_SHORT = "e2a2fda17"
CONFIG_ID = "m4pro__aarch64-neon__appleclang17__e2a2fda17__t1"
HARNESS_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0.0"


# --------------------------------------------------------------------------
# Registry and schema
# --------------------------------------------------------------------------


def load_ops(path=OPS_TOML):
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def load_schema(path=RESULT_SCHEMA):
    return json.loads(Path(path).read_text())


def schema_validator(schema=None):
    jsonschema = pytest.importorskip("jsonschema")
    schema = schema if schema is not None else load_schema()
    cls = jsonschema.validators.validator_for(schema)
    cls.check_schema(schema)
    return cls(schema)


def family_points(ops, family_name, groups=None):
    """Ordered union of a shape family's selected groups, first occurrence winning."""
    family = ops["shape_families"][family_name]
    selected = groups if groups is not None else family["default_groups"]
    seen, points = set(), []
    for name in selected:
        for group in family["groups"]:
            if group["name"] != name:
                continue
            for point in group["points"]:
                key = tuple(point)
                if key not in seen:
                    seen.add(key)
                    points.append(list(point))
    return points


def op_grid(ops, op):
    return family_points(ops, ops["ops"][op]["shape_family"])


def implemented_ops(ops):
    return sorted(k for k, v in ops["ops"].items() if v.get("status") == "implemented")


# --------------------------------------------------------------------------
# Reaching the implementations
# --------------------------------------------------------------------------


def script_path(name):
    return COMPARISON_DIR / name


def have_script(name):
    return script_path(name).is_file()


def require_script(name):
    path = script_path(name)
    if not path.is_file():
        pytest.skip(
            f"{path} does not exist yet; test_implementations_present.py is the "
            f"loud failure for that, this test has nothing to exercise"
        )
    return path


def import_impl(name):
    """Import `benchmarks/comparison/<name>` as a module, skipping if absent."""
    path = require_script(name)
    mod_name = "eigen_comparison_" + path.stem
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[mod_name]
        raise
    return module


def resolve_callable(module, *candidates):
    """First callable attribute among `candidates`, else a failure naming them all.

    Deliberately a failure and not a skip: an implementation that renamed the
    function is drift to reconcile, not an absence to tolerate.
    """
    for name in candidates:
        attribute = getattr(module, name, None)
        if callable(attribute):
            return attribute
    pytest.fail(
        f"{module.__name__} exposes none of {list(candidates)}; the test suite "
        f"needs one of them. Public names it does export: "
        f"{sorted(n for n in vars(module) if not n.startswith('_'))}"
    )


def run_cli(name, args, stdin=None, cwd=None, env=None, timeout=300):
    """Run one of the harness scripts, returning the CompletedProcess."""
    path = require_script(name)
    child_env = dict(os.environ)
    child_env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    if env:
        child_env.update(env)
    return subprocess.run(
        [sys.executable, str(path), *[str(a) for a in args]],
        input=stdin,
        capture_output=True,
        text=True,
        cwd=str(cwd or REPO_ROOT),
        env=child_env,
        timeout=timeout,
    )


def cli_json(name, args, **kwargs):
    """Run a script that writes JSON to stdout and parse it, asserting exit 0."""
    proc = run_cli(name, args, **kwargs)
    assert proc.returncode == 0, f"{name} {args} exited {proc.returncode}\n{proc.stderr}"
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:  # pragma: no cover - diagnostic path
        pytest.fail(f"{name} stdout is not JSON ({exc}):\n{proc.stdout[:2000]}")


# --------------------------------------------------------------------------
# Synthetic result documents
# --------------------------------------------------------------------------


def make_stat(median, mad, count=10):
    mean = median
    stddev = mad * 1.4826
    return {
        "median": median,
        "mad": mad,
        "mean": mean,
        "stddev": stddev,
        "cv": (stddev / mean) if mean else None,
        "min": median - mad,
        "max": median + mad,
        "count": count,
    }


def gemm_flops(m, n, k):
    return 2.0 * m * n * k


def benchmark_name(op, arm, scalar, shape_dims, shape, threads=1):
    fields = [op, arm, scalar] + [f"{d}:{shape[d]}" for d in shape_dims]
    if threads != 1:
        fields.append(f"threads:{threads}")
    return "/".join(fields)


def make_measurement(
    op="GEMM",
    arm="eigen",
    scalar="f64",
    shape=None,
    shape_dims=("m", "n", "k"),
    gflops=100.0,
    gflops_mad=1.0,
    threads=1,
    repetitions=10,
    flops=None,
    nominal=False,
    shape_group=None,
    source_time_unit="ns",
    name=None,
):
    shape = dict(shape or {"m": 512, "n": 512, "k": 512})
    shape_dims = list(shape_dims)
    flops = flops if flops is not None else gemm_flops(*[shape[d] for d in shape_dims])
    rate = gflops * 1e9
    rate_mad = gflops_mad * 1e9
    # A zero rate is a real, expressible outcome (an arm that completed no work per
    # unit time), and the pipeline has to say something honest about it, so the
    # helper must be able to construct one rather than dividing by it.
    seconds = flops / rate if rate else 0.0
    # MAD of t = flops/r under a small perturbation of r.
    seconds_mad = flops * rate_mad / (rate * rate) if rate else 0.0
    return {
        "name": name or benchmark_name(op, arm, scalar, shape_dims, shape, threads),
        "op": op,
        "arm": arm,
        "scalar": scalar,
        "shape": shape,
        "shape_dims": shape_dims,
        "shape_group": shape_group,
        "threads": threads,
        "repetitions": repetitions,
        "flops_per_iteration": flops,
        "flops_nominal": nominal,
        "stats": {
            "real_time_s": make_stat(seconds, seconds_mad, repetitions),
            "cpu_time_s": make_stat(seconds, seconds_mad, repetitions),
            "flop_rate": make_stat(rate, rate_mad, repetitions),
        },
        "source_time_unit": source_time_unit,
        "validated": True,
    }


def make_not_measured(op="GEMM", arm="accelerate", reason="reference_routine_absent", scalar="f64", shape=None, threads=1, detail=None):
    return {
        "op": op,
        "arm": arm,
        "scalar": scalar,
        "shape": dict(shape) if shape else None,
        "threads": threads,
        "reason": reason,
        "detail": detail,
    }


def make_provenance(
    machine="m4pro",
    timestamp="2026-08-01T12:00:00Z",
    arms=None,
    threads=1,
    argv=None,
    repetitions=10,
    frequency_governor=None,
    numa_policy=None,
):
    arms = arms or {
        "eigen": {"kind": "eigen", "library_name": "Eigen", "library_version": "5.0.1-master"},
        "accelerate": {
            "kind": "reference",
            "library_name": "Apple Accelerate",
            "library_version": "macOS 15.6",
            "provides": ["blas", "cblas", "lapack"],
            "threading_model": "gcd",
            "interface": "lp64",
        },
    }
    env = {var: str(threads) for var in THREAD_ENV_VARS}
    return {
        "timestamp_utc": timestamp,
        "hostname": "test-host",
        "machine_config_id": machine,
        "machine_config_sha256": DUMMY_SHA256,
        "cpu": {
            "model": "Apple M4 Pro",
            "vendor": "Apple",
            "sockets": 1,
            "cores_per_socket": 14,
            "threads_per_core": 1,
            "logical_cpus": 14,
            "performance_cores": 10,
            "efficiency_cores": 4,
            "smt_enabled": False,
            "frequency_governor": frequency_governor,
            "caches": [{"level": 1, "type": "Data", "size_bytes": 131072, "num_sharing": 1}],
        },
        "numa": {"nodes": 1, "policy": numa_policy, "cpu_binding": None, "node_of_run": None},
        "memory": {"total_bytes": 68719476736},
        "os": {"name": "Darwin", "release": "25.6.0", "kernel": "Darwin Kernel Version 25.6.0", "arch": "arm64"},
        "toolchain": {
            "compiler_id": "AppleClang",
            "compiler_version": "17.0.0.17000013",
            "cxx_standard": 17,
            "cxx_flags": ["-O3", "-DNDEBUG"],
            "link_flags": [],
            "isa_target": "aarch64-neon",
            "isa_flags": [],
            "cmake_build_type": "Release",
            "benchmark_library_version": "v1.9.5",
        },
        "threading": {
            "requested_threads": threads,
            "env": env,
            "eigen_nb_threads": threads,
            "eigen_has_openmp": False,
        },
        "arms": arms,
        "eigen": {
            "commit": EIGEN_COMMIT,
            "commit_short": EIGEN_COMMIT_SHORT,
            "branch": "benchmark-comparison-harness",
            "dirty": False,
        },
        "harness": {
            "name": "eigen-comparison-harness",
            "version": HARNESS_VERSION,
            "argv": argv or ["run.py", "--machine", machine],
            "benchmark_argv": ["--benchmark_repetitions=10", "--benchmark_min_time=0.5s"],
        },
        "run": {"repetitions": repetitions, "min_time": "0.5s", "report_aggregates_only": False},
    }


def make_result(
    run_id="m4pro-neon-accelerate-e2a2fda17-20260801T120000Z",
    measurements=None,
    not_measured=None,
    scope=None,
    provenance=None,
    provenance_gaps=None,
):
    scope = scope or {
        "ops": ["GEMM"],
        "arms": ["eigen", "accelerate"],
        "scalars": ["f64"],
        "threads": [1],
        "shape_groups": {"GEMM": ["small"]},
        "ops_toml_sha256": DUMMY_SHA256,
    }
    gaps = provenance_gaps if provenance_gaps is not None else [
        {"field": "/provenance/cpu/frequency_governor", "reason": "macOS exposes no user-visible CPU frequency governor"},
        {"field": "/provenance/numa/policy", "reason": "Non-NUMA platform; no allocation policy is exposed"},
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "eigen-benchmark-comparison-result",
        "run_id": run_id,
        "partial": True,
        "provenance": provenance or make_provenance(),
        "scope": scope,
        "measurements": list(measurements or []),
        "not_measured": list(not_measured or []),
        "provenance_gaps": gaps,
    }


# --------------------------------------------------------------------------
# Synthetic merged documents (CONTRACTS.md section 5)
# --------------------------------------------------------------------------


# Delegated, not re-implemented: the tests' notion of the plot abscissa must be
# the implementation's, or a fixture's expected value can diverge from what the
# renderer computes. The local copy this replaced also lacked the non-positive
# guard, so the two disagreed for a zero dimension.
from _common import size_key  # noqa: E402


def merged_arm_measured(gflops, mad, flops, reps=10, run_id="run-a"):
    rate, rate_mad = gflops * 1e9, mad * 1e9
    seconds = flops / rate
    return {
        "state": "measured",
        "gflops": gflops,
        "gflops_mad": mad,
        "time_s": seconds,
        "time_mad_s": flops * rate_mad / (rate * rate),
        "reps": reps,
        "cv": (mad * 1.4826 / gflops) if gflops else None,
        "run_id": run_id,
    }


def merged_arm_absent(reason, detail):
    return {"state": "not_measured", "reason": reason, "detail": detail}


def make_cell(
    op="GEMM",
    op_family="blas3",
    scalar="f64",
    shape=None,
    shape_dims=("m", "n", "k"),
    shape_group="medium",
    arms=None,
    flops=None,
    nominal=False,
    ratio=None,
    ratio_state="ok",
    config_id=CONFIG_ID,
):
    shape = dict(shape or {"m": 512, "n": 512, "k": 512})
    shape_dims = list(shape_dims)
    flops = flops if flops is not None else gemm_flops(*[shape[d] for d in shape_dims])
    return {
        "config_id": config_id,
        "op": op,
        "op_family": op_family,
        "scalar": scalar,
        "shape": shape,
        "shape_dims": shape_dims,
        "shape_group": shape_group,
        "size_key": size_key(shape),
        "flops_per_iteration": flops,
        "flops_nominal": nominal,
        "arms": arms or {},
        "ratio": ratio,
        "ratio_state": ratio_state,
    }


def make_merged(cells, baseline="accelerate", configs=None, arms=None, coverage=None, conflicts=None):
    cells = list(cells)
    configs = configs or {
        CONFIG_ID: {
            "machine_config_id": "m4pro",
            "isa_target": "aarch64-neon",
            "compiler": "AppleClang 17.0.0",
            "cxx_flags": ["-O3", "-DNDEBUG"],
            "eigen_commit": EIGEN_COMMIT,
            "eigen_commit_short": EIGEN_COMMIT_SHORT,
            "eigen_dirty": False,
            "threads": 1,
            "cpu_model": "Apple M4 Pro",
            "provenance_refs": ["m4pro-neon-accelerate-e2a2fda17-20260801T120000Z"],
        }
    }
    arms = arms or {
        "eigen": {"kind": "eigen", "library_name": "Eigen", "library_version": "5.0.1-master"},
        "accelerate": {"kind": "reference", "library_name": "Apple Accelerate", "library_version": "macOS 15.6"},
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "eigen-benchmark-comparison-merged",
        "generated_utc": "2026-08-20T00:00:00Z",
        "ops_toml_sha256": DUMMY_SHA256,
        "reducer_version": "1.0.0",
        "baseline": baseline,
        "configs": configs,
        "arms": arms,
        "cells": cells,
        "coverage": coverage if coverage is not None else coverage_of(cells, configs, arms),
        "conflicts": list(conflicts or []),
    }


def coverage_of(cells, configs, arms):
    ops, scalars = {}, {}
    measured = not_measured = 0
    for cell in cells:
        op = ops.setdefault(cell["op"], {"measured": 0, "not_measured": 0, "unaccounted": 0, "arms": sorted(arms)})
        scalar = scalars.setdefault(cell["scalar"], {"measured": 0, "not_measured": 0})
        for entry in cell["arms"].values():
            if entry["state"] == "measured":
                op["measured"] += 1
                scalar["measured"] += 1
                measured += 1
            else:
                op["not_measured"] += 1
                scalar["not_measured"] += 1
                not_measured += 1
    return {
        "configs": sorted(configs),
        "ops": ops,
        "scalars": scalars,
        "totals": {"measured": measured, "not_measured": not_measured, "unaccounted": 0},
        "missing_configs": [],
    }


# --------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------


def write_json(path, document, sort_keys=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=sort_keys) + "\n")
    return path


def read_json(path):
    return json.loads(Path(path).read_text())


def deep_copy(document):
    return json.loads(json.dumps(document))


def pointer_delete(document, pointer):
    """Delete the JSON-Pointer target, returning False when it is absent."""
    parts = [p.replace("~1", "/").replace("~0", "~") for p in pointer.split("/")[1:]]
    node = document
    for part in parts[:-1]:
        if isinstance(node, list):
            node = node[int(part)]
        elif part in node:
            node = node[part]
        else:
            return False
    last = parts[-1]
    if isinstance(node, dict) and last in node:
        del node[last]
        return True
    return False


def pointer_get(document, pointer):
    parts = [p.replace("~1", "/").replace("~0", "~") for p in pointer.split("/")[1:]]
    node = document
    for part in parts:
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def required_pointers(schema, defs, node=None, prefix="", seen=None):
    """Every '/'-pointer that the schema marks required, following $ref once."""
    node = schema if node is None else node
    seen = seen if seen is not None else set()
    pointers = []
    if "$ref" in node:
        target = node["$ref"].split("/")[-1]
        if target in seen:
            return pointers
        return required_pointers(schema, defs, defs[target], prefix, seen | {target})
    for name in node.get("required", []):
        child = node.get("properties", {}).get(name)
        pointer = f"{prefix}/{name}"
        pointers.append(pointer)
        if isinstance(child, dict):
            pointers.extend(required_pointers(schema, defs, child, pointer, seen))
    return pointers


def has_finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def find_nonfinite(document, path="$"):
    """Every path in `document` holding NaN or an infinity."""
    bad = []
    if isinstance(document, dict):
        for key, value in document.items():
            bad.extend(find_nonfinite(value, f"{path}.{key}"))
    elif isinstance(document, list):
        for index, value in enumerate(document):
            bad.extend(find_nonfinite(value, f"{path}[{index}]"))
    elif isinstance(document, float) and not math.isfinite(document):
        bad.append(path)
    return bad


def stub_executable(tmp_path, names, raw_json, trace_path=None):
    """Install the canned-output stub under every plausible executable name.

    `run.py` derives the executable name from `ops.toml` and the configured
    reference arm; the contract does not fix the spelling, so the stub is
    installed under every name the tests can predict and the trace file records
    which one was actually invoked.
    """
    bindir = Path(tmp_path)
    bindir.mkdir(parents=True, exist_ok=True)
    source = FIXTURES / "stub_benchmark.py"
    payload = bindir / "canned_benchmark_output.json"
    payload.write_text(json.dumps(raw_json, indent=2))
    installed = []
    for name in names:
        target = bindir / name
        shutil.copyfile(source, target)
        target.chmod(0o755)
        installed.append(target)
    return installed, payload
