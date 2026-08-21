#!/usr/bin/env python3
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

"""Measurement driver for the cross-library benchmark comparison harness.

Configures and builds the comparison targets for one machine profile, runs them
under an explicitly controlled thread and affinity environment, and distills the
Google Benchmark JSON into one result file per (machine, ISA target, reference
arm, thread count) that validates against ``result_schema.json`` before it is
written.

The module is split so that everything except configure/build/execute is a pure
function of plain data: the machine profile parser, the operation registry, the
cell planner, the benchmark-name parser, the thread-environment builder, the
pinning planner, the identifier and path constructors, the statistics, the
Google Benchmark distiller, and the provenance assembler are all importable and
testable without a compiler, a benchmark binary, or a particular host.

See ``benchmarks/comparison/CONTRACTS.md``; it is the source of truth for the
name grammar, the CLI surface, the exit codes, and the machine profile shape.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

# This script is run directly and is also imported by file path by the test
# suite, so its own directory is not reliably on sys.path when _common is needed.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _common import (  # noqa: E402
    OpsRegistry,
    PipelineError,
    UsageErrorArgumentParser,
    config_id_for,
    format_shape_suffix,
    is_arm_key,
    load_ops_registry,
    parse_benchmark_name,
    split_list,
)

HARNESS_NAME = "eigen-comparison-harness"

# Bump on any observable change in run.py's behaviour (CONTRACTS.md section 0).
HARNESS_VERSION = "1.0.0"

RESULT_SCHEMA_VERSION = "1.0.0"
RESULT_KIND = "eigen-benchmark-comparison-result"
MACHINE_SCHEMA_VERSION = "1.0.0"

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_CONFIG = 2
EXIT_DIRTY = 3
EXIT_NOISY = 4
EXIT_BUILD = 5
EXIT_RUNTIME = 6
EXIT_SCHEMA = 7


class HarnessError(PipelineError):
    """An error that maps onto one of the documented exit codes."""

    def __init__(self, message: str, exit_code: int = EXIT_CONFIG):
        super().__init__(message, exit_code)


# ---------------------------------------------------------------------------
# 1. The benchmark name grammar and the operation registry
# ---------------------------------------------------------------------------
#
# Both live in _common.py: run.py writes the result files reduce.py reads, so a
# name or a flop formula the two stages read differently is a pipeline that
# breaks after a valid file has already been written. Imported above.

_MACHINE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


# ---------------------------------------------------------------------------
# 3. The machine profile (CONTRACTS.md section 7)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmProfile:
    """One reference-library arm as the machine profile describes it."""

    key: str
    library_name: str
    cmake_options: tuple[str, ...] = ()
    version_fallback: str | None = None
    thread_env: Mapping[str, str] = field(default_factory=dict)
    threading_model: str | None = None
    interface: str | None = None
    provides: tuple[str, ...] = ()
    library_path: str | None = None
    available: bool = True
    unavailable_reason: str | None = None


@dataclass(frozen=True)
class PinningConfig:
    tool: str = "none"
    cpu_list: str | None = None
    numa_node: int | None = None
    membind: bool = True
    omp_proc_bind: str | None = None
    omp_places: str | None = None
    unavailable_reason: str | None = None


@dataclass(frozen=True)
class MachineProfile:
    """Everything ``run.py`` needs to know about the host it is measuring on."""

    id: str
    display_name: str
    cpu_model: str
    arch: str
    isa_targets: tuple[str, ...]
    default_isa_target: str
    sockets: int | None
    cores_per_socket: int | None
    threads_per_core: int | None
    smt_enabled: bool | None
    numa_nodes: int | None
    max_load_avg: float
    frequency_governor: str | None
    arms: Mapping[str, ArmProfile]
    isa: Mapping[str, Mapping[str, Any]]
    pinning: PinningConfig
    generator: str | None
    build_type: str
    cxx_standard: int
    locally_verified: bool
    notes: str | None
    sha256: str | None
    microarchitecture: str | None = None

    def isa_options(self, isa_target: str) -> tuple[str, ...]:
        return tuple(self.isa.get(isa_target, {}).get("cmake_options", ()))

    def isa_flags(self, isa_target: str) -> tuple[str, ...]:
        return tuple(self.isa.get(isa_target, {}).get("flags", ()))

    def reference_arms(self) -> tuple[str, ...]:
        return tuple(self.arms)


def parse_machine_profile(data: Mapping[str, Any], stem: str, sha256: str | None = None) -> MachineProfile:
    """Validate and normalise a machine profile mapping. Pure."""
    required = (
        "schema_version",
        "id",
        "display_name",
        "cpu_model",
        "arch",
        "isa_targets",
        "default_isa_target",
        "max_load_avg",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise HarnessError(f"machine profile {stem!r} is missing required key(s): {', '.join(missing)}")

    version = str(data["schema_version"])
    if version.split(".")[0] != MACHINE_SCHEMA_VERSION.split(".")[0]:
        raise HarnessError(
            f"machine profile {stem!r} declares schema_version {version!r}; "
            f"this harness understands {MACHINE_SCHEMA_VERSION.split('.')[0]}.x"
        )

    machine_id = str(data["id"])
    if machine_id != stem:
        raise HarnessError(f"machine profile id {machine_id!r} does not match its filename stem {stem!r}")
    if not _MACHINE_ID_RE.match(machine_id):
        raise HarnessError(f"machine profile id {machine_id!r} does not match ^[a-z0-9][a-z0-9._-]*$")

    isa_targets = tuple(str(target) for target in data["isa_targets"])
    if not isa_targets:
        raise HarnessError(f"machine profile {machine_id!r} lists no isa_targets")
    default_isa = str(data["default_isa_target"])
    if default_isa not in isa_targets:
        raise HarnessError(
            f"machine profile {machine_id!r} default_isa_target {default_isa!r} is not in isa_targets"
        )

    governor = data.get("frequency", {}).get("governor", "")
    governor = str(governor) if governor else None

    pinning_raw = data.get("pinning", {})
    pinning = PinningConfig(
        tool=str(pinning_raw.get("tool", "none")),
        cpu_list=(str(pinning_raw["cpu_list"]) if pinning_raw.get("cpu_list") else None),
        numa_node=(int(pinning_raw["numa_node"]) if pinning_raw.get("numa_node") is not None else None),
        membind=bool(pinning_raw.get("membind", True)),
        omp_proc_bind=(str(pinning_raw["omp_proc_bind"]) if pinning_raw.get("omp_proc_bind") else None),
        omp_places=(str(pinning_raw["omp_places"]) if pinning_raw.get("omp_places") else None),
        unavailable_reason=(
            str(pinning_raw["unavailable_reason"]) if pinning_raw.get("unavailable_reason") else None
        ),
    )
    if pinning.tool not in ("none", "taskset", "numactl"):
        raise HarnessError(f"machine profile {machine_id!r} names unknown pinning tool {pinning.tool!r}")

    arms: dict[str, ArmProfile] = {}
    for key, raw in data.get("arms", {}).items():
        if not is_arm_key(key):
            raise HarnessError(f"machine profile {machine_id!r} arm key {key!r} does not match ^[a-z][a-z0-9_]*$")
        if key == "eigen":
            raise HarnessError(
                f"machine profile {machine_id!r} declares an 'eigen' arm; the Eigen arm is implicit "
                "and always present"
            )
        arms[key] = ArmProfile(
            key=key,
            library_name=str(raw.get("library_name", key)),
            cmake_options=tuple(str(option) for option in raw.get("cmake_options", ())),
            version_fallback=(str(raw["version_fallback"]) if raw.get("version_fallback") else None),
            thread_env={str(name): str(value) for name, value in raw.get("thread_env", {}).items()},
            threading_model=(str(raw["threading_model"]) if raw.get("threading_model") else None),
            interface=(str(raw["interface"]) if raw.get("interface") else None),
            provides=tuple(str(item) for item in raw.get("provides", ())),
            library_path=(str(raw["library_path"]) if raw.get("library_path") else None),
            available=bool(raw.get("available", True)),
            unavailable_reason=(str(raw["unavailable_reason"]) if raw.get("unavailable_reason") else None),
        )

    build = data.get("build", {})
    return MachineProfile(
        id=machine_id,
        display_name=str(data["display_name"]),
        cpu_model=str(data["cpu_model"]),
        arch=str(data["arch"]),
        isa_targets=isa_targets,
        default_isa_target=default_isa,
        sockets=_optional_int(data.get("sockets")),
        cores_per_socket=_optional_int(data.get("cores_per_socket")),
        threads_per_core=_optional_int(data.get("threads_per_core")),
        smt_enabled=(bool(data["smt_enabled"]) if data.get("smt_enabled") is not None else None),
        numa_nodes=_optional_int(data.get("numa_nodes")),
        max_load_avg=float(data["max_load_avg"]),
        frequency_governor=governor,
        arms=arms,
        isa={str(name): dict(value) for name, value in data.get("isa", {}).items()},
        pinning=pinning,
        generator=(str(build["generator"]) if build.get("generator") else None),
        build_type=str(build.get("build_type", "Release")),
        cxx_standard=int(build.get("cxx_standard", 17)),
        locally_verified=bool(data.get("locally_verified", False)),
        notes=(str(data["notes"]) if data.get("notes") else None),
        sha256=sha256,
        microarchitecture=(str(data["microarchitecture"]) if data.get("microarchitecture") else None),
    )


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def machine_profile_path(machines_dir: Path, machine_id: str) -> Path:
    return machines_dir / f"{machine_id}.toml"


def load_machine_profile(path: Path) -> MachineProfile:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise HarnessError(f"cannot read machine profile {path}: {exc}") from exc
    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise HarnessError(f"invalid machine profile {path}: {exc}") from exc
    return parse_machine_profile(data, path.stem, hashlib.sha256(raw).hexdigest())


# ---------------------------------------------------------------------------
# 4. Thread control
# ---------------------------------------------------------------------------

# Every vendor thread-count variable this harness knows about. All of them are
# set on every run, including the ones for libraries that are not linked: an
# unset OPENBLAS_NUM_THREADS silently races a multithreaded BLAS against a
# single-threaded Eigen, and "the vendor is not linked today" is not a property
# the harness can rely on the next time a build option changes.
THREAD_COUNT_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NVPL_BLAS_NUM_THREADS",
    "ARMPL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "ACCELERATE_MAXIMUM_THREADS",
    "BLAS_NUM_THREADS",
    # Inert for Eigen itself: nbThreads() reads omp_get_max_threads(), never an
    # environment variable. result_schema.json asks for it, and a reader must not
    # take its presence as evidence that the Eigen arm was capped -- that fact is
    # provenance.threading.eigen_nb_threads, reported by the binary.
    "EIGEN_NB_THREADS",
)

# Set alongside the counts so a runtime cannot re-expand the pool it was given.
THREAD_FIXED_ENV: Mapping[str, str] = {
    "OMP_DYNAMIC": "FALSE",
    "MKL_DYNAMIC": "FALSE",
}


def build_thread_env(
    threads: int,
    *,
    arm: ArmProfile | None = None,
    pinning_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Construct the thread-control environment for one run. Pure.

    Returns exactly the variables the harness sets, which is also exactly what
    ``provenance.threading.env`` records: an absent variable is meaningfully
    different from one set to its default, so nothing is recorded that was not
    set and nothing is set that is not recorded.

    A machine profile's per-arm ``thread_env`` is applied last. Values may
    contain ``{threads}``, which is substituted with the requested thread count;
    a literal value is used verbatim, which is how a library that must stay
    sequential regardless of the run's thread count is expressed.
    """
    if threads < 1:
        raise ValueError(f"thread count must be at least 1, got {threads}")
    env = {name: str(threads) for name in THREAD_COUNT_ENV_VARS}
    env.update(THREAD_FIXED_ENV)
    if pinning_env:
        env.update({str(k): str(v) for k, v in pinning_env.items()})
    if arm is not None:
        for name, value in arm.thread_env.items():
            env[str(name)] = str(value).format(threads=threads)
    return dict(sorted(env.items()))


# ---------------------------------------------------------------------------
# 5. CPU pinning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PinningPlan:
    """How (or whether) the benchmark process will be bound to CPUs."""

    tool: str
    command_prefix: tuple[str, ...] = ()
    description: str | None = None
    numa_policy: str | None = None
    node_of_run: int | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    applied: bool = False
    unavailable_reason: str | None = None


def plan_pinning(
    machine: MachineProfile,
    *,
    system: str | None = None,
    tool_exists: Callable[[str], bool] | None = None,
) -> PinningPlan:
    """Decide the affinity wrapper for a run. Pure given ``system``/``tool_exists``.

    Pinning is a correctness property on hybrid performance/efficiency-core parts
    and on NUMA hosts: an unpinned run there measures whichever core the
    scheduler happened to pick. Where the platform provides no affinity control
    the plan says so, and ``run.py`` writes a ``provenance_gaps`` entry rather
    than reporting a binding that was never applied.
    """
    system = platform.system() if system is None else system
    exists = (lambda name: shutil.which(name) is not None) if tool_exists is None else tool_exists
    config = machine.pinning

    omp_env: dict[str, str] = {}
    if config.omp_proc_bind:
        omp_env["OMP_PROC_BIND"] = config.omp_proc_bind
    if config.omp_places:
        omp_env["OMP_PLACES"] = config.omp_places

    def unavailable(reason: str | None) -> PinningPlan:
        """No affinity was applied; the OpenMP hints still stand, and why is recorded."""
        return PinningPlan(tool="none", env=omp_env, applied=False, unavailable_reason=reason)

    if config.tool == "none":
        reason = config.unavailable_reason
        if reason is None and (machine.numa_nodes or 1) > 1:
            reason = (
                f"machine profile {machine.id!r} reports {machine.numa_nodes} NUMA nodes but requests no "
                "pinning tool; the run was not bound to a node"
            )
        return unavailable(reason)

    if system == "Darwin":
        return unavailable(
            f"machine profile requests pinning with {config.tool!r}, but macOS exposes no CPU-affinity "
            "API equivalent to taskset or numactl; thread placement was left to the scheduler"
        )
    if system != "Linux":
        return unavailable(
            f"machine profile requests pinning with {config.tool!r}, which this harness only applies on "
            f"Linux; the run was made on {system}"
        )

    if not exists(config.tool):
        return unavailable(f"pinning tool {config.tool!r} is not installed on this host")

    if config.tool == "taskset":
        if not config.cpu_list:
            return unavailable("machine profile requests taskset but sets no [pinning].cpu_list")
        return PinningPlan(
            tool="taskset",
            command_prefix=("taskset", "-c", config.cpu_list),
            description=f"taskset -c {config.cpu_list}",
            numa_policy=None,
            env=omp_env,
            applied=True,
        )

    # numactl
    if config.numa_node is None:
        return unavailable("machine profile requests numactl but sets no [pinning].numa_node")
    prefix = ["numactl", f"--cpunodebind={config.numa_node}"]
    policy = f"bind={config.numa_node}"
    if config.membind:
        prefix.append(f"--membind={config.numa_node}")
        policy = f"cpubind={config.numa_node},membind={config.numa_node}"
    if config.cpu_list:
        prefix = ["numactl", f"--physcpubind={config.cpu_list}"]
        if config.membind:
            prefix.append(f"--membind={config.numa_node}")
        policy = f"physcpubind={config.cpu_list},membind={config.numa_node}"
    return PinningPlan(
        tool="numactl",
        command_prefix=tuple(prefix),
        description=" ".join(prefix),
        numa_policy=policy,
        node_of_run=config.numa_node,
        env=omp_env,
        applied=True,
    )


# ---------------------------------------------------------------------------
# 6. Identifiers and paths
# ---------------------------------------------------------------------------


def compact_timestamp(moment: datetime) -> str:
    """``20260820T160111Z`` from an aware datetime."""
    return moment.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def rfc3339_timestamp(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_run_id(
    machine_id: str,
    isa_target: str,
    arm: str,
    commit_short: str,
    timestamp: str,
    threads: int = 1,
) -> str:
    """Build the run identifier.

    Follows the convention in ``result_schema.json``:
    ``<machine>-<isa_target>-<arm>-<commit_short>-<compact UTC timestamp>``. A
    non-default thread count is appended, because two thread counts measured in
    the same second would otherwise collide on an identifier the schema requires
    to be globally unique.
    """
    parts = [machine_id, isa_target, arm, commit_short, timestamp]
    run_id = "-".join(part for part in parts if part)
    if threads != 1:
        run_id = f"{run_id}-t{threads}"
    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*$", run_id):
        raise ValueError(f"constructed run_id {run_id!r} violates the schema pattern")
    return run_id


def make_output_path(results_dir: Path, machine_id: str, commit_short: str, run_id: str) -> Path:
    """``<results-dir>/<machine>/<eigen_short>/<run_id>.json``."""
    return Path(results_dir) / machine_id / commit_short / f"{run_id}.json"


def make_invalid_output_path(path: Path) -> Path:
    """Where a file that failed schema validation is written instead."""
    name = path.name
    if name.endswith(".json"):
        name = name[: -len(".json")]
    return path.with_name(f"{name}.invalid.json")


def make_config_id(
    machine_id: str,
    isa_target: str,
    compiler_id: str,
    compiler_version: str,
    commit_short: str,
    threads: int,
) -> str:
    """The merge key ``reduce.py`` will derive, from this stage's loose fields.

    A thin adapter over the one implementation in ``_common``: the two used to
    compute the compiler segment differently, so ``run.py`` and ``reduce.py``
    could name the same configuration two ways.
    """
    return config_id_for(
        {
            "machine_config_id": machine_id,
            "toolchain": {
                "isa_target": isa_target,
                "compiler_id": compiler_id,
                "compiler_version": compiler_version,
            },
            "eigen": {"commit_short": commit_short},
        },
        threads,
    )


def make_build_dir(base: Path, isa_target: str, arm: str) -> Path:
    """One build tree per (ISA target, reference arm); they differ in compile flags."""
    return Path(base) / f"{isa_target}__{arm}"


# ---------------------------------------------------------------------------
# 7. Cell planning and filter construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One planned measurement: an (op, arm, scalar, shape, threads) point."""

    op: str
    arm: str
    scalar: str
    shape_dims: tuple[str, ...]
    values: tuple[int, ...]
    group: str | None
    threads: int = 1

    @property
    def shape(self) -> dict[str, int]:
        return dict(zip(self.shape_dims, self.values))

    @property
    def name(self) -> str:
        base = f"{self.op}/{self.arm}/{self.scalar}/{format_shape_suffix(self.shape_dims, self.values)}"
        return base if self.threads == 1 else f"{base}/threads:{self.threads}"

    @property
    def key(self) -> tuple:
        return (self.op, self.arm, self.scalar, self.values, self.threads)


def resolve_groups(registry: OpsRegistry, op_key: str, requested: Sequence[str] | None) -> tuple[str, ...]:
    if not requested:
        return registry.default_groups(op_key)
    known = set(registry.group_names(op_key))
    unknown = [name for name in requested if name not in known]
    if unknown:
        raise HarnessError(
            f"operation {op_key!r} has no shape group(s) {', '.join(sorted(unknown))}; "
            f"known groups: {', '.join(sorted(known))}"
        )
    return tuple(requested)


def plan_cells(
    registry: OpsRegistry,
    *,
    ops: Sequence[str],
    arms: Sequence[str],
    scalars: Sequence[str],
    groups: Sequence[str] | None = None,
    threads: int = 1,
) -> list[Cell]:
    """Cartesian product of the selection, restricted to what the registry allows. Pure."""
    cells: list[Cell] = []
    for op_key in ops:
        op = registry.op(op_key)
        dims = registry.shape_dims(op_key)
        selected = resolve_groups(registry, op_key, groups)
        points = registry.shape_points(op_key, selected)
        supported = set(op.get("scalars", ()))
        for arm in arms:
            for scalar in scalars:
                if scalar not in supported:
                    continue
                for group_name, values in points:
                    cells.append(
                        Cell(
                            op=op_key,
                            arm=arm,
                            scalar=scalar,
                            shape_dims=dims,
                            values=values,
                            group=group_name,
                            threads=threads,
                        )
                    )
    return cells


# A filter that enumerated every shape would be precise but unboundedly long.
# Past this many distinct shapes for one op the filter falls back to selecting
# the op/arm/scalar prefix and the planned shapes are enforced when the rows are
# distilled, which is where an unplanned row would be dropped anyway.
MAX_SHAPE_ALTERNATIVES = 256


def apply_extra_filter(cells: Sequence[Cell], regex: str | None) -> tuple[list[Cell], list[Cell]]:
    """Split planned cells on a user ``--filter``. Pure.

    The user filter is ANDed with the generated one by narrowing the plan before
    the generated filter is built, rather than by combining two regexes: POSIX
    ERE has no conjunction, and narrowing also stops the binary from spending
    time on points the user excluded.
    """
    if not regex:
        return list(cells), []
    pattern = re.compile(regex)
    kept = [cell for cell in cells if pattern.search(cell.name)]
    dropped = [cell for cell in cells if not pattern.search(cell.name)]
    return kept, dropped


def make_benchmark_filter(
    cells: Sequence[Cell],
    *,
    max_shape_alternatives: int = MAX_SHAPE_ALTERNATIVES,
) -> str:
    """Build the ``--benchmark_filter`` POSIX ERE selecting the planned cells. Pure."""
    if not cells:
        return "$^"
    by_op: dict[str, dict[str, set]] = {}
    for cell in cells:
        entry = by_op.setdefault(cell.op, {"arms": set(), "scalars": set(), "shapes": set()})
        entry["arms"].add(cell.arm)
        entry["scalars"].add(cell.scalar)
        entry["shapes"].add(format_shape_suffix(cell.shape_dims, cell.values))

    alternatives = []
    for op_key in sorted(by_op):
        entry = by_op[op_key]
        arms = "|".join(sorted(entry["arms"]))
        scalars = "|".join(sorted(entry["scalars"]))
        shapes = sorted(entry["shapes"])
        if len(shapes) <= max_shape_alternatives:
            shape_part = "(" + "|".join(shapes) + ")"
        else:
            # Past the cap the filter selects the op/arm/scalar prefix and the
            # shape restriction is enforced when the rows are distilled.
            shape_part = "([a-z][a-z0-9_]*:[0-9]+/)*[a-z][a-z0-9_]*:[0-9]+"
        alternatives.append(f"{op_key}/({arms})/({scalars})/{shape_part}")

    return "^(" + "|".join(alternatives) + ")(/threads:[0-9]+)?$"


# ---------------------------------------------------------------------------
# 8. Statistics and Google Benchmark distillation
# ---------------------------------------------------------------------------

TIME_UNIT_SECONDS: Mapping[str, float] = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}


def median_absolute_deviation(values: Sequence[float]) -> float:
    """Median of the absolute deviations from the median. Zero for one sample."""
    if not values:
        raise ValueError("median_absolute_deviation of an empty sample")
    if len(values) == 1:
        return 0.0
    centre = statistics.median(values)
    return float(statistics.median([abs(value - centre) for value in values]))


def summarize(values: Sequence[float]) -> dict:
    """The ``$defs/stat`` summary of one measured quantity.

    Median plus MAD are the reported pair, per ``.agents/benchmarking.md``:
    compare medians and a dispersion measure, never the best run. Mean, stddev
    and CV are retained for cross-checking; min and max bound the sample so a
    reader can see how wide it was.
    """
    if not values:
        raise ValueError("summarize of an empty sample")
    ordered = list(values)
    mean = float(statistics.fmean(ordered))
    stddev = float(statistics.stdev(ordered)) if len(ordered) > 1 else 0.0
    return {
        "median": float(statistics.median(ordered)),
        "mad": median_absolute_deviation(ordered),
        "mean": mean,
        "stddev": stddev,
        "cv": (stddev / mean) if mean else None,
        "min": float(min(ordered)),
        "max": float(max(ordered)),
        "count": len(ordered),
    }


@dataclass
class Distillation:
    measurements: list[dict] = field(default_factory=list)
    seen_keys: set = field(default_factory=set)
    errors: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    context: dict = field(default_factory=dict)


def _row_threads(run_name: str, parsed: Mapping[str, Any], configured: int | None) -> int:
    """The thread count a distilled row is recorded under.

    The configured count wins, because the harness controls threading through the
    library environment rather than Google Benchmark's ``->Threads(n)``. A name
    that spells the field out anyway must agree with it.
    """
    if configured is None:
        return int(parsed["threads"])
    if "/threads:" in run_name and int(parsed["threads"]) != int(configured):
        raise HarnessError(
            f"{run_name!r} registers {parsed['threads']} Google Benchmark thread(s) but the run was "
            f"configured for {configured}; the two describe different measurements"
        )
    return int(configured)


def distill_benchmark_json(
    document: Mapping[str, Any],
    registry: OpsRegistry,
    *,
    planned: Sequence[Cell] = (),
    flop_counter_name: str | None = None,
    threads: int | None = None,
) -> Distillation:
    """Turn one Google Benchmark JSON document into schema measurement rows. Pure.

    Only ``run_type == "iteration"`` rows are read: the vendor's own aggregate
    rows are discarded so that the dispersion statistic is identical across every
    contribution regardless of which Google Benchmark version produced it
    (CONTRACTS.md section 1.4).

    ``threads`` is the run's thread count. It comes from the run configuration,
    never from the benchmark name: ``/threads:N`` is Google Benchmark's OWN
    multi-threaded registration form (``->Threads(n)``), which these benchmarks
    do not use — the harness controls threading through the library environment
    instead (CONTRACTS.md section 7). Keying the plan on the parsed name would
    therefore drop every row of a ``--threads N`` run for N > 1, after measuring
    the whole grid. A name that does carry the field must still agree with the
    configuration, or the two are describing different runs.
    """
    counter_name = flop_counter_name or registry.flop_counter_name
    result = Distillation(context=dict(document.get("context", {})))
    groups_by_point = {(cell.op, cell.scalar, cell.values): cell.group for cell in planned}
    planned_points = {(cell.op, cell.scalar, cell.values) for cell in planned} if planned else None

    rows_by_name: dict[str, list[Mapping[str, Any]]] = {}
    for row in document.get("benchmarks", []):
        if row.get("run_type") not in (None, "iteration"):
            continue
        if row.get("error_occurred"):
            failed_name = str(row.get("run_name") or row.get("name", ""))
            failure = {"name": failed_name, "message": str(row.get("error_message", ""))}
            # A benchmark that calls SkipWithError still leaves the process exit
            # status 0, so this row is the only evidence that the cell produced
            # nothing trustworthy. Attribute it to its cell key so the caller can
            # state "runtime_error" rather than inheriting the generic
            # "not_implemented" default, which would describe a wrong-but-fast
            # kernel as a benchmark nobody had written yet.
            try:
                parsed_failure = parse_benchmark_name(failed_name)
            except ValueError:
                result.warnings.append(f"an errored row carries the unparseable name {failed_name!r}")
            else:
                failure["op"] = parsed_failure["op"]
                failure["arm"] = parsed_failure["arm"]
                failure["scalar"] = parsed_failure["scalar"]
                failure["shape"] = dict(parsed_failure["shape"])
                failure["threads"] = _row_threads(failed_name, parsed_failure, threads)
                failure["key"] = (
                    parsed_failure["op"],
                    parsed_failure["arm"],
                    parsed_failure["scalar"],
                    tuple(parsed_failure["shape"][dim] for dim in parsed_failure["shape_dims"]),
                    failure["threads"],
                )
            result.errors.append(failure)
            continue
        run_name = row.get("run_name")
        if not run_name:
            result.warnings.append("a benchmark row carries no 'run_name'; skipped")
            continue
        rows_by_name.setdefault(str(run_name), []).append(row)

    for run_name in sorted(rows_by_name):
        rows = sorted(rows_by_name[run_name], key=lambda entry: entry.get("repetition_index", 0))
        try:
            parsed = parse_benchmark_name(run_name)
        except ValueError as exc:
            raise HarnessError(str(exc)) from exc

        op_key = parsed["op"]
        if op_key not in registry.ops:
            result.warnings.append(f"unknown_op: {run_name!r} names {op_key!r}, absent from ops.toml; skipped")
            continue
        op = registry.op(op_key)
        if parsed["scalar"] not in op.get("scalars", ()):
            raise HarnessError(
                f"{run_name!r} uses scalar {parsed['scalar']!r}, which ops.toml does not list for {op_key!r}"
            )
        expected_dims = registry.shape_dims(op_key)
        if tuple(parsed["shape_dims"]) != expected_dims:
            raise HarnessError(
                f"{run_name!r} carries dimensions {tuple(parsed['shape_dims'])!r} but ops.toml declares "
                f"{expected_dims!r} for {op_key!r}; the C++ registration and the registry have diverged"
            )

        values = tuple(parsed["shape"][dim] for dim in expected_dims)
        point = (op_key, parsed["scalar"], values)
        if planned_points is not None and point not in planned_points:
            continue
        row_threads = _row_threads(run_name, parsed, threads)

        unit = str(rows[0].get("time_unit", "ns"))
        if unit not in TIME_UNIT_SECONDS:
            raise HarnessError(f"{run_name!r} reports unknown time unit {unit!r}")
        scale = TIME_UNIT_SECONDS[unit]

        flops = registry.flops_per_iteration(op_key, parsed["scalar"], parsed["shape"])
        if not flops > 0:
            result.warnings.append(
                f"{run_name!r} has a non-positive flop count ({flops}); the row cannot be recorded"
            )
            continue

        real_times = [float(row["real_time"]) * scale for row in rows]
        cpu_times = [float(row.get("cpu_time", row["real_time"])) * scale for row in rows]
        # The published rate is flops over WALL time, computed here rather than taken
        # from the binary's counter. Google Benchmark's kIsIterationInvariantRate
        # counters divide by CPU time while the reported real_time is wall clock, so
        # taking the counter would put two numbers measured against different clocks
        # in one cell. They agree only while a run is single-threaded and
        # undisturbed: a threaded vendor BLAS accumulates CPU time across its
        # workers and the counter would report it as several times slower than it
        # is, and a descheduled run reports a rate faster than anything that
        # happened. Wall clock is also the honest basis for a throughput comparison.
        # ->UseRealTime() would fix the counter's divisor but renames every
        # benchmark to `<name>/real_time`, which the name grammar rejects outright.
        rates: list[float] = []
        counter_missing = False
        for row, real_time in zip(rows, real_times):
            if counter_name not in row:
                counter_missing = True
            rates.append(flops / real_time if real_time > 0 else 0.0)
        if counter_missing:
            result.warnings.append(
                f"{run_name!r} emits no {counter_name!r} counter, so the C++ flop formula could not be "
                "cross-checked against ops.toml for this row"
            )

        # Two independent statements of the same quantity land in this row: `flops`,
        # evaluated from ops.toml's flops.real, and the binary's counter, computed
        # from the C++ helper in bench_common.h. Nothing else compares them, and a
        # drift between the two rescales every published GFLOP/s for this op while
        # leaving the ratio intact -- both arms are wrong identically, so the table
        # looks self-consistent and only the absolute numbers are false. The counter
        # is a rate, so flops_per_iteration is rate * time.
        if not counter_missing and flops > 0:
            for row, cpu_time in zip(rows, cpu_times):
                counter = float(row.get(counter_name, 0.0))
                if cpu_time <= 0 or counter <= 0:
                    continue
                # The counter divides by CPU time (see above), so recover the flop
                # count it was built from using cpu_time, not the wall time the
                # published rate uses.
                implied = counter * cpu_time
                # Generous slack: this is looking for a formula mismatch (a clean
                # integer factor), not for timer noise.
                if implied > 0 and not (0.8 <= implied / flops <= 1.25):
                    result.warnings.append(
                        f"{run_name!r}: the {counter_name!r} counter implies {implied:.4g} flops per "
                        f"iteration but ops.toml's flops.real gives {flops:.4g} "
                        f"(ratio {implied / flops:.3f}); the C++ flop formula and the registry have "
                        "diverged, so every rate published for this operation is scaled wrongly"
                    )
                    break

        key = (op_key, parsed["arm"], parsed["scalar"], values, row_threads)
        if key in result.seen_keys:
            raise HarnessError(f"duplicate cell key for {run_name!r} within one result file")
        result.seen_keys.add(key)

        result.measurements.append(
            {
                "name": run_name,
                "op": op_key,
                "arm": parsed["arm"],
                "scalar": parsed["scalar"],
                "shape": {dim: parsed["shape"][dim] for dim in expected_dims},
                "shape_dims": list(expected_dims),
                "shape_group": groups_by_point.get(point),
                "threads": row_threads,
                "repetitions": len(rows),
                "iterations_per_repetition": [int(row.get("iterations", 1)) for row in rows],
                "flops_per_iteration": flops,
                "flops_nominal": registry.flops_nominal(op_key),
                "samples": {
                    "real_time_s": real_times,
                    "cpu_time_s": cpu_times,
                    "flop_rate": rates,
                },
                "stats": {
                    "real_time_s": summarize(real_times),
                    "cpu_time_s": summarize(cpu_times),
                    "flop_rate": summarize(rates),
                },
                "source_time_unit": unit,
                "validated": None,
            }
        )
    return result


def _reference_family_missing(reference: Mapping[str, Any], provides: Sequence[str]) -> bool:
    """Whether the linked library lacks the interface family this op's reference needs.

    `provides` comes from the vendor table via vendor_info.json, so it describes the
    library that was actually linked rather than what the op wishes for. An empty
    list means the build published nothing, and silence is not evidence of absence,
    so nothing is claimed in that case.
    """
    kind = str(reference.get("kind") or "")
    if not kind or kind == "none" or not provides:
        return False
    return kind not in {str(x) for x in provides}


def diff_not_measured(
    planned: Sequence[Cell],
    measured_keys: Iterable[tuple],
    registry: OpsRegistry,
    *,
    default_reason: str = "not_implemented",
    default_detail: str | None = None,
    provides: Sequence[str] = (),
) -> list[dict]:
    """Explicit negative results for planned cells that produced no timing. Pure.

    A combination in ``scope`` that appears in neither ``measurements`` nor here
    is a harness bug (``unaccounted`` in the reducer), so every planned cell that
    is not measured leaves an entry with a reason.
    """
    measured = set(measured_keys)
    entries: list[dict] = []
    for cell in planned:
        if cell.key in measured:
            continue
        reason, detail = default_reason, default_detail
        # A caller that already knows why the cell is missing (a build failure, a
        # user filter, an unavailable library) states it; only the generic case
        # falls back to what the registry says about the operation.
        if default_reason == "not_implemented":
            op = registry.ops.get(cell.op, {})
            reference = op.get("reference", {})
            if cell.arm != "eigen" and reference.get("kind") == "none":
                reason = "no_reference_equivalent"
                detail = str(reference.get("reason", "")) or None
            elif cell.arm != "eigen" and _reference_family_missing(reference, provides):
                # The op needs an interface family this library does not expose --
                # netlib ships BLAS and CBLAS but no LAPACK, so a ?potrf comparison
                # against it cannot exist. Distinguishing this from "Eigen has no
                # such benchmark" is the difference between "this library cannot do
                # it" and "we did not measure it", which a published page must not
                # blur. reference_routine_absent had a renderer and a schema slot
                # and no producer until here.
                family = str(reference.get("kind"))
                reason = "reference_routine_absent"
                routine = str(reference.get("routine") or "the routine")
                detail = (
                    f"{cell.arm} exposes {', '.join(provides) or 'no declared interface'}, not {family}, "
                    f"so {routine} is unavailable in the library this run linked"
                )
            elif op.get("status") and op.get("status") != "implemented":
                detail = (
                    f"ops.toml marks {cell.op} as status={op.get('status')!r}; no benchmark source registers it"
                )
        entries.append(
            {
                "op": cell.op,
                "arm": cell.arm,
                "scalar": cell.scalar,
                "shape": cell.shape,
                "threads": cell.threads,
                "reason": reason,
                "detail": detail,
            }
        )
    return entries


def collapse_not_measured(entries: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Collapse per-shape entries that cover a whole (op, arm, scalar) grid.

    ``shape: null`` means "the whole grid for this op", which is both what the
    schema documents and what keeps a wholly unimplemented operation from
    writing hundreds of identical rows.
    """
    grouped: dict[tuple, list[Mapping[str, Any]]] = {}
    order: list[tuple] = []
    for entry in entries:
        key = (entry["op"], entry["arm"], entry.get("scalar"), entry["reason"], entry.get("detail"))
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(entry)

    collapsed: list[dict] = []
    for key in order:
        members = grouped[key]
        op, arm, scalar, reason, detail = key
        threads = {member.get("threads") for member in members}
        if len(members) > 1:
            collapsed.append(
                {
                    "op": op,
                    "arm": arm,
                    "scalar": scalar,
                    "shape": None,
                    "threads": threads.pop() if len(threads) == 1 else None,
                    "reason": reason,
                    "detail": detail,
                }
            )
        else:
            collapsed.append(dict(members[0]))
    return collapsed


# ---------------------------------------------------------------------------
# 9. Host probing (the only place that reads the machine it runs on)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HostFacts:
    hostname: str
    system: str
    os_name: str
    os_release: str
    kernel: str | None
    arch: str
    cpu_model: str | None
    logical_cpus: int
    sockets: int | None
    cores_per_socket: int | None
    threads_per_core: int | None
    performance_cores: int | None
    efficiency_cores: int | None
    frequency_governor: str | None
    memory_total_bytes: int | None
    transparent_huge_pages: str | None
    load_avg: tuple[float, float, float] | None


def _sysctl(name: str) -> str | None:
    try:
        out = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, check=False)
    except OSError:
        return None
    value = out.stdout.strip()
    return value if out.returncode == 0 and value else None


def _sysctl_int(name: str) -> int | None:
    value = _sysctl(name)
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _read_text(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def probe_host() -> HostFacts:
    """Collect the host facts the schema's provenance block requires."""
    system = platform.system()
    try:
        load_avg = tuple(float(value) for value in os.getloadavg())  # type: ignore[assignment]
    except (OSError, AttributeError):
        load_avg = None

    if system == "Darwin":
        performance = _sysctl_int("hw.perflevel0.physicalcpu")
        efficiency = _sysctl_int("hw.perflevel1.physicalcpu")
        physical = _sysctl_int("hw.physicalcpu")
        logical = _sysctl_int("hw.logicalcpu") or os.cpu_count() or 1
        return HostFacts(
            hostname=platform.node() or "unknown",
            system=system,
            os_name="macOS",
            os_release=platform.mac_ver()[0] or platform.release(),
            kernel=f"Darwin {platform.release()}",
            arch=platform.machine(),
            cpu_model=_sysctl("machdep.cpu.brand_string"),
            logical_cpus=logical,
            sockets=_sysctl_int("hw.packages") or 1,
            cores_per_socket=physical,
            threads_per_core=(logical // physical) if physical else None,
            performance_cores=performance,
            efficiency_cores=efficiency,
            # macOS exposes no user-selectable governor; see plan_pinning and the
            # provenance_gaps entry the caller writes for this.
            frequency_governor=None,
            memory_total_bytes=_sysctl_int("hw.memsize"),
            transparent_huge_pages=None,
            load_avg=load_avg,
        )

    if system == "Linux":
        cpuinfo = _read_text("/proc/cpuinfo") or ""
        model = None
        for line in cpuinfo.splitlines():
            if line.lower().startswith(("model name", "cpu model")):
                model = line.split(":", 1)[1].strip()
                break
        physical_ids = {
            line.split(":", 1)[1].strip() for line in cpuinfo.splitlines() if line.startswith("physical id")
        }
        core_ids = {line.split(":", 1)[1].strip() for line in cpuinfo.splitlines() if line.startswith("core id")}
        logical = os.cpu_count() or 1
        sockets = len(physical_ids) or 1
        cores_per_socket = len(core_ids) or None
        governor = _read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
        meminfo = _read_text("/proc/meminfo") or ""
        total_bytes = None
        for line in meminfo.splitlines():
            if line.startswith("MemTotal:"):
                total_bytes = int(line.split()[1]) * 1024
                break
        thp = _read_text("/sys/kernel/mm/transparent_hugepage/enabled")
        release = ""
        os_release = _read_text("/etc/os-release") or ""
        for line in os_release.splitlines():
            if line.startswith("PRETTY_NAME="):
                release = line.split("=", 1)[1].strip().strip('"')
                break
        return HostFacts(
            hostname=platform.node() or "unknown",
            system=system,
            os_name="Linux",
            os_release=release or platform.release(),
            kernel=platform.release(),
            arch=platform.machine(),
            cpu_model=model,
            logical_cpus=logical,
            sockets=sockets,
            cores_per_socket=cores_per_socket,
            threads_per_core=(logical // (sockets * cores_per_socket)) if cores_per_socket else None,
            performance_cores=None,
            efficiency_cores=None,
            frequency_governor=governor.strip() if governor else None,
            memory_total_bytes=total_bytes,
            transparent_huge_pages=thp.strip() if thp else None,
            load_avg=load_avg,
        )

    return HostFacts(
        hostname=platform.node() or "unknown",
        system=system,
        os_name=system or "unknown",
        os_release=platform.release() or "unknown",
        kernel=platform.release() or None,
        arch=platform.machine() or "unknown",
        cpu_model=None,
        logical_cpus=os.cpu_count() or 1,
        sockets=None,
        cores_per_socket=None,
        threads_per_core=None,
        performance_cores=None,
        efficiency_cores=None,
        frequency_governor=None,
        memory_total_bytes=None,
        transparent_huge_pages=None,
        load_avg=load_avg,
    )


@dataclass(frozen=True)
class GitFacts:
    commit: str
    commit_short: str
    dirty: bool
    branch: str | None
    describe: str | None
    available: bool


NULL_COMMIT = "0" * 40


def build_tree_pathspecs(repo_root: Path, build_base: Path | None) -> list[str]:
    """Pathspecs excluding the build trees this run owns, if they are in the repo.

    `dirty` asks whether the measured code can differ from the recorded commit,
    so it has to see an untracked .cpp or a modified header.  A build tree is an
    output, not an input, and Eigen's .gitignore does not cover `build*/` -- so
    without this the harness's own default --build-dir sits in the worktree as
    untracked and trips the guard on the very first run.  A guard that always
    fires teaches everyone to pass --allow-dirty by reflex, which marks every
    result unreproducible and destroys the provenance it exists to protect.

    Excluding by pathspec rather than by filtering the porcelain output keeps
    git responsible for quoting, renames and collapsed untracked directories.
    """
    if build_base is None:
        return []
    try:
        relative = build_base.resolve().relative_to(repo_root.resolve())
    except (ValueError, OSError):
        # Outside the worktree (or unresolvable): it cannot appear in status.
        return []
    if relative == Path("."):
        # --build-dir is the repo root itself; excluding it would exclude
        # everything and silently disable the guard.
        return []
    return [".", f":(exclude,top){relative.as_posix()}"]


def probe_git(repo_root: Path, build_base: Path | None = None) -> GitFacts:
    """Read-only interrogation of the Eigen checkout."""

    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", "-C", str(repo_root), *args], capture_output=True, text=True, check=False
            )
        except OSError:
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    if not commit or not re.match(r"^[0-9a-f]{40}$", commit):
        return GitFacts(NULL_COMMIT, NULL_COMMIT[:9], True, None, None, available=False)
    pathspecs = build_tree_pathspecs(repo_root, build_base)
    status = git("status", "--porcelain", *(["--", *pathspecs] if pathspecs else []))
    return GitFacts(
        commit=commit,
        commit_short=commit[:9],
        dirty=bool(status),
        branch=git("rev-parse", "--abbrev-ref", "HEAD"),
        describe=git("describe", "--always", "--dirty"),
        available=True,
    )


def probe_eigen_version(repo_root: Path) -> dict[str, Any]:
    """Parse ``Eigen/Version`` for the version macros the provenance records."""
    text = _read_text(str(repo_root / "Eigen" / "Version")) or ""
    out: dict[str, Any] = {"world": None, "major": None, "minor": None, "string": None}
    for key, macro in (
        ("world", "EIGEN_WORLD_VERSION"),
        ("major", "EIGEN_MAJOR_VERSION"),
        ("minor", "EIGEN_MINOR_VERSION"),
    ):
        match = re.search(rf"define\s+{macro}\s+([0-9]+)", text)
        if match:
            out[key] = int(match.group(1))
    match = re.search(r'define\s+EIGEN_VERSION_STRING\s+"([^"]*)"', text)
    if match:
        out["string"] = match.group(1)
    return out


def parse_compiler_version_output(text: str) -> tuple[str, str]:
    """Recover ``(compiler_id, version)`` from a ``--version`` banner. Pure."""
    first = text.strip().splitlines()[0] if text.strip() else ""
    if "Apple clang" in first:
        match = re.search(r"version ([0-9][0-9.]*)", first)
        return "AppleClang", match.group(1) if match else "unknown"
    if "clang" in first.lower():
        match = re.search(r"version ([0-9][0-9.]*)", first)
        return "Clang", match.group(1) if match else "unknown"
    if first.lower().startswith("g++") or "Free Software Foundation" in text:
        match = re.search(r"([0-9]+\.[0-9]+\.[0-9]+)", first)
        return "GNU", match.group(1) if match else "unknown"
    if "Intel" in first:
        match = re.search(r"([0-9]+\.[0-9]+\.[0-9]+)", first)
        return "IntelLLVM", match.group(1) if match else "unknown"
    match = re.search(r"([0-9]+\.[0-9]+(?:\.[0-9]+)?)", first)
    return "unknown", match.group(1) if match else "unknown"


def read_cmake_cache(build_dir: Path) -> dict[str, str]:
    """Parse ``CMakeCache.txt`` into a plain mapping."""
    text = _read_text(str(Path(build_dir) / "CMakeCache.txt"))
    if not text:
        return {}
    cache: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "//")):
            continue
        name, sep, value = line.partition("=")
        if not sep:
            continue
        cache[name.split(":", 1)[0]] = value
    return cache


def extract_cxx_flags(compile_commands: Sequence[Mapping[str, Any]], source: str) -> list[str]:
    """The exact compile flags for one translation unit. Pure.

    Reads ``compile_commands.json`` so ``provenance.toolchain.cxx_flags`` is the
    command a reader can reconstruct, not a summary of CMake variables. The
    compiler, the output arguments and the source itself are dropped; everything
    else is preserved in order.
    """
    stem = Path(source).name
    for entry in compile_commands:
        entry_file = str(entry.get("file", ""))
        if Path(entry_file).name != stem:
            continue
        if "arguments" in entry:
            tokens = [str(token) for token in entry["arguments"]]
        else:
            tokens = shlex.split(str(entry.get("command", "")))
        flags: list[str] = []
        skip_next = False
        for index, token in enumerate(tokens):
            if index == 0 or skip_next:
                skip_next = False
                continue
            if token in ("-o", "-c", "-MF", "-MT", "-MQ"):
                skip_next = token != "-c"
                continue
            if Path(token).name == stem:
                continue
            flags.append(token)
        return flags
    return []


# ---------------------------------------------------------------------------
# 10. Provenance assembly
# ---------------------------------------------------------------------------


@dataclass
class ProvenanceInputs:
    """Plain data the provenance block is assembled from. No side effects."""

    timestamp: datetime
    machine: MachineProfile
    host: HostFacts
    git: GitFacts
    eigen_version: Mapping[str, Any]
    isa_target: str
    isa_flags: Sequence[str]
    arm_key: str | None
    arm_profile: ArmProfile | None
    arm_context: Mapping[str, str]
    compiler_id: str
    compiler_version: str
    compiler_path: str | None
    cxx_standard: int
    cxx_flags: Sequence[str]
    cmake_build_type: str
    benchmark_library_version: str | None
    threads: int
    thread_env: Mapping[str, str]
    eigen_nb_threads: int | None
    pinning: PinningPlan
    caches: Sequence[Mapping[str, Any]]
    cpu_scaling_enabled: bool | None
    argv: Sequence[str]
    benchmark_argv: Sequence[str]
    executable: str | None
    repetitions: int
    min_time: str
    benchmark_filter: str | None
    load_avg_before: Sequence[float] | None
    load_avg_after: Sequence[float] | None
    duration_s: float | None
    notes: str | None


def assemble_provenance(inputs: ProvenanceInputs) -> tuple[dict, list[dict]]:
    """Build the provenance block and the matching ``provenance_gaps``. Pure.

    Every nullable field left null obliges a gap entry stating why the platform
    could not supply it: a published number must be able to say what was not
    knowable rather than quietly omitting it.
    """
    machine, host, git_facts = inputs.machine, inputs.host, inputs.git
    gaps: list[dict] = []

    def gap(pointer: str, reason: str) -> None:
        gaps.append({"field": pointer, "reason": reason})

    cpu_model = host.cpu_model or machine.cpu_model
    governor = host.frequency_governor or machine.frequency_governor
    if governor is None:
        gap(
            "/provenance/cpu/frequency_governor",
            f"{host.os_name} exposes no user-visible CPU frequency governor or power policy",
        )
    if host.memory_total_bytes is None:
        gap("/provenance/memory/total_bytes", f"total memory could not be probed on {host.os_name}")
    if host.transparent_huge_pages is None:
        gap(
            "/provenance/memory/transparent_huge_pages",
            f"{host.os_name} exposes no transparent-huge-page setting comparable to Linux's",
        )

    cpu = {
        "model": cpu_model,
        "vendor": None,
        "microarchitecture": machine.microarchitecture,
        "sockets": host.sockets if host.sockets is not None else machine.sockets,
        "cores_per_socket": (
            host.cores_per_socket if host.cores_per_socket is not None else machine.cores_per_socket
        ),
        "threads_per_core": (
            host.threads_per_core if host.threads_per_core is not None else machine.threads_per_core
        ),
        "logical_cpus": host.logical_cpus,
        "performance_cores": host.performance_cores,
        "efficiency_cores": host.efficiency_cores,
        "smt_enabled": machine.smt_enabled,
        "frequency_governor": governor,
        "turbo_enabled": None,
        "base_frequency_mhz": None,
        "max_frequency_mhz": None,
        "caches": [dict(entry) for entry in inputs.caches],
    }
    if cpu["smt_enabled"] is None:
        gap("/provenance/cpu/smt_enabled", "the machine profile does not state whether SMT was enabled")
    if cpu["turbo_enabled"] is None:
        gap("/provenance/cpu/turbo_enabled", f"{host.os_name} exposes no turbo/boost state to a user process")
    if cpu["vendor"] is None:
        gap("/provenance/cpu/vendor", "no vendor string was probed; cpu.model carries the vendor's own text")

    numa_policy = inputs.pinning.numa_policy
    if numa_policy is None:
        numa_policy = "default" if (machine.numa_nodes or 1) > 1 else "n/a"
    numa = {
        "nodes": machine.numa_nodes,
        "policy": numa_policy,
        "cpu_binding": inputs.pinning.description,
        "node_of_run": inputs.pinning.node_of_run,
    }
    if numa["cpu_binding"] is None:
        gap(
            "/provenance/numa/cpu_binding",
            inputs.pinning.unavailable_reason or "no CPU affinity was requested by the machine profile",
        )
    if numa["node_of_run"] is None and (machine.numa_nodes or 1) > 1:
        gap("/provenance/numa/node_of_run", "the run was not bound to a NUMA node")

    # The binary's own report, never the requested thread count: benchmarks/
    # CMakeLists.txt compiles no OpenMP, so Eigen's GEMM is sequential no matter
    # what OMP_NUM_THREADS says, while the vendor library obeys it. Deriving this
    # from inputs.threads would state "openmp" in the same document whose
    # provenance.threading.eigen_has_openmp says false.
    eigen_has_openmp = _optional_bool(inputs.arm_context.get("eigen_bench.eigen_has_openmp"))
    if eigen_has_openmp is None:
        eigen_threading = None
        gap(
            "/provenance/arms/eigen/threading_model",
            "the benchmark binary did not report whether Eigen was built with OpenMP, so whether the Eigen "
            "arm could use more than one thread is unknown",
        )
    elif eigen_has_openmp and (inputs.eigen_nb_threads or 1) > 1:
        eigen_threading = "openmp"
    else:
        eigen_threading = "sequential"
    arms: dict[str, dict] = {
        "eigen": {
            "kind": "eigen",
            "library_name": "Eigen",
            "library_version": _eigen_library_version(inputs.eigen_version, git_facts),
            "library_path": None,
            "provides": [],
            "threading_model": eigen_threading,
            "interface": None,
        }
    }
    if inputs.threads != 1 and eigen_threading == "sequential":
        note = (
            f"the run requested {inputs.threads} threads, but Eigen was built without OpenMP and ran "
            "sequentially; only the reference library is threaded here, so the ratio is Eigen-sequential "
            "against a threaded vendor"
        )
        gaps.append({"field": "/provenance/threading/requested_threads", "reason": note})
    if inputs.arm_key:
        profile = inputs.arm_profile
        context = inputs.arm_context
        version = (
            context.get("eigen_bench.reference_library_version")
            or (profile.version_fallback if profile else None)
            or "unknown"
        )
        if version == "unknown":
            gap(
                f"/provenance/arms/{inputs.arm_key}/library_version",
                "the reference library exposes no runtime version query and the machine profile supplies no "
                "version_fallback; vendor releases change kernels without changing the API, so this number "
                "cannot be attributed to a specific build",
            )
        library_path = context.get("eigen_bench.reference_library_path") or (
            profile.library_path if profile else None
        )
        if not library_path:
            library_path = None
            gap(
                f"/provenance/arms/{inputs.arm_key}/library_path",
                "the benchmark binary did not report the resolved path of the reference library",
            )
        arms[inputs.arm_key] = {
            "kind": "reference",
            "library_name": (
                context.get("eigen_bench.reference_library_name")
                or (profile.library_name if profile else inputs.arm_key)
            ),
            "library_version": version,
            "library_path": library_path,
            "provides": list(profile.provides) if profile else [],
            "threading_model": (
                context.get("eigen_bench.reference_threading")
                or (profile.threading_model if profile else None)
                or None
            ),
            "interface": (
                context.get("eigen_bench.reference_interface") or (profile.interface if profile else None) or None
            ),
        }

    if inputs.compiler_path is None:
        gap("/provenance/toolchain/compiler_path", "the compiler path was not recorded by the build")
    if inputs.benchmark_library_version is None:
        gap(
            "/provenance/toolchain/benchmark_library_version",
            "the benchmark output carried no 'library_version' context key",
        )

    toolchain = {
        "compiler_id": inputs.compiler_id,
        "compiler_version": inputs.compiler_version,
        "compiler_path": inputs.compiler_path,
        "cxx_standard": inputs.cxx_standard,
        "cxx_flags": list(inputs.cxx_flags),
        # Nothing probes the link line: the benchmarks link no reference library
        # through CMake's link flags, so this is recorded as the empty fact it is.
        "link_flags": [],
        "isa_target": inputs.isa_target,
        "isa_flags": list(inputs.isa_flags),
        "cmake_build_type": inputs.cmake_build_type,
        "benchmark_library_version": inputs.benchmark_library_version,
    }

    threading = {
        "requested_threads": inputs.threads,
        "env": dict(inputs.thread_env),
        "eigen_nb_threads": inputs.eigen_nb_threads,
        "eigen_has_openmp": _optional_bool(inputs.arm_context.get("eigen_bench.eigen_has_openmp")),
    }
    if threading["eigen_nb_threads"] is None:
        gap(
            "/provenance/threading/eigen_nb_threads",
            "the benchmark binary did not report Eigen::nbThreads(); Eigen threading may not be compiled in",
        )
    if threading["eigen_has_openmp"] is None:
        gap(
            "/provenance/threading/eigen_has_openmp",
            "the benchmark binary did not report whether Eigen was built with OpenMP",
        )

    if inputs.cpu_scaling_enabled is None:
        gap(
            "/provenance/run/cpu_scaling_enabled",
            "the benchmark output carried no 'cpu_scaling_enabled' context key",
        )

    notes = inputs.notes
    if not git_facts.available:
        note = "the Eigen checkout is not a git repository; the commit is recorded as forty zeros"
        notes = f"{notes}; {note}" if notes else note
        gap("/provenance/eigen/commit", note)
    if inputs.pinning.unavailable_reason and inputs.pinning.applied is False:
        note = inputs.pinning.unavailable_reason
        notes = f"{notes}; {note}" if notes else note
    if not machine.locally_verified:
        note = (
            f"machine profile {machine.id!r} is marked locally_verified = false: its static facts have not "
            "been confirmed against the hardware they describe"
        )
        notes = f"{notes}; {note}" if notes else note

    provenance = {
        "timestamp_utc": rfc3339_timestamp(inputs.timestamp),
        "hostname": host.hostname,
        "machine_config_id": machine.id,
        "machine_config_sha256": machine.sha256,
        "cpu": cpu,
        "numa": numa,
        "memory": {
            "total_bytes": host.memory_total_bytes,
            "transparent_huge_pages": host.transparent_huge_pages,
        },
        "os": {
            "name": host.os_name,
            "release": host.os_release,
            "kernel": host.kernel,
            "arch": host.arch,
            "container_image": os.environ.get("EIGEN_BENCH_CONTAINER_IMAGE") or None,
        },
        "toolchain": toolchain,
        "threading": threading,
        "arms": arms,
        "eigen": {
            "commit": git_facts.commit,
            "commit_short": git_facts.commit_short,
            "describe": git_facts.describe,
            "branch": git_facts.branch,
            "dirty": git_facts.dirty,
            "world_version": inputs.eigen_version.get("world"),
            "major_version": inputs.eigen_version.get("major"),
            "minor_version": inputs.eigen_version.get("minor"),
        },
        "harness": {
            "name": HARNESS_NAME,
            "version": HARNESS_VERSION,
            "argv": list(inputs.argv),
            "benchmark_argv": list(inputs.benchmark_argv),
            "executable": inputs.executable,
            "ci_job_url": os.environ.get("CI_JOB_URL") or None,
        },
        "run": {
            "repetitions": inputs.repetitions,
            "min_time": inputs.min_time,
            "benchmark_filter": inputs.benchmark_filter,
            "report_aggregates_only": False,
            "cpu_scaling_enabled": inputs.cpu_scaling_enabled,
            "load_avg_before": list(inputs.load_avg_before) if inputs.load_avg_before else None,
            "load_avg_after": list(inputs.load_avg_after) if inputs.load_avg_after else None,
            "duration_s": inputs.duration_s,
            "notes": notes,
        },
    }
    if provenance["os"]["container_image"] is None:
        gap("/provenance/os/container_image", "the run was not made inside a declared container image")
    if provenance["harness"]["ci_job_url"] is None:
        gap("/provenance/harness/ci_job_url", "the run was made outside CI")
    return provenance, gaps


def _eigen_library_version(version: Mapping[str, Any], git_facts: GitFacts) -> str:
    text = version.get("string")
    if text:
        return f"{text} ({git_facts.commit_short})"
    return git_facts.commit_short


def _optional_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    lowered = str(value).strip().lower()
    if lowered in ("true", "1", "yes", "on"):
        return True
    if lowered in ("false", "0", "no", "off"):
        return False
    return None


def normalise_caches(entries: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Map Google Benchmark's context ``caches`` onto the schema's shape. Pure."""
    out: list[dict] = []
    for entry in entries:
        size = entry.get("size_bytes", entry.get("size"))
        if size is None:
            continue
        out.append(
            {
                "level": int(entry.get("level", 1)),
                "type": str(entry.get("type", "Unified")),
                "size_bytes": int(size),
                "num_sharing": (
                    int(entry["num_sharing"]) if entry.get("num_sharing") not in (None, "") else None
                ),
            }
        )
    return out


# ---------------------------------------------------------------------------
# 11. Result document assembly and validation
# ---------------------------------------------------------------------------


def build_result_document(
    *,
    run_id: str,
    provenance: Mapping[str, Any],
    gaps: Sequence[Mapping[str, Any]],
    scope: Mapping[str, Any],
    measurements: Sequence[Mapping[str, Any]],
    not_measured: Sequence[Mapping[str, Any]],
    source_files: Sequence[Mapping[str, Any]] = (),
) -> dict:
    """Assemble the result file. Pure."""
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "kind": RESULT_KIND,
        "run_id": run_id,
        "partial": True,
        "provenance": dict(provenance),
        "scope": dict(scope),
        "measurements": [dict(row) for row in measurements],
        "not_measured": [dict(row) for row in not_measured],
        "provenance_gaps": [dict(row) for row in gaps],
        "source_files": [dict(row) for row in source_files],
    }


def build_scope(
    *,
    ops: Sequence[str],
    arms: Sequence[str],
    scalars: Sequence[str],
    threads: Sequence[int],
    shape_groups: Mapping[str, Sequence[str]],
    ops_toml_sha256: str | None,
) -> dict:
    return {
        "ops": list(ops),
        "arms": list(arms),
        "scalars": list(scalars),
        "threads": [int(value) for value in threads],
        "shape_groups": {op: list(groups) for op, groups in shape_groups.items()},
        "ops_toml_sha256": ops_toml_sha256,
    }


def load_result_validator(schema_path: Path):
    """Compile ``result_schema.json`` once; ``None`` when jsonschema is absent.

    The schema is 29KB and compiling a validator from it is not free, while a run
    validates one document per measurement unit against the same schema every
    time; the compiled validator is therefore built by the caller and reused.
    """
    try:
        import jsonschema
    except ImportError:
        return None
    try:
        schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessError(f"cannot read result schema {schema_path}: {exc}") from exc
    return jsonschema.Draft202012Validator(schema)


def validate_result(document: Mapping[str, Any], validator: Any) -> list[str]:
    """Validate against a compiled validator; return human-readable errors.

    A ``None`` validator -- jsonschema is not installed -- is reported as a
    warning-shaped error prefixed ``skipped:``; the caller treats that as "not
    validated" rather than invalid, because refusing to record a measurement over
    an absent developer dependency would lose the measurement.
    """
    if validator is None:
        return ["skipped: the jsonschema package is not installed; the result was written unvalidated"]
    return [
        f"{'/'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
        for error in sorted(validator.iter_errors(document), key=lambda err: list(err.absolute_path))
    ]


def write_json(path: Path, document: Mapping[str, Any]) -> None:
    path = Path(path)
    if str(path) != "-":
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
    else:
        json.dump(document, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# 12. Configure, build, execute
# ---------------------------------------------------------------------------


@dataclass
class Reporter:
    verbosity: int = 0

    def info(self, message: str) -> None:
        if self.verbosity >= 1:
            print(f"run.py: {message}", file=sys.stderr)

    def debug(self, message: str) -> None:
        if self.verbosity >= 2:
            print(f"run.py: {message}", file=sys.stderr)

    def result(self, message: str) -> None:
        print(f"run.py: wrote {message}", file=sys.stderr)

    def warn(self, message: str) -> None:
        print(f"run.py: warning: {message}", file=sys.stderr)

    def error(self, message: str) -> None:
        print(f"run.py: error: {message}", file=sys.stderr)


def configure_command(
    *,
    source_dir: Path,
    build_dir: Path,
    machine: MachineProfile,
    isa_target: str,
    arm: ArmProfile | None,
    cxx_standard: int,
) -> list[str]:
    """The CMake configure command line. Pure."""
    command = ["cmake", "-S", str(source_dir), "-B", str(build_dir)]
    if machine.generator:
        command += ["-G", machine.generator]
    command += [
        f"-DCMAKE_BUILD_TYPE={machine.build_type}",
        f"-DCMAKE_CXX_STANDARD={cxx_standard}",
        "-DCMAKE_CXX_STANDARD_REQUIRED=ON",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
    ]
    command += list(machine.isa_options(isa_target))
    if arm is not None:
        command += list(arm.cmake_options)
    return command


def build_command(build_dir: Path, target: str, jobs: int) -> list[str]:
    return ["cmake", "--build", str(build_dir), "--target", target, "--parallel", str(jobs)]


def load_vendor_info(build_dir: Path) -> dict[str, Any] | None:
    """What the build itself recorded about the configuration it produced.

    CMake writes this beside the binaries at configure time.  Everything in it was
    decided by the build; re-deriving the same facts from CMakeCache.txt, the
    compile database and a search of the build tree makes run.py's account and the
    build's account two independent stories that can silently disagree -- about
    which executable was run, which reference library it was linked against, and
    which ops.toml it was compiled for.  Read it, and cross-check the rest.
    """
    path = Path(build_dir) / "comparison" / "vendor_info.json"
    if not path.is_file():
        path = Path(build_dir) / "vendor_info.json"
    if not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            info = json.load(handle)
    except (OSError, ValueError) as exc:
        raise HarnessError(f"{path} exists but could not be read: {exc}", EXIT_CONFIG) from exc
    return info if isinstance(info, dict) else None


def executable_from_vendor_info(info: Mapping[str, Any] | None, target: str) -> Path | None:
    """The path the build recorded for a target, in preference to guessing."""
    for entry in (info or {}).get("targets", []) or []:
        if entry.get("target") == target and entry.get("executable"):
            candidate = Path(str(entry["executable"]))
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate
    return None


def vendor_info_provides(info: Mapping[str, Any] | None) -> list[str]:
    """The interface families the linked reference library actually exposes.

    The vendor table records this per row; without it nothing can tell that, say,
    netlib supplies BLAS and CBLAS but no LAPACK, and a LAPACK op would be planned
    against a library that cannot perform it.
    """
    reference = ((info or {}).get("reference") or {})
    return [str(x) for x in (reference.get("provides") or [])]


def find_benchmark_executable(build_dir: Path, target: str) -> Path | None:
    """Locate a built benchmark executable within a build tree."""
    build_dir = Path(build_dir)
    candidates = [
        build_dir / "comparison" / target,
        build_dir / target,
        build_dir / "comparison" / f"{target}.exe",
        build_dir / f"{target}.exe",
    ]
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    for candidate in sorted(build_dir.rglob(target)):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def benchmark_command(
    executable: Path,
    *,
    out_path: Path,
    repetitions: int,
    min_time: str,
    benchmark_filter: str | None,
    pinning: PinningPlan,
) -> list[str]:
    """The full benchmark invocation, affinity wrapper included. Pure."""
    command = list(pinning.command_prefix) + [str(executable)]
    command += [
        "--benchmark_format=json",
        f"--benchmark_out={out_path}",
        "--benchmark_out_format=json",
        f"--benchmark_repetitions={repetitions}",
        f"--benchmark_min_time={min_time}",
        "--benchmark_report_aggregates_only=false",
        "--benchmark_display_aggregates_only=false",
    ]
    if benchmark_filter:
        command.append(f"--benchmark_filter={benchmark_filter}")
    return command


def run_command(
    command: Sequence[str],
    *,
    reporter: Reporter,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess:
    reporter.info("$ " + " ".join(command))
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    return subprocess.run(
        list(command),
        env=full_env,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
    )


# ---------------------------------------------------------------------------
# 13. Command line
# ---------------------------------------------------------------------------


def repo_root_from(script: Path) -> Path:
    return Path(script).resolve().parent.parent.parent


def build_parser(default_root: Path) -> argparse.ArgumentParser:
    comparison_dir = default_root / "benchmarks" / "comparison"
    parser = UsageErrorArgumentParser(
        prog="run.py",
        description="Measure Eigen against a reference library and write one validated result file.",
    )
    parser.add_argument("--machine", required=True, metavar="ID", help="key of machines/<ID>.toml")
    parser.add_argument("--build-dir", default="build-comparison", metavar="DIR")
    parser.add_argument("--ops", action="append", metavar="LIST", help="repeatable; 'all' accepted")
    parser.add_argument("--scalars", action="append", metavar="LIST")
    parser.add_argument("--arms", action="append", metavar="LIST", help="'eigen' is always included")
    parser.add_argument("--groups", action="append", metavar="LIST")
    parser.add_argument("--threads", action="append", metavar="LIST")
    parser.add_argument(
        "--allow-sequential-eigen",
        action="store_true",
        help="permit --threads N>1 against a build whose Eigen has no OpenMP; the Eigen arm then runs "
        "sequentially against a threaded reference and the result says so",
    )
    parser.add_argument(
        "--isa",
        action="append",
        metavar="LIST",
        help="ISA targets from the machine profile; default is its default_isa_target",
    )
    parser.add_argument("--repetitions", type=int, default=10, metavar="N")
    parser.add_argument("--min-time", default="0.5s", metavar="SPEC")
    parser.add_argument("--filter", default=None, metavar="REGEX")
    parser.add_argument("--results-dir", default=str(comparison_dir / "results"), metavar="DIR")
    parser.add_argument("--out", default=None, metavar="PATH", help="'-' writes to stdout")
    parser.add_argument("--configure", dest="configure", action="store_true", default=True)
    parser.add_argument("--no-configure", dest="configure", action="store_false")
    parser.add_argument("--build", dest="build", action="store_true", default=True)
    parser.add_argument("--no-build", dest="build", action="store_false")
    parser.add_argument("-j", dest="jobs", type=int, default=os.cpu_count() or 1, metavar="N")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-cells", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-noisy", action="store_true")
    parser.add_argument("--note", default=None, metavar="TEXT")
    parser.add_argument("--machines-dir", default=str(comparison_dir / "machines"), metavar="DIR")
    parser.add_argument("--ops-toml", default=str(comparison_dir / "ops.toml"), metavar="PATH")
    parser.add_argument("--schema", default=str(comparison_dir / "result_schema.json"), metavar="PATH")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument("--version", action="version", version=f"{HARNESS_NAME} {HARNESS_VERSION}")
    return parser


# ---------------------------------------------------------------------------
# 14. Orchestration
# ---------------------------------------------------------------------------


@dataclass
class Selection:
    """The resolved measurement selection, independent of any host state."""

    ops: list[str]
    arms: list[str]
    scalars: list[str]
    groups: list[str]
    threads: list[int]
    isa_targets: list[str]


def resolve_selection(
    args: argparse.Namespace,
    registry: OpsRegistry,
    machine: MachineProfile,
) -> Selection:
    """Turn parsed arguments plus the registry and profile into a selection. Pure."""
    requested_ops = split_list(args.ops)
    if not requested_ops or requested_ops == ["all"]:
        ops = registry.implemented_ops() if not requested_ops else list(registry.ops)
    else:
        unknown = [op for op in requested_ops if op not in registry.ops]
        if unknown:
            raise HarnessError(f"unknown operation(s): {', '.join(unknown)}")
        ops = requested_ops
    if not ops:
        raise HarnessError("no operation in ops.toml has status = \"implemented\"")

    scalars = split_list(args.scalars) or ["f64"]
    unknown = [tag for tag in scalars if tag not in registry.scalars]
    if unknown:
        raise HarnessError(f"unknown scalar tag(s): {', '.join(unknown)}")

    requested_arms = split_list(args.arms)
    available = list(machine.reference_arms())
    if not requested_arms:
        arms = ["eigen"] + available
    else:
        unknown = [arm for arm in requested_arms if arm != "eigen" and arm not in machine.arms]
        if unknown:
            raise HarnessError(
                f"machine profile {machine.id!r} declares no arm(s): {', '.join(unknown)}; "
                f"it declares {', '.join(available) or '(none)'}"
            )
        arms = ["eigen"] + [arm for arm in requested_arms if arm != "eigen"]

    groups = split_list(args.groups)
    for op_key in ops:
        resolve_groups(registry, op_key, groups)

    threads_raw = split_list(args.threads) or ["1"]
    threads: list[int] = []
    for value in threads_raw:
        if not value.isdigit() or int(value) < 1:
            raise HarnessError(f"--threads takes positive integers, got {value!r}")
        threads.append(int(value))

    isa_raw = split_list(args.isa) or [machine.default_isa_target]
    unknown = [target for target in isa_raw if target not in machine.isa_targets]
    if unknown:
        raise HarnessError(
            f"machine profile {machine.id!r} does not list ISA target(s): {', '.join(unknown)}; "
            f"it lists {', '.join(machine.isa_targets)}"
        )
    return Selection(ops=ops, arms=arms, scalars=scalars, groups=groups, threads=threads, isa_targets=isa_raw)


def run_units(selection: Selection) -> list[tuple[str, str | None, int]]:
    """One measurement unit — and therefore one result file — per (ISA, arm, threads).

    Every unit is independent, so measuring one operation or one vendor writes a
    valid partial result; nothing here requires the matrix to be complete.
    """
    references = [arm for arm in selection.arms if arm != "eigen"]
    if not references:
        references = [None]
    return [
        (isa, arm, threads)
        for isa in selection.isa_targets
        for arm in references
        for threads in selection.threads
    ]


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    root = repo_root_from(Path(__file__))
    parser = build_parser(root)
    args = parser.parse_args(argv)
    reporter = Reporter(args.verbose)

    try:
        return _run(args, argv, root, reporter)
    except PipelineError as exc:
        reporter.error(str(exc))
        return exc.exit_code


def _run(args: argparse.Namespace, argv: Sequence[str], root: Path, reporter: Reporter) -> int:
    registry = load_ops_registry(Path(args.ops_toml))
    machine_path = machine_profile_path(Path(args.machines_dir), args.machine)
    if not machine_path.is_file():
        raise HarnessError(f"no machine profile for {args.machine!r} at {machine_path}")
    machine = load_machine_profile(machine_path)
    selection = resolve_selection(args, registry, machine)

    if args.list_cells:
        for isa, arm_key, threads in run_units(selection):
            arms = ["eigen"] + ([arm_key] if arm_key else [])
            for cell in plan_cells(
                registry,
                ops=selection.ops,
                arms=arms,
                scalars=selection.scalars,
                groups=selection.groups,
                threads=threads,
            ):
                dims = " ".join(f"{name}:{value}" for name, value in zip(cell.shape_dims, cell.values))
                print(f"{cell.op} {cell.arm} {cell.scalar} {dims} threads:{cell.threads} isa:{isa}")
        return EXIT_OK

    git_facts = probe_git(root, Path(args.build_dir))
    host = probe_host()
    eigen_version = probe_eigen_version(root)

    if args.dry_run:
        _print_plan(args, registry, machine, selection, git_facts, host)
        return EXIT_OK

    if git_facts.dirty and not args.allow_dirty:
        reporter.error(
            "the Eigen worktree has uncommitted changes; a measurement made from it is not reproducible. "
            "Commit, stash, or pass --allow-dirty to record the result as dirty."
        )
        return EXIT_DIRTY
    if host.load_avg and host.load_avg[0] > machine.max_load_avg and not args.allow_noisy:
        reporter.error(
            f"1-minute load average {host.load_avg[0]:.2f} exceeds the machine profile's max_load_avg "
            f"{machine.max_load_avg}; stop competing work or pass --allow-noisy."
        )
        return EXIT_NOISY
    if git_facts.dirty:
        reporter.warn("measuring from a dirty worktree; the result is marked dirty and is not reproducible")
    if host.load_avg and host.load_avg[0] > machine.max_load_avg:
        reporter.warn(f"measuring at load average {host.load_avg[0]:.2f}; the numbers carry that caveat")
    if host.cpu_model and host.cpu_model != machine.cpu_model:
        reporter.warn(
            f"the host reports CPU model {host.cpu_model!r} but machine profile {machine.id!r} declares "
            f"{machine.cpu_model!r}; the probed value is recorded"
        )

    pinning = plan_pinning(machine)
    if pinning.unavailable_reason:
        reporter.warn(pinning.unavailable_reason)

    validator = load_result_validator(Path(args.schema))

    written: list[Path] = []
    invalid = 0
    runtime_failures = 0
    units = run_units(selection)
    for isa_target, arm_key, threads in units:
        outcome = _measure_unit(
            args=args,
            argv=argv,
            root=root,
            reporter=reporter,
            registry=registry,
            machine=machine,
            selection=selection,
            isa_target=isa_target,
            arm_key=arm_key,
            threads=threads,
            host=host,
            git_facts=git_facts,
            eigen_version=eigen_version,
            pinning=pinning,
            single_unit=len(units) == 1,
            validator=validator,
        )
        if outcome.path is not None:
            written.append(outcome.path)
        invalid += 1 if outcome.invalid else 0
        runtime_failures += 1 if outcome.runtime_failed else 0

    if runtime_failures == len(units) and units:
        reporter.error("every benchmark executable failed at runtime; nothing was measured")
        return EXIT_RUNTIME
    if invalid:
        return EXIT_SCHEMA
    for path in written:
        if str(path) != "-":
            reporter.result(str(path))
    return EXIT_OK


@dataclass
class UnitOutcome:
    path: Path | None = None
    invalid: bool = False
    runtime_failed: bool = False


def _measure_unit(
    *,
    args: argparse.Namespace,
    argv: Sequence[str],
    root: Path,
    reporter: Reporter,
    registry: OpsRegistry,
    machine: MachineProfile,
    selection: Selection,
    isa_target: str,
    arm_key: str | None,
    threads: int,
    host: HostFacts,
    git_facts: GitFacts,
    eigen_version: Mapping[str, Any],
    pinning: PinningPlan,
    single_unit: bool,
    validator: Any,
) -> UnitOutcome:
    started = datetime.now(timezone.utc)
    clock_started = time.monotonic()
    arm_profile = machine.arms.get(arm_key) if arm_key else None
    arms = ["eigen"] + ([arm_key] if arm_key else [])
    label = f"{isa_target}/{arm_key or 'eigen-only'}/t{threads}"

    planned, excluded = apply_extra_filter(
        plan_cells(
            registry,
            ops=selection.ops,
            arms=arms,
            scalars=selection.scalars,
            groups=selection.groups,
            threads=threads,
        ),
        args.filter,
    )
    shape_groups = {op: list(resolve_groups(registry, op, selection.groups)) for op in selection.ops}

    build_dir = make_build_dir(Path(args.build_dir), isa_target, arm_key or "eigen")
    if not build_dir.is_absolute():
        build_dir = root / build_dir

    thread_env = build_thread_env(threads, arm=arm_profile, pinning_env=pinning.env)
    reporter.info(f"[{label}] thread environment: {json.dumps(thread_env, sort_keys=True)}")

    measurements: list[dict] = []
    not_measured: list[dict] = []

    # What the linked reference library actually exposes, per the vendor table.
    reference_provides = vendor_info_provides(load_vendor_info(build_dir))

    def mark(cells: Sequence[Cell], reason: str, detail: str | None, measured: Iterable[tuple] = ()) -> None:
        """Record why these planned cells produced no timing.

        Every cell in scope must end up in ``measurements`` or here; the reducer
        reports anything in neither as a harness bug.
        """
        not_measured.extend(
            diff_not_measured(
                cells, measured, registry, default_reason=reason, default_detail=detail,
                provides=reference_provides,
            )
        )

    mark(
        excluded,
        "excluded_by_filter",
        f"--filter {args.filter!r} did not match" if args.filter else None,
    )
    source_files: list[dict] = []
    context: dict[str, Any] = {}
    executable_used: str | None = None
    benchmark_argv: list[str] = []
    benchmark_filter: str | None = None
    runtime_failed = False

    if arm_profile is not None and not arm_profile.available:
        reason = arm_profile.unavailable_reason or f"the machine profile marks {arm_key!r} as unavailable"
        reporter.warn(f"[{label}] {reason}")
        mark([cell for cell in planned if cell.arm == arm_key], "reference_library_unavailable", reason)
        planned_runnable = [cell for cell in planned if cell.arm != arm_key]
    else:
        planned_runnable = planned

    if args.configure:
        command = configure_command(
            source_dir=root / "benchmarks",
            build_dir=build_dir,
            machine=machine,
            isa_target=isa_target,
            arm=arm_profile if (arm_profile and arm_profile.available) else None,
            cxx_standard=machine.cxx_standard,
        )
        completed = run_command(command, reporter=reporter)
        if completed.returncode != 0:
            reporter.error(f"[{label}] CMake configure failed:\n{completed.stderr.strip()}")
            raise HarnessError(f"configure failed for {label}", EXIT_BUILD)

    targets: dict[str, list[str]] = {}
    for op_key in selection.ops:
        target = registry.target_for(op_key)
        if target is None:
            mark(
                [cell for cell in planned_runnable if cell.op == op_key],
                "not_implemented",
                f"ops.toml names no benchmark source for {op_key}",
            )
            continue
        targets.setdefault(target, []).append(op_key)

    # What the build recorded about itself. Cross-checking it against what run.py
    # plans is the only thing that can notice that the binary about to be measured
    # was not built from the registry being planned from, or against the reference
    # library the result file is about to name.
    vendor_info = load_vendor_info(build_dir)
    if vendor_info is not None:
        built_sha = str(vendor_info.get("ops_toml_sha256") or "")
        if built_sha and registry.sha256 and built_sha != registry.sha256:
            raise HarnessError(
                f"the binaries in {build_dir} were compiled against ops.toml {built_sha[:12]} but this run "
                f"plans from {registry.sha256[:12]}; the grid, the flop formulas and the registrations may "
                "all differ. Rebuild, or point --ops-toml at the registry the build used.",
                EXIT_CONFIG,
            )
        built_arm = str((vendor_info.get("reference") or {}).get("arm") or "")
        if built_arm and arm_key not in ("eigen", built_arm):
            raise HarnessError(
                f"this unit measures arm {arm_key!r} but {build_dir} was linked against {built_arm!r}; "
                "the result would attribute the numbers to a library that did not produce them",
                EXIT_CONFIG,
            )

    for target, ops_in_target in sorted(targets.items()):
        target_cells = [cell for cell in planned_runnable if cell.op in ops_in_target]
        if args.build:
            completed = run_command(build_command(build_dir, target, args.jobs), reporter=reporter)
            if completed.returncode != 0:
                reporter.error(f"[{label}] building {target} failed:\n{completed.stderr.strip()}")
                mark(target_cells, "build_failed", f"target {target} failed to build")
                continue
        executable = executable_from_vendor_info(vendor_info, target) or find_benchmark_executable(build_dir, target)
        if executable is None:
            reporter.warn(f"[{label}] no executable {target} in {build_dir}")
            mark(target_cells, "build_failed", f"no executable named {target} was found in {build_dir}")
            continue

        benchmark_filter = make_benchmark_filter(target_cells)
        raw_path = build_dir / f"{target}.gbench.json"
        command = benchmark_command(
            executable,
            out_path=raw_path,
            repetitions=args.repetitions,
            min_time=args.min_time,
            benchmark_filter=benchmark_filter,
            pinning=pinning,
        )
        benchmark_argv = command[len(pinning.command_prefix) + 1 :]
        executable_used = str(executable)
        completed = run_command(command, reporter=reporter, env=thread_env)
        if completed.returncode != 0 or not raw_path.is_file():
            runtime_failed = True
            reporter.error(f"[{label}] {target} failed at runtime:\n{completed.stderr.strip()[-2000:]}")
            mark(
                target_cells,
                "runtime_error",
                f"{target} exited with status {completed.returncode}; see the harness log",
            )
            continue

        document = json.loads(raw_path.read_text(encoding="utf-8"))
        distilled = distill_benchmark_json(document, registry, planned=target_cells, threads=threads)
        context.update(distilled.context)
        for warning in distilled.warnings:
            reporter.warn(f"[{label}] {warning}")
        measurements.extend(distilled.measurements)
        errored_keys = set()
        for failure in distilled.errors:
            reporter.warn(f"[{label}] benchmark reported an error for {failure['name']}: {failure['message']}")
            if "key" not in failure:
                continue
            errored_keys.add(failure["key"])
            not_measured.append(
                {
                    "op": failure["op"],
                    "arm": failure["arm"],
                    "scalar": failure["scalar"],
                    "shape": failure["shape"],
                    "threads": failure["threads"],
                    "reason": "runtime_error",
                    "detail": failure["message"] or None,
                }
            )
        mark(
            target_cells,
            "not_implemented",
            "the benchmark binary registered no such point; the grid in the C++ source and the "
            "grid in ops.toml differ",
            measured=set(distilled.seen_keys) | errored_keys,
        )
        source_files.append(
            {
                "path": _relative_to_results(raw_path, root),
                "sha256": sha256_file(raw_path),
                "benchmark_library_version": distilled.context.get("library_version"),
            }
        )

    # A run that measured nothing at all while it had runnable cells is not a
    # coverage manifest saying "unimplemented"; it is a broken run. Publishing it
    # would state that an operation is unimplemented on a machine where the whole
    # grid was just measured and then discarded.
    if planned_runnable and not measurements and not runtime_failed:
        runtime_failed = True
        reporter.error(
            f"[{label}] {len(planned_runnable)} runnable cell(s) were planned and none was measured; "
            "the result is written for inspection but the run is a failure"
        )

    eigen_has_openmp = _optional_bool(context.get("eigen_bench.eigen_has_openmp"))
    if threads != 1 and eigen_has_openmp is False and not args.allow_sequential_eigen:
        raise HarnessError(
            f"[{label}] --threads {threads} was requested but the benchmark binary reports that Eigen was "
            "built without OpenMP, so Eigen::nbThreads() is 1 and the Eigen arm runs sequentially while the "
            f"reference library honours the {threads}-thread environment. That comparison is not the one the "
            "table would claim. Build with OpenMP, drop --threads, or pass --allow-sequential-eigen to record "
            "it explicitly as a sequential-Eigen run.",
            EXIT_CONFIG,
        )

    compiler_id, compiler_version, compiler_path = _toolchain_from(build_dir, context, reporter)
    cxx_flags = _cxx_flags_from(build_dir, registry, selection.ops, context)

    inputs = ProvenanceInputs(
        timestamp=started,
        machine=machine,
        host=host,
        git=git_facts,
        eigen_version=eigen_version,
        isa_target=isa_target,
        isa_flags=machine.isa_flags(isa_target),
        arm_key=arm_key,
        arm_profile=arm_profile,
        arm_context={k: str(v) for k, v in context.items() if k.startswith("eigen_bench.")},
        compiler_id=compiler_id,
        compiler_version=compiler_version,
        compiler_path=compiler_path,
        cxx_standard=int(context.get("eigen_bench.cxx_standard", machine.cxx_standard) or machine.cxx_standard),
        cxx_flags=cxx_flags,
        cmake_build_type=machine.build_type,
        benchmark_library_version=context.get("library_version"),
        threads=threads,
        thread_env=thread_env,
        eigen_nb_threads=_optional_int_str(context.get("eigen_bench.eigen_nb_threads")),
        pinning=pinning,
        caches=normalise_caches(context.get("caches", [])),
        cpu_scaling_enabled=_optional_bool(str(context["cpu_scaling_enabled"]))
        if "cpu_scaling_enabled" in context
        else None,
        argv=list(argv),
        benchmark_argv=benchmark_argv,
        executable=executable_used,
        repetitions=args.repetitions,
        min_time=args.min_time,
        benchmark_filter=benchmark_filter,
        load_avg_before=host.load_avg,
        load_avg_after=_load_avg_now(),
        duration_s=time.monotonic() - clock_started,
        notes=args.note,
    )
    provenance, gaps = assemble_provenance(inputs)

    commit_short = git_facts.commit_short
    timestamp = compact_timestamp(started)
    run_id = make_run_id(machine.id, isa_target, arm_key or "eigen", commit_short, timestamp, threads)

    document = build_result_document(
        run_id=run_id,
        provenance=provenance,
        gaps=gaps,
        scope=build_scope(
            ops=selection.ops,
            arms=arms,
            scalars=selection.scalars,
            threads=[threads],
            shape_groups=shape_groups,
            ops_toml_sha256=registry.sha256,
        ),
        measurements=measurements,
        not_measured=collapse_not_measured(not_measured),
        source_files=source_files,
    )

    if args.out:
        if not single_unit:
            raise HarnessError(
                "--out names a single file, but the selection spans several ISA targets, reference arms or "
                "thread counts, each of which is its own result file; narrow the selection or drop --out",
                EXIT_USAGE,
            )
        out_path = Path(args.out)
    else:
        out_path = make_output_path(Path(args.results_dir), machine.id, commit_short, run_id)
        if not out_path.is_absolute():
            out_path = root / out_path

    errors = validate_result(document, validator)
    skipped = [error for error in errors if error.startswith("skipped:")]
    hard = [error for error in errors if not error.startswith("skipped:")]
    for message in skipped:
        reporter.warn(message[len("skipped: ") :])
    if hard:
        invalid_path = make_invalid_output_path(out_path if str(out_path) != "-" else Path(f"{run_id}.json"))
        write_json(invalid_path, document)
        reporter.error(
            f"[{label}] the produced result failed result_schema.json validation and was written to "
            f"{invalid_path} for inspection:\n  " + "\n  ".join(hard[:20])
        )
        return UnitOutcome(path=None, invalid=True, runtime_failed=runtime_failed)

    write_json(out_path, document)
    reporter.info(
        f"[{label}] {len(measurements)} measured, {len(document['not_measured'])} not measured -> {out_path}"
    )
    return UnitOutcome(path=out_path, runtime_failed=runtime_failed)


def _relative_to_results(path: Path, root: Path) -> str:
    """Raw-JSON path for the audit trail, relative when it stays inside the tree."""
    try:
        relative = os.path.relpath(path, start=(root / "benchmarks" / "comparison"))
    except ValueError:
        return str(path.resolve())
    return str(path.resolve()) if relative.startswith("..") else relative


def _optional_int_str(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(str(value))
    except ValueError:
        return None


def _load_avg_now() -> tuple[float, ...] | None:
    try:
        return tuple(float(value) for value in os.getloadavg())
    except (OSError, AttributeError):
        return None


def _toolchain_from(
    build_dir: Path, context: Mapping[str, Any], reporter: Reporter
) -> tuple[str, str, str | None]:
    """Compiler identity: the binary's own report first, then the CMake cache."""
    compiler_id = str(context.get("eigen_bench.compiler_id") or "")
    compiler_version = str(context.get("eigen_bench.compiler_version") or "")
    cache = read_cmake_cache(build_dir)
    compiler_path = cache.get("CMAKE_CXX_COMPILER") or None
    if compiler_id and compiler_version:
        return compiler_id, compiler_version, compiler_path
    if compiler_path:
        try:
            out = subprocess.run([compiler_path, "--version"], capture_output=True, text=True, check=False)
            probed_id, probed_version = parse_compiler_version_output(out.stdout)
            return compiler_id or probed_id, compiler_version or probed_version, compiler_path
        except OSError:
            pass
    reporter.warn("the compiler identity could not be determined; recorded as 'unknown'")
    return compiler_id or "unknown", compiler_version or "unknown", compiler_path


def _cxx_flags_from(
    build_dir: Path, registry: OpsRegistry, ops: Sequence[str], context: Mapping[str, Any]
) -> list[str]:
    """The exact per-translation-unit flags, from compile_commands.json."""
    reported = context.get("eigen_bench.cxx_flags")
    database = build_dir / "compile_commands.json"
    if database.is_file():
        try:
            entries = json.loads(database.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            entries = []
        for op_key in ops:
            source = registry.ops.get(op_key, {}).get("source")
            if not source:
                continue
            flags = extract_cxx_flags(entries, str(source))
            if flags:
                return flags
    if reported:
        return shlex.split(str(reported))
    return []


def _print_plan(
    args: argparse.Namespace,
    registry: OpsRegistry,
    machine: MachineProfile,
    selection: Selection,
    git_facts: GitFacts,
    host: HostFacts,
) -> None:
    units = run_units(selection)
    print(f"machine:      {machine.id} ({machine.display_name})")
    print(f"cpu (probed): {host.cpu_model or 'unknown'}  logical_cpus={host.logical_cpus}")
    print(f"eigen:        {git_facts.commit_short}{' (dirty)' if git_facts.dirty else ''}")
    print(f"ops:          {', '.join(selection.ops)}")
    print(f"scalars:      {', '.join(selection.scalars)}")
    print(f"arms:         {', '.join(selection.arms)}")
    print(f"groups:       {', '.join(selection.groups) or '(each family default_groups)'}")
    print(f"isa targets:  {', '.join(selection.isa_targets)}")
    print(f"threads:      {', '.join(str(value) for value in selection.threads)}")
    print(f"repetitions:  {args.repetitions}   min_time: {args.min_time}")
    pinning = plan_pinning(machine)
    print(f"pinning:      {pinning.description or 'none'}"
          + (f"   ({pinning.unavailable_reason})" if pinning.unavailable_reason else ""))
    total = 0
    for isa_target, arm_key, threads in units:
        arms = ["eigen"] + ([arm_key] if arm_key else [])
        cells = plan_cells(
            registry,
            ops=selection.ops,
            arms=arms,
            scalars=selection.scalars,
            groups=selection.groups,
            threads=threads,
        )
        total += len(cells)
        targets = sorted({registry.target_for(op) or f"<no source for {op}>" for op in selection.ops})
        build_dir = make_build_dir(Path(args.build_dir), isa_target, arm_key or "eigen")
        run_id = make_run_id(
            machine.id,
            isa_target,
            arm_key or "eigen",
            git_facts.commit_short,
            compact_timestamp(datetime.now(timezone.utc)),
            threads,
        )
        print()
        print(f"unit {isa_target} / {arm_key or 'eigen-only'} / threads={threads}")
        print(f"  build dir:  {build_dir}")
        print(f"  targets:    {', '.join(targets)}")
        print(f"  cells:      {len(cells)}")
        thread_env = build_thread_env(
            threads, arm=machine.arms.get(arm_key) if arm_key else None, pinning_env=pinning.env
        )
        out_path = make_output_path(Path(args.results_dir), machine.id, git_facts.commit_short, run_id)
        print(f"  thread env: {json.dumps(thread_env, sort_keys=True)}")
        print(f"  out:        {out_path}")
    print()
    print(f"total planned cells: {total}")


if __name__ == "__main__":
    sys.exit(main())
