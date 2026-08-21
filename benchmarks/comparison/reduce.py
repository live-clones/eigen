#!/usr/bin/env python3
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

"""Merge benchmark comparison result files into the normalised intermediate.

`reduce.py` is the only tool that reads raw result files; `render.py` and
`plots.py` consume its output and nothing else.  See
``benchmarks/comparison/CONTRACTS.md`` sections 1 and 5, which this module
implements.

Two invariants drive the whole file:

* A benchmark name that does not parse is a hard error naming the offending
  string.  A silently dropped row is a number that quietly leaves a published
  table, which is worse than a crash.
* A combination that was not measured is carried through as an explicit
  ``not_measured`` state with a reason, never as zero and never by omission.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import glob as _glob
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# This script is run directly and is also imported by file path by the test
# suite, so its own directory is not reliably on sys.path when _common is needed.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _common import (  # noqa: E402
    EIGEN_ARM,
    PipelineError,
    UsageErrorArgumentParser,
    config_id_for,
    load_ops_registry,
    parse_benchmark_name,
    scalar_flops,
    size_key,
    split_list,
)

REDUCER_VERSION = "1.0.0"
MERGED_SCHEMA_VERSION = "1.0.0"
MERGED_KIND = "eigen-benchmark-comparison-merged"
RESULT_KIND = "eigen-benchmark-comparison-result"

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OPS_TOML = os.path.join(HERE, "ops.toml")
DEFAULT_SCHEMA = os.path.join(HERE, "result_schema.json")

# Exit codes, CONTRACTS.md section 4.2.
EXIT_OK = 0
EXIT_USAGE = 1
EXIT_INPUT_INVALID = 2
EXIT_CONFLICT = 3
EXIT_NO_INPUT = 4


class ReduceError(PipelineError):
    """Fatal, user-facing error.  Carries the process exit code."""

    def __init__(self, message: str, code: int = EXIT_INPUT_INVALID):
        super().__init__(message, code)


# ---------------------------------------------------------------------------
# ops.toml
# ---------------------------------------------------------------------------
#
# The name grammar, the flop-formula evaluator and the registry loader live in
# _common.py: reduce.py reads what run.py writes, so a name or a formula the two
# stages disagree about breaks the pipeline after a valid file is on disk.


def flops_for(registry: Mapping[str, Any], op: str, scalar: str, shape: Mapping[str, int]) -> Tuple[float, bool]:
    """Scalar flop count of one iteration, and whether that count is nominal."""
    flops = registry["ops"][op].get("flops", {})
    formula = flops.get("real")
    nominal = bool(flops.get("nominal", False))
    if not formula:
        return 0.0, nominal
    scale = float(registry["scalars"].get(scalar, {}).get("flop_scale", 2.0))
    return scalar_flops(formula, scale, shape), nominal


def op_shape_dims(registry: Mapping[str, Any], op: str) -> List[str]:
    family = registry["ops"][op]["shape_family"]
    try:
        return list(registry["shape_families"][family]["dims"])
    except KeyError as exc:
        raise ReduceError(f"op {op!r} names unknown shape family {family!r}") from exc


def op_grid(registry: Mapping[str, Any], op: str, groups: Optional[Sequence[str]] = None) -> List[Dict[str, int]]:
    """Ordered union of the selected shape groups, first occurrence winning."""
    family_key = registry["ops"][op]["shape_family"]
    family = registry["shape_families"][family_key]
    dims = list(family["dims"])
    selected = list(groups) if groups else list(family.get("default_groups", []))
    seen: set = set()
    points: List[Dict[str, int]] = []
    for group in family.get("groups", []):
        if group.get("name") not in selected:
            continue
        for point in group.get("points", []):
            key = tuple(point)
            if key in seen:
                continue
            seen.add(key)
            points.append({dim: int(value) for dim, value in zip(dims, point)})
    return points


def shape_group_index(registry: Mapping[str, Any], op: str) -> Dict[Tuple[int, ...], Optional[str]]:
    """``{point: group name}`` for an op's shape family, first group winning."""
    family = registry["shape_families"][registry["ops"][op]["shape_family"]]
    index: Dict[Tuple[int, ...], Optional[str]] = {}
    for group in family.get("groups", []):
        for point in group.get("points", []):
            index.setdefault(tuple(point), group.get("name"))
    return index


def shape_group_of(
    registry: Mapping[str, Any],
    op: str,
    shape: Mapping[str, int],
    index: Optional[Mapping[Tuple[int, ...], Optional[str]]] = None,
) -> Optional[str]:
    dims = registry["shape_families"][registry["ops"][op]["shape_family"]]["dims"]
    if sorted(shape) != sorted(dims):
        return None
    if index is None:
        index = shape_group_index(registry, op)
    return index.get(tuple(int(shape[dim]) for dim in dims))


class _OpCache:
    """Per-reduction memo of the registry lookups that dominate cell building.

    Both answers depend only on the op, but both used to be recomputed per cell:
    ``shape_group_of`` rescanned every group of every family, and ``op_grid``
    rebuilt its whole point list for each not-measured row that fanned out over
    it.  Over a full grid that is thousands of repetitions of a few dozen answers.
    """

    def __init__(self, registry: Mapping[str, Any]):
        self._registry = registry
        self._groups: Dict[str, Mapping[Tuple[int, ...], Optional[str]]] = {}
        self._grids: Dict[Tuple[str, Optional[Tuple[str, ...]]], List[Dict[str, int]]] = {}

    def group_index(self, op: str) -> Mapping[Tuple[int, ...], Optional[str]]:
        if op not in self._groups:
            self._groups[op] = shape_group_index(self._registry, op)
        return self._groups[op]

    def grid(self, op: str, groups: Optional[Sequence[str]] = None) -> List[Dict[str, int]]:
        """Shared, never mutated by a caller: every consumer copies what it keeps."""
        key = (op, tuple(groups) if groups else None)
        if key not in self._grids:
            self._grids[key] = op_grid(self._registry, op, groups)
        return self._grids[key]


# ---------------------------------------------------------------------------
# Keys and derived quantities
# ---------------------------------------------------------------------------


def shape_key(shape: Mapping[str, int], shape_dims: Sequence[str]) -> Tuple[Tuple[str, int], ...]:
    return tuple((dim, int(shape[dim])) for dim in shape_dims)


def config_record(provenance: Mapping[str, Any], threads: int) -> Dict[str, Any]:
    toolchain = provenance.get("toolchain", {})
    eigen = provenance.get("eigen", {})
    cpu = provenance.get("cpu", {})
    commit = eigen.get("commit") or ""
    return {
        "machine_config_id": provenance.get("machine_config_id"),
        "isa_target": toolchain.get("isa_target"),
        "compiler": f"{toolchain.get('compiler_id', 'unknown')} {toolchain.get('compiler_version', 'unknown')}",
        "compiler_id": toolchain.get("compiler_id"),
        "compiler_version": toolchain.get("compiler_version"),
        "cxx_flags": list(toolchain.get("cxx_flags", [])),
        "eigen_commit": commit or None,
        "eigen_commit_short": eigen.get("commit_short") or (commit[:9] if commit else None),
        "eigen_dirty": bool(eigen.get("dirty", False)),
        "threads": int(threads),
        "cpu_model": cpu.get("model"),
        "os": (provenance.get("os", {}) or {}).get("name"),
        "provenance_refs": [],
        "provenance_gaps": [],
    }


# ---------------------------------------------------------------------------
# Reduction
# ---------------------------------------------------------------------------


def _stat(measurement: Mapping[str, Any], quantity: str, field: str) -> Optional[float]:
    stats = measurement.get("stats") or {}
    entry = stats.get(quantity) or {}
    value = entry.get(field)
    return None if value is None else float(value)


def _intervals_overlap(a_center: float, a_half: float, b_center: float, b_half: float) -> bool:
    return (a_center - a_half) <= (b_center + b_half) and (b_center - b_half) <= (a_center + a_half)


def _timestamp(result: Mapping[str, Any]) -> str:
    return str((result.get("provenance", {}) or {}).get("timestamp_utc", ""))


def _now_utc() -> str:
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if epoch and epoch.strip().isdigit():
        moment = _datetime.datetime.fromtimestamp(int(epoch), tz=_datetime.timezone.utc)
    else:
        moment = _datetime.datetime.now(tz=_datetime.timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def choose_baseline(results: Iterable[Mapping[str, Any]], explicit: Optional[str] = None) -> Optional[str]:
    """The reference arm ratios are taken against.  Ambiguity is an error."""
    if explicit:
        return explicit
    arms: set = set()
    for result in results:
        arms.update((result.get("provenance", {}) or {}).get("arms", {}) or {})
        for row in result.get("measurements", []) or []:
            arms.add(row.get("arm"))
        for row in result.get("not_measured", []) or []:
            arms.add(row.get("arm"))
    arms.discard(EIGEN_ARM)
    arms.discard(None)
    if not arms:
        return None
    if len(arms) > 1:
        raise ReduceError(
            "several reference arms present (" + ", ".join(sorted(arms)) + "); pass --baseline to choose one",
            EXIT_USAGE,
        )
    return sorted(arms)[0]


def _op_metadata(registry: Mapping[str, Any], op: str) -> Dict[str, Any]:
    entry = registry["ops"][op]
    reference = dict(entry.get("reference", {}) or {})
    return {
        "display_name": entry.get("display_name", op),
        "long_name": entry.get("long_name", ""),
        "family": entry.get("family", ""),
        "description": entry.get("description", ""),
        "eigen_expr": entry.get("eigen_expr", ""),
        "eigen_class": entry.get("eigen_class", ""),
        "eigen_doc": entry.get("eigen_doc", ""),
        "shape_family": entry.get("shape_family", ""),
        "shape_dims": op_shape_dims(registry, op),
        "status": entry.get("status", "planned"),
        "base_mnemonic": entry.get("base_mnemonic", op),
        "variant": entry.get("variant", ""),
        "reference": reference,
        "flops_nominal": bool((entry.get("flops", {}) or {}).get("nominal", False)),
    }


def _blank_cell(
    registry: Mapping[str, Any],
    cache: "_OpCache",
    config_id: str,
    op: str,
    scalar: str,
    shape: Mapping[str, int],
    shape_dims: Sequence[str],
    threads: int,
) -> Dict[str, Any]:
    flops, nominal = flops_for(registry, op, scalar, shape)
    return {
        "config_id": config_id,
        "op": op,
        "op_family": registry["ops"][op].get("family", ""),
        "scalar": scalar,
        "shape": {dim: int(shape[dim]) for dim in shape_dims},
        "shape_dims": list(shape_dims),
        "shape_group": shape_group_of(registry, op, shape, cache.group_index(op)),
        "size_key": size_key(shape, shape_dims),
        "flops_per_iteration": flops,
        "flops_nominal": nominal,
        "arms": {},
        "ratio": None,
        "ratio_state": "not_measured",
    }


def reduce_results(
    results: Sequence[Mapping[str, Any]],
    registry: Mapping[str, Any],
    *,
    baseline: Optional[str] = None,
    on_conflict: str = "latest",
    inconclusive_rule: str = "mad-overlap",
    ops_toml_sha256: Optional[str] = None,
    generated_utc: Optional[str] = None,
    warn=None,
) -> Dict[str, Any]:
    """Merge result documents into the normalised intermediate.

    Pure with respect to the filesystem: it takes already-parsed result
    documents and returns the merged document.
    """
    if warn is None:
        warn = lambda message: None  # noqa: E731
    known_ops = registry["ops"]
    cache = _OpCache(registry)

    cells: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    configs: Dict[str, Dict[str, Any]] = {}
    arms_meta: Dict[str, Dict[str, Any]] = {}
    arms_meta_stamp: Dict[str, str] = {}
    conflicts: List[Dict[str, Any]] = []
    missing_configs: List[Dict[str, Any]] = []

    gaps_for_config: List[Mapping[str, Any]] = []

    def touch_config(provenance: Mapping[str, Any], threads: int, run_id: str) -> str:
        config_id = config_id_for(provenance, threads)
        # Not setdefault: its second argument is always evaluated, so the record
        # was rebuilt for every cell of every config it already held.
        record = configs.get(config_id)
        if record is None:
            record = configs[config_id] = config_record(provenance, threads)
            record["provenance_gaps"] = [dict(g) for g in gaps_for_config]
        else:
            # A caveat must survive being merged with a run that lacks it.
            # eigen_dirty is not part of config_id, so a clean run and a dirty run
            # of the same commit share a configuration; taking the first run's
            # value meant the "measured from a dirty worktree, not reproducible"
            # warning disappeared or appeared purely according to which filename
            # sorted first. A caveat is a property of the set: if any contributing
            # run carries it, the configuration carries it.
            if bool((provenance.get("eigen", {}) or {}).get("dirty", False)):
                record["eigen_dirty"] = True
        for gap in gaps_for_config:
            if gap not in record["provenance_gaps"]:
                record["provenance_gaps"].append(dict(gap))
        if run_id and run_id not in record["provenance_refs"]:
            record["provenance_refs"].append(run_id)
        return config_id

    def cell_for(config_id: str, op: str, scalar: str, shape, shape_dims, threads: int) -> Dict[str, Any]:
        key = (config_id, op, scalar, shape_key(shape, shape_dims), int(threads))
        cell = cells.get(key)
        if cell is None:
            cell = _blank_cell(registry, cache, config_id, op, scalar, shape, shape_dims, threads)
            cell["threads"] = int(threads)
            cells[key] = cell
        return cell

    for result in results:
        run_id = str(result.get("run_id", ""))
        provenance = result.get("provenance", {}) or {}
        timestamp = _timestamp(result)
        # A gap is the run telling you what it could NOT establish -- that Eigen ran
        # sequentially against a threaded vendor, that the CPU could not be pinned,
        # that the governor is unknown. Dropping them here meant every caveat the
        # harness took care to record died at the reducer and no published artifact
        # ever mentioned it, which is the opposite of what they exist for.
        run_gaps = [g for g in (result.get("provenance_gaps") or []) if isinstance(g, Mapping)]
        gaps_for_config[:] = run_gaps
        for arm, meta in (provenance.get("arms", {}) or {}).items():
            # Parsed, not string-compared: _parse_timestamp's own docstring warns
            # that fractional seconds are schema-legal and '.' < 'Z', so a raw
            # string compare ranks 12:00:00.500Z BEFORE the whole second it
            # follows -- which published the OLDER library version for the arm.
            if arm not in arms_meta or _parse_timestamp(timestamp) >= _parse_timestamp(
                arms_meta_stamp.get(arm) or None
            ):
                arms_meta[arm] = {
                    "kind": meta.get("kind", "reference" if arm != EIGEN_ARM else "eigen"),
                    "library_name": meta.get("library_name", arm),
                    "library_version": meta.get("library_version", "unknown"),
                    "library_path": meta.get("library_path"),
                    "threading_model": meta.get("threading_model"),
                    "interface": meta.get("interface"),
                }
                arms_meta_stamp[arm] = timestamp

        # --- measured rows -------------------------------------------------
        # The thread count this result file was configured with. A result file is
        # one measurement unit, so scope.threads holds a single value; if it ever
        # holds more, only the benchmark name can disambiguate a row.
        _scope_threads = [int(v) for v in ((result.get('scope', {}) or {}).get('threads', []) or [1])]
        scope_threads_for_rows = _scope_threads[0] if len(_scope_threads) == 1 else 1

        for row in result.get("measurements", []) or []:
            name = row.get("name", "")
            parsed = parse_benchmark_name(name)
            op = parsed["op"]
            if op not in known_ops:
                raise ReduceError(f"benchmark name {name!r} names op {op!r}, absent from ops.toml")
            declared = op_shape_dims(registry, op)
            if parsed["shape_dims"] != declared:
                raise ReduceError(
                    f"benchmark name {name!r} carries dimensions {parsed['shape_dims']} but ops.toml "
                    f"declares {declared} for {op}; the C++ registrations and the registry have diverged"
                )
            scalar = parsed["scalar"]
            if scalar not in (known_ops[op].get("scalars") or []):
                raise ReduceError(f"benchmark name {name!r} uses scalar {scalar!r}, not declared for {op}")
            # The name carries /threads:N only when the registration used
            # ->Threads(n); these benchmarks do not, so a name always parses back
            # as 1. Keying on that alone filed every measurement of an N-thread
            # run under a "1 thread(s)" configuration while the explicit
            # negatives, which come from scope.threads, built a second, entirely
            # empty N-thread configuration beside it. The run's own scope is what
            # was configured, so it decides; the name only overrides it when the
            # registration genuinely encoded a thread count.
            threads = parsed["threads"] if parsed.get("threads_in_name") else scope_threads_for_rows
            config_id = touch_config(provenance, threads, run_id)
            cell = cell_for(config_id, op, scalar, parsed["shape"], parsed["shape_dims"], threads)
            if row.get("shape_group"):
                cell["shape_group"] = row["shape_group"]
            if row.get("flops_per_iteration"):
                cell["flops_per_iteration"] = float(row["flops_per_iteration"])
            cell["flops_nominal"] = bool(row.get("flops_nominal", cell["flops_nominal"]))

            rate = _stat(row, "flop_rate", "median")
            rate_mad = _stat(row, "flop_rate", "mad") or 0.0
            entry = {
                "state": "measured",
                # The single place flops-per-second becomes GFLOP/s.
                "gflops": None if rate is None else rate / 1e9,
                "gflops_mad": rate_mad / 1e9,
                "time_s": _stat(row, "real_time_s", "median"),
                "time_mad_s": _stat(row, "real_time_s", "mad") or 0.0,
                "reps": int(row.get("repetitions", 1)),
                "cv": _stat(row, "real_time_s", "cv"),
                "run_id": run_id,
                "timestamp_utc": timestamp,
                "validated": row.get("validated"),
            }
            # A non-finite statistic is not a measurement. Rendered as a number it
            # becomes a table cell nobody questions, and JSON cannot represent it
            # portably, so it is restated as the gap it actually is (section 6).
            nonfinite = [
                key
                for key in ("gflops", "gflops_mad", "time_s", "time_mad_s", "cv")
                if isinstance(entry[key], float) and not math.isfinite(entry[key])
            ]
            if nonfinite:
                warn(f"{name}: non-finite {', '.join(nonfinite)}; recorded as not measured")
                entry = {
                    "state": "not_measured",
                    "reason": "runtime_error",
                    "detail": f"non-finite statistic ({', '.join(nonfinite)}) in the raw result",
                    "run_id": run_id,
                    "timestamp_utc": timestamp,
                }
            _install_arm(cell, parsed["arm"], entry, on_conflict, conflicts, warn)

        # --- explicit negatives --------------------------------------------
        scope = result.get("scope", {}) or {}
        scope_scalars = list(scope.get("scalars", []) or [])
        scope_threads = [int(value) for value in (scope.get("threads", []) or [1])]
        shape_groups = scope.get("shape_groups", {}) or {}
        for row in result.get("not_measured", []) or []:
            op = row.get("op")
            if op not in known_ops:
                raise ReduceError(f"not_measured entry names op {op!r}, absent from ops.toml")
            if row.get("reason") == "machine_unavailable":
                missing_configs.append(
                    {
                        "machine_config_id": provenance.get("machine_config_id"),
                        "op": op,
                        "reason": "machine_unavailable",
                    }
                )
            declared = op_shape_dims(registry, op)
            scalars = [row["scalar"]] if row.get("scalar") else [
                tag for tag in (known_ops[op].get("scalars") or []) if not scope_scalars or tag in scope_scalars
            ]
            threads_list = [int(row["threads"])] if row.get("threads") else scope_threads
            shapes = [row["shape"]] if row.get("shape") else cache.grid(op, shape_groups.get(op))
            entry = {
                "state": "not_measured",
                "reason": row.get("reason", "unaccounted"),
                "detail": row.get("detail"),
                "run_id": run_id,
                "timestamp_utc": timestamp,
            }
            for scalar in scalars:
                for threads in threads_list:
                    config_id = touch_config(provenance, threads, run_id)
                    for shape in shapes:
                        if sorted(shape) != sorted(declared):
                            raise ReduceError(
                                f"not_measured entry for {op} carries shape {sorted(shape)}, "
                                f"but ops.toml declares {declared}"
                            )
                        cell = cell_for(config_id, op, scalar, shape, declared, threads)
                        _install_arm(cell, row.get("arm"), dict(entry), on_conflict, conflicts, warn)

        # --- unaccounted: in scope, in neither list --------------------------
        _fill_unaccounted(registry, cache, result, cells, touch_config, warn)

    baseline_arm = choose_baseline(results, baseline)
    ordered = _finalise_cells(registry, cells, baseline_arm, inconclusive_rule)
    coverage = build_coverage(registry, ordered, sorted(configs), missing_configs)

    ops_used = sorted({cell["op"] for cell in ordered})
    return {
        "schema_version": MERGED_SCHEMA_VERSION,
        "kind": MERGED_KIND,
        "generated_utc": generated_utc or _now_utc(),
        "ops_toml_sha256": ops_toml_sha256,
        "reducer_version": REDUCER_VERSION,
        "baseline": baseline_arm,
        "configs": configs,
        "arms": arms_meta,
        "ops": {op: _op_metadata(registry, op) for op in ops_used},
        "cells": ordered,
        "coverage": coverage,
        "conflicts": conflicts,
    }


def _install_arm(
    cell: Dict[str, Any],
    arm: Optional[str],
    entry: Dict[str, Any],
    on_conflict: str,
    conflicts: List[Dict[str, Any]],
    warn,
) -> None:
    if not arm:
        raise ReduceError("a row carries no arm key")
    previous = cell["arms"].get(arm)
    if previous is None:
        cell["arms"][arm] = entry
        return
    if _same_measurement(previous, entry):
        return
    # A config whose numbers move between runs is information about the
    # machine, so a conflict is recorded even when it resolves silently.
    keep, drop = _resolve_conflict(previous, entry, on_conflict)
    conflicts.append(
        {
            "config_id": cell["config_id"],
            "op": cell["op"],
            "scalar": cell["scalar"],
            "shape": dict(cell["shape"]),
            "threads": cell.get("threads", 1),
            "arm": arm,
            "kept": _conflict_side(keep),
            "dropped": [_conflict_side(drop)],
            "policy": on_conflict,
        }
    )
    if on_conflict == "error":
        raise ReduceError(
            f"conflicting contributions for {cell['op']}/{arm}/{cell['scalar']} {cell['shape']} "
            f"in config {cell['config_id']}",
            EXIT_CONFLICT,
        )
    if on_conflict == "keep-all":
        keep = dict(keep)
        keep.setdefault("alternatives", []).append(_conflict_side(drop))
    else:
        warn(
            f"conflict for {cell['op']}/{arm}/{cell['scalar']} {cell['shape']}: keeping "
            f"run {keep.get('run_id')!r} under policy {on_conflict!r}"
        )
    cell["arms"][arm] = keep


def _same_measurement(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    if a.get("state") != b.get("state"):
        return False
    if a.get("state") == "measured":
        return a.get("gflops") == b.get("gflops") and a.get("time_s") == b.get("time_s")
    return a.get("reason") == b.get("reason")


#: Every stamp that cannot be parsed sorts before every one that can, so a
#: contribution carrying a real time always outranks one that carries none.
_EPOCH = _datetime.datetime.min.replace(tzinfo=_datetime.timezone.utc)


def _parse_timestamp(value: Any) -> "_datetime.datetime":
    """RFC 3339 stamp as an aware datetime.

    Never compare these as strings: fractional seconds are schema-legal and
    ``'.' < 'Z'``, so ``2026-08-01T12:00:00.500Z`` would sort BEFORE the whole
    second it follows and a re-measurement would be silently discarded in favour
    of the stale number it was meant to replace.
    """
    if not value:
        return _EPOCH
    text = str(value).strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        moment = _datetime.datetime.fromisoformat(text)
    except ValueError:
        return _EPOCH
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=_datetime.timezone.utc)
    return moment


def _resolve_conflict(previous: Dict[str, Any], candidate: Dict[str, Any], policy: str):
    # A measured row always outranks a not_measured one for the same key, under
    # EVERY policy. This precedence used to be skipped for "first", so a run that
    # merely declined to measure a cell -- an --ops filter, a crash, an excluded
    # group -- could erase a real measurement of it purely by being older.
    if previous.get("state") == "measured" and candidate.get("state") != "measured":
        return previous, candidate
    if candidate.get("state") == "measured" and previous.get("state") != "measured":
        return candidate, previous
    if policy == "first":
        # Oldest measurement, not "whichever file was read first". Inputs are
        # processed in sorted-pathname order, so the old behaviour made the winner
        # depend on what the contributions happened to be named -- renaming a file
        # changed the published number. "first" is the symmetric opposite of
        # "latest", so it ranks by the same clock.
        if _parse_timestamp(candidate.get("timestamp_utc")) < _parse_timestamp(previous.get("timestamp_utc")):
            return candidate, previous
        return previous, candidate
    if _parse_timestamp(candidate.get("timestamp_utc")) >= _parse_timestamp(previous.get("timestamp_utc")):
        return candidate, previous
    return previous, candidate


def _conflict_side(entry: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "run_id": entry.get("run_id"),
        "gflops": entry.get("gflops"),
        "state": entry.get("state"),
        "reason": entry.get("reason"),
        "timestamp_utc": entry.get("timestamp_utc"),
    }


def _fill_unaccounted(registry, cache, result, cells, touch_config, warn) -> None:
    """Cells inside a contribution's scope that appeared in neither list.

    Two of these are derivable facts rather than harness bugs and are labelled
    as such: an op whose ``reference.kind`` is ``none`` has no reference arm to
    measure, and an op whose ``status`` is not ``implemented`` has no benchmark
    source yet.  Everything else stays ``unaccounted``, which is always a bug.
    """
    scope = result.get("scope", {}) or {}
    provenance = result.get("provenance", {}) or {}
    run_id = str(result.get("run_id", ""))
    scope_ops = [op for op in (scope.get("ops", []) or []) if op in registry["ops"]]
    scope_arms = list(scope.get("arms", []) or [])
    scope_scalars = list(scope.get("scalars", []) or [])
    scope_threads = [int(value) for value in (scope.get("threads", []) or [1])]
    shape_groups = scope.get("shape_groups", {}) or {}
    for op in scope_ops:
        entry = registry["ops"][op]
        declared = op_shape_dims(registry, op)
        reference_kind = (entry.get("reference", {}) or {}).get("kind", "none")
        status = entry.get("status", "planned")
        scalars = [tag for tag in (entry.get("scalars") or []) if tag in scope_scalars]
        points = cache.grid(op, shape_groups.get(op))
        if not points or not scalars:
            continue
        # The config id is a function of (provenance, threads) alone, so it is
        # resolved once per op rather than once per point of the op's grid.
        config_ids = {threads: touch_config(provenance, threads, run_id) for threads in scope_threads}
        for shape in points:
            for scalar in scalars:
                for threads in scope_threads:
                    config_id = config_ids[threads]
                    key = (config_id, op, scalar, shape_key(shape, declared), threads)
                    cell = cells.get(key)
                    if cell is None:
                        cell = _blank_cell(registry, cache, config_id, op, scalar, shape, declared, threads)
                        cell["threads"] = threads
                        cells[key] = cell
                    for arm in scope_arms:
                        if arm in cell["arms"]:
                            continue
                        if arm != EIGEN_ARM and reference_kind == "none":
                            reason, detail = "no_reference_equivalent", (entry.get("reference", {}) or {}).get("reason")
                        elif status != "implemented":
                            reason, detail = "not_implemented", f"no benchmark source registers {op} yet"
                        else:
                            reason = "unaccounted"
                            detail = (
                                f"In scope of run {run_id!r} but reported by neither measurements nor "
                                "not_measured; this is a harness bug."
                            )
                            shape_text = "/".join(f"{dim}:{shape[dim]}" for dim in declared)
                            warn(f"unaccounted: {op}/{arm}/{scalar}/{shape_text} in run {run_id!r}")
                        cell["arms"][arm] = {
                            "state": "not_measured",
                            "reason": reason,
                            "detail": detail,
                            "run_id": run_id,
                            "timestamp_utc": _timestamp(result),
                        }


def _finalise_cells(registry, cells, baseline_arm, inconclusive_rule) -> List[Dict[str, Any]]:
    ordered = sorted(
        cells.values(),
        key=lambda cell: (
            cell["config_id"],
            cell["op"],
            cell["scalar"],
            cell["size_key"],
            tuple(cell["shape"][dim] for dim in cell["shape_dims"]),
            cell.get("threads", 1),
        ),
    )
    for cell in ordered:
        cell.setdefault("threads", 1)
        reference_kind = (registry["ops"][cell["op"]].get("reference", {}) or {}).get("kind", "none")
        eigen = cell["arms"].get(EIGEN_ARM)
        other = cell["arms"].get(baseline_arm) if baseline_arm else None
        if reference_kind == "none":
            cell["ratio"] = None
            cell["ratio_state"] = "no_reference_equivalent"
            continue
        if (
            eigen
            and other
            and eigen.get("state") == "measured"
            and other.get("state") == "measured"
            # `is not None`, not truthiness: a measured 0.0 is a measurement, and
            # treating it as absent gave the cell ratio_state "not_measured" and a
            # footnote claiming an arm was never measured, on a page whose coverage
            # manifest counted it as measured. Zero still cannot be a divisor, so
            # it is excluded explicitly below rather than by falsiness.
            and other.get("gflops") is not None
            and other.get("gflops") != 0.0
        ):
            cell["ratio"] = eigen["gflops"] / other["gflops"]
            if inconclusive_rule == "mad-overlap" and _intervals_overlap(
                eigen["gflops"], eigen.get("gflops_mad") or 0.0, other["gflops"], other.get("gflops_mad") or 0.0
            ):
                cell["ratio_state"] = "inconclusive"
            else:
                cell["ratio_state"] = "ok"
        elif (
            eigen
            and other
            and eigen.get("state") == "measured"
            and other.get("state") == "measured"
        ):
            # Both arms ran, so "not measured" would be a false statement about the
            # dataset. The ratio is still undefined -- a zero rate means the arm
            # completed no work per unit time, which is a broken measurement, not a
            # result -- so it gets its own state rather than borrowing the absent one.
            cell["ratio"] = None
            cell["ratio_state"] = "degenerate"
        else:
            cell["ratio"] = None
            cell["ratio_state"] = "not_measured"
    return ordered


def build_coverage(registry, cells, config_ids, missing_configs) -> Dict[str, Any]:
    ops: Dict[str, Dict[str, Any]] = {}
    scalars: Dict[str, Dict[str, int]] = {}
    totals = {"measured": 0, "not_measured": 0, "unaccounted": 0}
    for cell in cells:
        op_entry = ops.setdefault(
            cell["op"], {"measured": 0, "not_measured": 0, "unaccounted": 0, "arms": []}
        )
        scalar_entry = scalars.setdefault(cell["scalar"], {"measured": 0, "not_measured": 0})
        for arm, arm_entry in cell["arms"].items():
            if arm not in op_entry["arms"]:
                op_entry["arms"].append(arm)
            if arm_entry.get("state") == "measured":
                op_entry["measured"] += 1
                scalar_entry["measured"] += 1
                totals["measured"] += 1
            else:
                op_entry["not_measured"] += 1
                scalar_entry["not_measured"] += 1
                totals["not_measured"] += 1
                if arm_entry.get("reason") == "unaccounted":
                    op_entry["unaccounted"] += 1
                    totals["unaccounted"] += 1
    for entry in ops.values():
        entry["arms"].sort()
    deduped: List[Dict[str, Any]] = []
    for entry in missing_configs:
        if entry not in deduped:
            deduped.append(entry)
    return {
        "configs": list(config_ids),
        "ops": ops,
        "scalars": scalars,
        "totals": totals,
        "missing_configs": deduped,
    }


# ---------------------------------------------------------------------------
# Merging an existing merged file
# ---------------------------------------------------------------------------


def merged_reference_arms(document: Mapping[str, Any]) -> set:
    """Reference arms a merged document carries, `eigen` excluded."""
    arms: set = set(document.get("arms") or {})
    for cell in document.get("cells", []) or []:
        arms.update(cell.get("arms") or {})
    arms.discard(EIGEN_ARM)
    arms.discard(None)
    return arms


def _merged_cell_key(cell: Mapping[str, Any]) -> Tuple[Any, ...]:
    """Identity of a cell in a merged document."""
    return (
        cell["config_id"],
        cell["op"],
        cell["scalar"],
        shape_key(cell["shape"], cell["shape_dims"]),
        cell.get("threads", 1),
    )


def merge_merged(
    base: Mapping[str, Any],
    addition: Mapping[str, Any],
    registry,
    on_conflict: str,
    warn,
    baseline: Optional[str] = None,
    inconclusive_rule: str = "mad-overlap",
) -> Dict[str, Any]:
    """Additively fold `addition` into `base`, both merged documents.

    `base` is folded into in place and must not be read by the caller afterwards:
    it is loaded, passed straight in and dropped, and the JSON round-trip that
    used to deep-copy it was two thirds of the cost of a merge.

    Ambiguity is re-checked here, not inherited. Keeping the base's reference arm
    while folding in a second vendor's numbers leaves every new cell with a ratio
    against an arm it never measured -- `ratio_state: "not_measured"` on a cell
    whose two arms were both measured, under a column header naming the wrong
    library. `reduce.py a.json b.json` already refuses that; the `--merge` path
    has to refuse it identically or the two documented routes to the same store
    disagree.
    """
    arms = merged_reference_arms(base) | merged_reference_arms(addition)
    if baseline is None and len(arms) > 1:
        raise ReduceError(
            "several reference arms present (" + ", ".join(sorted(arms)) + "); pass --baseline to choose one",
            EXIT_USAGE,
        )
    merged: Dict[str, Any] = dict(base)
    conflicts: List[Dict[str, Any]] = list(merged.get("conflicts", []))
    cells: List[Dict[str, Any]] = list(merged.get("cells", []))
    index: Dict[Tuple[Any, ...], Dict[str, Any]] = {_merged_cell_key(cell): cell for cell in cells}
    for cell in addition.get("cells", []):
        key = _merged_cell_key(cell)
        target = index.get(key)
        if target is None:
            # A shallow copy down to the arm entries, which are the only part
            # _install_arm and _finalise_cells write to; `addition` is left alone
            # without deep-copying every shape and every statistic.
            adopted = dict(cell)
            adopted["arms"] = {arm: dict(entry) for arm, entry in (cell.get("arms") or {}).items()}
            index[key] = adopted
            cells.append(adopted)
            continue
        for arm, entry in cell["arms"].items():
            _install_arm(target, arm, dict(entry), on_conflict, conflicts, warn)
    for section in ("configs", "arms", "ops"):
        for key, value in (addition.get(section) or {}).items():
            merged.setdefault(section, {}).setdefault(key, value)
    for config_id, record in (addition.get("configs") or {}).items():
        refs = merged["configs"][config_id].setdefault("provenance_refs", [])
        for ref in record.get("provenance_refs", []):
            if ref not in refs:
                refs.append(ref)
    merged["conflicts"] = conflicts + [c for c in addition.get("conflicts", []) if c not in conflicts]
    merged["baseline"] = baseline or merged.get("baseline") or addition.get("baseline")
    merged["cells"] = _finalise_cells(
        registry,
        dict(enumerate(cells)),
        merged.get("baseline"),
        inconclusive_rule,
    )
    merged["coverage"] = build_coverage(
        registry,
        merged["cells"],
        sorted(merged.get("configs", {})),
        list(merged.get("coverage", {}).get("missing_configs", []))
        + list(addition.get("coverage", {}).get("missing_configs", [])),
    )
    merged["generated_utc"] = addition.get("generated_utc") or merged.get("generated_utc")
    return merged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _validator(schema_path: str):
    try:
        import jsonschema  # type: ignore
    except ImportError:
        return None
    try:
        with open(schema_path, "rb") as handle:
            schema = json.load(handle)
    except OSError as exc:
        raise ReduceError(f"cannot read schema {schema_path}: {exc}")
    return jsonschema.Draft202012Validator(schema)


def build_parser() -> argparse.ArgumentParser:
    parser = UsageErrorArgumentParser(
        prog="reduce.py",
        description="Merge Eigen benchmark comparison result files into the normalised intermediate.",
    )
    parser.add_argument("files", nargs="*", metavar="FILE", help="result files; '-' or none reads paths from stdin")
    parser.add_argument("--glob", action="append", default=[], metavar="PATTERN", help="repeatable glob of result files")
    parser.add_argument("--out", default="-", metavar="PATH", help="merged output; '-' is stdout (default)")
    parser.add_argument("--merge", default=None, metavar="INTO", help="additively merge into an existing merged file")
    parser.add_argument("--ops-toml", default=DEFAULT_OPS_TOML, metavar="PATH")
    parser.add_argument("--schema", default=DEFAULT_SCHEMA, metavar="PATH")
    parser.add_argument("--validate", dest="validate", action="store_true", default=True)
    parser.add_argument("--no-validate", dest="validate", action="store_false")
    parser.add_argument("--skip-invalid", action="store_true", help="warn and continue instead of failing")
    parser.add_argument("--baseline", default=None, metavar="ARM")
    parser.add_argument("--on-conflict", choices=["latest", "first", "error", "keep-all"], default="latest")
    parser.add_argument("--inconclusive-rule", choices=["mad-overlap", "none"], default="mad-overlap")
    parser.add_argument("--pretty", dest="pretty", action="store_true", default=True)
    parser.add_argument("--compact", dest="pretty", action="store_false")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    return parser


def resolve_inputs(files: Sequence[str], globs: Sequence[str], stdin=None) -> List[str]:
    paths: List[str] = []
    read_stdin = not files
    for item in files:
        if item == "-":
            read_stdin = True
        else:
            paths.append(item)
    for pattern in globs:
        paths.extend(sorted(_glob.glob(pattern, recursive=True)))
    if read_stdin and not globs:
        stream = sys.stdin if stdin is None else stdin
        if stream is not None and not stream.isatty():
            paths.extend(line.strip() for line in stream.read().splitlines() if line.strip())
    seen: set = set()
    unique: List[str] = []
    for path in sorted(paths):
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    def warn(message: str) -> None:
        print(f"reduce.py: {message}", file=sys.stderr)

    try:
        ops = load_ops_registry(args.ops_toml)
        registry = ops.data
        paths = resolve_inputs(args.files, split_list(args.glob))
        if not paths:
            warn("no input files resolved")
            return EXIT_NO_INPUT
        validator = _validator(args.schema) if args.validate else None
        if args.validate and validator is None:
            warn("jsonschema is unavailable; inputs were not validated")

        results: List[Mapping[str, Any]] = []
        for path in paths:
            try:
                with open(path, "rb") as handle:
                    document = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                if args.skip_invalid:
                    warn(f"skipping {path}: {exc}")
                    continue
                raise ReduceError(f"cannot read result file {path}: {exc}")
            if validator is not None:
                errors = sorted(validator.iter_errors(document), key=lambda error: list(error.path))
                if errors:
                    detail = "; ".join(f"{'/'.join(str(p) for p in e.path)}: {e.message}" for e in errors[:5])
                    if args.skip_invalid:
                        warn(f"skipping {path}: schema validation failed: {detail}")
                        continue
                    raise ReduceError(f"{path} failed schema validation: {detail}")
            results.append(document)
        if not results:
            warn("every input was skipped as invalid")
            return EXIT_NO_INPUT

        merged = reduce_results(
            results,
            registry,
            baseline=args.baseline,
            on_conflict=args.on_conflict,
            inconclusive_rule=args.inconclusive_rule,
            ops_toml_sha256=ops.sha256,
            warn=warn if args.verbose else (lambda message: None),
        )
        if args.merge:
            try:
                with open(args.merge, "rb") as handle:
                    existing = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                raise ReduceError(f"cannot read merged file {args.merge}: {exc}")
            merged = merge_merged(
                existing,
                merged,
                registry,
                args.on_conflict,
                warn,
                baseline=args.baseline,
                inconclusive_rule=args.inconclusive_rule,
            )

        text = json.dumps(merged, indent=2, sort_keys=True) if args.pretty else json.dumps(merged, sort_keys=True)
        if args.out == "-":
            sys.stdout.write(text + "\n")
        else:
            directory = os.path.dirname(os.path.abspath(args.out))
            os.makedirs(directory, exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as handle:
                handle.write(text + "\n")
        return EXIT_OK
    except PipelineError as exc:
        warn(str(exc))
        return exc.code
    except ValueError as exc:
        warn(str(exc))
        return EXIT_INPUT_INVALID


if __name__ == "__main__":
    sys.exit(main())
