# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

"""Logic shared by ``run.py``, ``reduce.py``, ``render.py`` and ``plots.py``.

The four scripts form one pipeline -- ``run.py`` writes the files ``reduce.py``
reads, and ``render.py``/``plots.py`` read what ``reduce.py`` writes -- so a
rule implemented twice is a rule that can disagree with itself between two
stages. A benchmark name one stage accepts and the next rejects, or a flop
formula one stage evaluates and the next refuses, breaks the pipeline in the
middle with a valid file already on disk. Everything that more than one stage
has to agree about therefore lives here and only here.

See ``benchmarks/comparison/CONTRACTS.md``; it remains the source of truth for
the name grammar, the registry shape and the exit codes.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

#: CONTRACTS.md section 4: argparse's own usage exit code is 2, the harness
#: contract says 1. Every script in the pipeline uses the same value.
EXIT_USAGE = 1

#: The exit code every script uses for "the input is not what it claims to be":
#: ``run.py``'s EXIT_CONFIG, ``reduce.py``'s EXIT_INPUT_INVALID, ``render.py``'s
#: and ``plots.py``'s EXIT_INPUT are all this number.
EXIT_INPUT = 2

#: The arm key of Eigen itself. It is implicit and always present, and is never
#: a *reference* arm, so ratio and baseline logic everywhere excludes it by name.
EIGEN_ARM = "eigen"


class PipelineError(Exception):
    """A fatal, user-facing error carrying the process exit code.

    The four scripts spell their own error types differently and spell the code
    attribute differently -- ``run.py`` reads ``exit_code``, the other three read
    ``code`` -- so both names are set here and every script's error type derives
    from this one. That lets shared code raise a single exception that each
    script's ``main`` already catches.
    """

    def __init__(self, message: str, code: int = EXIT_INPUT):
        super().__init__(message)
        self.code = code
        self.exit_code = code


# ---------------------------------------------------------------------------
# 1. The benchmark name grammar (CONTRACTS.md section 1)
# ---------------------------------------------------------------------------

#: Suffixes Google Benchmark appends for registration forms a comparison
#: benchmark must not use. Seeing one means the C++ side used Range/Complexity/
#: UseManualTime/etc., which the reducer cannot key on; it is an error, not a
#: field to ignore. "threads" is the one reserved field that is accepted.
RESERVED_DIMS = frozenset(
    {
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
)

SCALAR_TAGS = ("f16", "bf16", "f32", "f64", "c32", "c64")

_OP_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_ARM_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_DIM_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def parse_benchmark_name(run_name: str) -> dict:
    """Parse a Google Benchmark ``run_name`` into its grammar fields.

    ``run_name`` is used rather than ``name`` because aggregate rows append
    ``_mean``/``_median``/``_stddev``/``_cv`` to ``name``, and stripping such a
    suffix is unsafe: ``_`` is legal inside both arm keys and op keys
    (``TRSM_LLNN``).

    One parser, not one per stage: ``run.py`` writes the file ``reduce.py``
    reads, so a name the producer accepts and the consumer rejects strands a
    measurement in a file that cannot be reduced. Every check either side used
    to make alone is made here for both.
    """
    if not isinstance(run_name, str) or not run_name:
        raise ValueError(f"empty benchmark name {run_name!r}")
    fields = run_name.split("/")
    if len(fields) < 4:
        raise ValueError(f"benchmark name {run_name!r} has fewer than four fields")
    op, arm, scalar = fields[0], fields[1], fields[2]
    if not _OP_RE.match(op):
        raise ValueError(f"illegal op field {op!r} in {run_name!r}")
    if not _ARM_RE.match(arm):
        raise ValueError(f"illegal arm field {arm!r} in {run_name!r}")
    if scalar not in SCALAR_TAGS:
        raise ValueError(f"illegal scalar field {scalar!r} in {run_name!r}")
    threads = 1
    threads_in_name = False
    shape: dict[str, int] = {}
    dim_order: list[str] = []
    for entry in fields[3:]:
        key, sep, value = entry.partition(":")
        if not sep or not value.isdigit():
            raise ValueError(f"unparsable field {entry!r} in {run_name!r}")
        if key == "threads":
            threads = int(value)
            threads_in_name = True
        elif key in RESERVED_DIMS:
            raise ValueError(f"unsupported registration form {key!r} in {run_name!r}")
        elif not _DIM_RE.match(key):
            raise ValueError(f"illegal dimension name {key!r} in {run_name!r}")
        elif key in shape:
            raise ValueError(f"duplicate dimension {key!r} in {run_name!r}")
        else:
            dim_order.append(key)
            shape[key] = int(value)
    if not dim_order:
        raise ValueError(f"benchmark name {run_name!r} carries no shape dimension")
    return {
        "op": op,
        "arm": arm,
        "scalar": scalar,
        "shape": shape,
        "shape_dims": dim_order,
        "threads": threads,
        # Whether the registration actually encoded a thread count. Absent, the
        # 1 above is a default, not an observation -- a caller keying on it would
        # silently file an N-thread run as single-threaded.
        "threads_in_name": threads_in_name,
    }


def is_arm_key(key: str) -> bool:
    """True when ``key`` is a legal reference-arm key (CONTRACTS.md section 1.1).

    The same grammar governs the arm field of a benchmark name and the arm keys a
    machine profile declares, so a profile cannot name an arm no benchmark could.
    """
    return bool(_ARM_RE.match(key))


def format_shape_suffix(shape_dims: Sequence[str], values: Sequence[int]) -> str:
    """Render the ``dim:value`` part of a benchmark name."""
    if len(shape_dims) != len(values):
        raise ValueError(f"{len(shape_dims)} dimension names against {len(values)} values")
    return "/".join(f"{name}:{int(value)}" for name, value in zip(shape_dims, values))


# ---------------------------------------------------------------------------
# 2. Derived quantities every stage has to agree on
# ---------------------------------------------------------------------------


def size_key(shape: Mapping[str, int], shape_dims: Sequence[str] | None = None) -> int:
    """Plot abscissa: the geometric mean of the dimensions, rounded.

    Defined once here so every plot and table agrees. For a square GEMM it is
    the order; for 10000x8x8 it is 86. A non-positive dimension has no geometric
    mean, so it yields 0 rather than a complex number or a domain error.
    """
    dims = list(shape_dims) if shape_dims else sorted(shape)
    values = [float(shape[dim]) for dim in dims]
    if not values or any(value <= 0.0 for value in values):
        return 0
    product = 1.0
    for value in values:
        product *= value
    return int(round(product ** (1.0 / len(values))))


def compact_shape(cell: Mapping[str, Any]) -> str:
    """``64x64x64``: a cell's shape as one token, where the op names the dims."""
    return "x".join(str(cell["shape"][dim]) for dim in cell["shape_dims"])


def _compiler_major(version: str) -> str:
    head = str(version).strip().split(".", 1)[0]
    digits = "".join(ch for ch in head if ch.isdigit())
    return digits or head or "0"


def config_id_for(provenance: Mapping[str, Any], threads: int) -> str:
    """CONTRACTS.md section 5.2. A compiler *minor* bump does not split a config."""
    toolchain = provenance.get("toolchain", {})
    eigen = provenance.get("eigen", {})
    commit = eigen.get("commit") or ""
    short = eigen.get("commit_short") or (commit[:9] if commit else "unknown")
    parts = [
        str(provenance.get("machine_config_id", "unknown")),
        str(toolchain.get("isa_target", "unknown")),
        f"{str(toolchain.get('compiler_id', 'unknown')).lower()}"
        f"{_compiler_major(toolchain.get('compiler_version', '0'))}",
        str(short),
        f"t{int(threads)}",
    ]
    return "__".join(parts)


# ---------------------------------------------------------------------------
# 3. The operation registry (ops.toml)
# ---------------------------------------------------------------------------

#: ``//`` and ``%`` are deliberately absent. They used to be accepted here and
#: rejected by the reducer, so a formula using them produced a result file that
#: broke the pipeline one stage later; the stricter reading wins because a flop
#: count is a continuous quantity and integer division of one is a mistake.
_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)

#: ops.toml holds a few dozen formulas that are each evaluated once per grid
#: point per scalar per arm -- thousands of times for the same handful of
#: strings -- so the parse is cached and only the walk is repeated.
_FORMULA_CACHE: dict[str, ast.Expression] = {}


def _formula_ast(expression: str) -> ast.Expression:
    tree = _FORMULA_CACHE.get(expression)
    if tree is None:
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise PipelineError(f"unparsable formula {expression!r}: {exc}") from exc
        _FORMULA_CACHE[expression] = tree
    return tree


def evaluate_flop_formula(expression: str, shape: Mapping[str, int]) -> float:
    """Evaluate an ``ops.toml`` ``flops.real`` formula over a shape.

    Deliberately a restricted AST walk rather than ``eval``: ``ops.toml`` is data
    contributed alongside benchmark results, so it must not be able to run code.
    Names resolve to shape dimensions and nothing else, and everything is
    computed in ``float`` so a cubic term in a five-digit dimension cannot
    overflow.
    """

    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant):
            # `True` is an int in Python; a boolean in a flop count is a typo.
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise PipelineError(f"non-numeric constant in flop formula {expression!r}")
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id not in shape:
                raise PipelineError(f"formula {expression!r} references unknown dimension {node.id!r}")
            return float(shape[node.id])
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = walk(node.operand)
            return operand if isinstance(node.op, ast.UAdd) else -operand
        if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                if right == 0.0:
                    raise PipelineError(f"division by zero in flop formula {expression!r}")
                return left / right
            return left**right
        raise PipelineError(f"illegal construct in formula {expression!r}")

    return float(walk(_formula_ast(expression)))


def scalar_flops(expression: str, flop_scale: float, shape: Mapping[str, int]) -> float:
    """Scalar flop count of one iteration.

    ``flops.real`` already carries the real multiply-add factor of two, so GEMM
    reads as the familiar ``2*m*n*k``; ``flop_scale / 2`` lifts that real count
    to the scalar type actually measured.
    """
    return evaluate_flop_formula(expression, shape) * float(flop_scale) / 2.0


@dataclass(frozen=True)
class OpsRegistry:
    """Parsed ``ops.toml`` plus the digest of the bytes it was parsed from."""

    data: Mapping[str, Any]
    sha256: str
    path: Path | None = None

    @property
    def ops(self) -> Mapping[str, Any]:
        return self.data.get("ops", {})

    @property
    def scalars(self) -> Mapping[str, Any]:
        return self.data.get("scalars", {})

    @property
    def shape_families(self) -> Mapping[str, Any]:
        return self.data.get("shape_families", {})

    @property
    def flop_counter_name(self) -> str:
        return str(self.data.get("flop_counter_name", "GFLOPS"))

    def op(self, key: str) -> Mapping[str, Any]:
        try:
            return self.ops[key]
        except KeyError:
            raise PipelineError(f"unknown operation {key!r}: not a key of [ops] in ops.toml") from None

    def family(self, op_key: str) -> Mapping[str, Any]:
        name = self.op(op_key).get("shape_family")
        if name not in self.shape_families:
            raise PipelineError(f"operation {op_key!r} names unknown shape family {name!r}")
        return self.shape_families[name]

    def shape_dims(self, op_key: str) -> tuple[str, ...]:
        return tuple(self.family(op_key)["dims"])

    def default_groups(self, op_key: str) -> tuple[str, ...]:
        return tuple(self.family(op_key).get("default_groups", ()))

    def group_names(self, op_key: str) -> tuple[str, ...]:
        return tuple(str(group["name"]) for group in self.family(op_key).get("groups", ()))

    def shape_points(self, op_key: str, groups: Sequence[str]) -> list[tuple[str, tuple[int, ...]]]:
        """Ordered union of the selected groups' points, first occurrence winning."""
        family = self.family(op_key)
        by_name = {str(group["name"]): group for group in family.get("groups", ())}
        unknown = [name for name in groups if name not in by_name]
        if unknown:
            raise PipelineError(
                f"operation {op_key!r} has no shape group(s) {', '.join(sorted(unknown))}; "
                f"known groups: {', '.join(sorted(by_name))}"
            )
        arity = len(family["dims"])
        seen: set[tuple[int, ...]] = set()
        points: list[tuple[str, tuple[int, ...]]] = []
        for name in groups:
            for raw in by_name[name].get("points", ()):
                point = tuple(int(value) for value in raw)
                if len(point) != arity:
                    raise PipelineError(
                        f"group {name!r} of {op_key!r} has a point of arity {len(point)}, expected {arity}"
                    )
                if point in seen:
                    continue
                seen.add(point)
                points.append((name, point))
        return points

    def flops_per_iteration(self, op_key: str, scalar: str, shape: Mapping[str, int]) -> float:
        """Scalar flop count of one iteration, from the registry formula."""
        formula = self.op(op_key).get("flops", {}).get("real")
        if not formula:
            raise PipelineError(f"operation {op_key!r} has no flops.real formula")
        return scalar_flops(str(formula), self.scalars[scalar]["flop_scale"], shape)

    def flops_nominal(self, op_key: str) -> bool:
        return bool(self.op(op_key).get("flops", {}).get("nominal", False))

    def implemented_ops(self) -> list[str]:
        return [key for key, op in self.ops.items() if op.get("status") == "implemented"]

    def target_for(self, op_key: str) -> str | None:
        """CMake target that carries this op, derived from its ``source`` stem."""
        source = self.op(op_key).get("source")
        if not source:
            return None
        return Path(str(source)).stem


def load_ops_registry(path: str | Path) -> OpsRegistry:
    """Read, parse and digest ``ops.toml`` from one buffer.

    One read, not two: the digest has to be of exactly the bytes that were
    parsed, and re-reading the file to hash it both doubles the I/O and leaves a
    window in which the two could differ.
    """
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise PipelineError(f"cannot read ops registry {path}: {exc}") from exc
    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise PipelineError(f"invalid ops registry {path}: {exc}") from exc
    for required in ("ops", "scalars", "shape_families"):
        if required not in data:
            raise PipelineError(f"ops registry {path} is missing [{required}]")
    return OpsRegistry(data=data, sha256=hashlib.sha256(raw).hexdigest(), path=path)


# ---------------------------------------------------------------------------
# 4. The merged-document front end shared by render.py and plots.py
# ---------------------------------------------------------------------------


def load_merged(source: str) -> dict:
    """Load a merged document from a path, or from stdin when ``source`` is '-'."""
    try:
        if source == "-":
            document = json.load(sys.stdin)
        else:
            with open(source, "rb") as handle:
                document = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"cannot parse merged input {source}: {exc}") from exc
    if not isinstance(document, dict) or "cells" not in document:
        raise PipelineError(f"{source} is not an eigen-benchmark-comparison-merged document")
    return document


def _within_cell_order(cell: Mapping[str, Any]) -> tuple:
    return (
        cell.get("size_key", 0),
        tuple(cell["shape"][dim] for dim in cell["shape_dims"]),
        cell.get("threads", 1),
    )


def select_cells(merged: Mapping[str, Any], options: Any, *, by_config: bool = False) -> list[Mapping[str, Any]]:
    """Cells matching ``options``' config/op/scalar selection, in a stable order.

    ``options`` is any object carrying ``configs``, ``ops`` and ``scalars``;
    ``render.py`` and ``plots.py`` each have their own options dataclass but
    select identically. ``by_config`` prefixes the sort with the table identity
    ``render.py`` groups on -- ``plots.py`` sorts within an already narrowed
    slice and would only be re-sorting by three constants.
    """
    configs = set(options.configs) if options.configs else None
    ops = set(options.ops) if options.ops else None
    scalars = set(options.scalars) if options.scalars else None
    chosen = [
        cell
        for cell in merged.get("cells", [])
        if (configs is None or cell["config_id"] in configs)
        and (ops is None or cell["op"] in ops)
        and (scalars is None or cell["scalar"] in scalars)
    ]
    if by_config:
        return sorted(
            chosen,
            key=lambda cell: (cell["config_id"], cell["scalar"], cell["op"]) + _within_cell_order(cell),
        )
    return sorted(chosen, key=_within_cell_order)


def arms_in(cells: Iterable[Mapping[str, Any]], baseline: str | None) -> list[str]:
    """Arms present across ``cells``: Eigen first, then the baseline, then the rest."""
    present: set = set()
    for cell in cells:
        present.update(cell.get("arms") or {})
    ordered = [EIGEN_ARM] if EIGEN_ARM in present else []
    if baseline and baseline in present and baseline != EIGEN_ARM:
        ordered.append(baseline)
    ordered.extend(sorted(arm for arm in present if arm not in ordered))
    return ordered


def resolve_baseline(merged: Mapping[str, Any], requested: str | None) -> str | None:
    """The arm the ratios were computed against, refusing to relabel them.

    `ratio` is computed once, by reduce.py, against `merged["baseline"]`. Nothing
    downstream can recompute it: a cell keeps only the arms' rates and that one
    number. So a presentation-time `--baseline` naming a different arm would
    retitle the page and the ratio column while the ratios underneath still
    divide by the original arm -- publishing, for example, "Eigen vs OpenBLAS"
    over Eigen/Accelerate numbers, on a page whose own neighbouring column names
    Accelerate. Choosing a different baseline is a reduce-time decision.
    """
    recorded = merged.get("baseline")
    if requested is None or requested == recorded:
        return recorded
    if recorded is None:
        raise PipelineError(
            f"--baseline {requested!r} was given but this document records no baseline, so it has no "
            "ratios to label; re-run reduce.py --baseline to compute them",
            EXIT_USAGE,
        )
    raise PipelineError(
        f"--baseline {requested!r} does not match the baseline these ratios were computed against "
        f"({recorded!r}). Relabelling would publish a ratio against one library under another "
        f"library's name; re-run reduce.py --baseline {requested} instead",
        EXIT_USAGE,
    )


def arm_display(merged: Mapping[str, Any], arm: str) -> str:
    meta = (merged.get("arms") or {}).get(arm) or {}
    return meta.get("library_name") or arm


def config_parts(merged: Mapping[str, Any], config_id: str, *, fallback: str | None = None) -> list[str]:
    """The fields that identify a measurement configuration to a reader.

    Shared so a table and the plot of the same cell cannot describe that cell
    differently. In particular an uncommitted worktree cannot be reconstructed
    from the recorded commit, so a rate measured against one says so *wherever*
    it is shown -- the plot used to drop that caveat while the table kept it.
    """
    config = (merged.get("configs") or {}).get(config_id) or {}
    parts = [
        config.get("cpu_model") or config.get("machine_config_id") or fallback,
        config.get("isa_target"),
        config.get("compiler"),
        f"{config.get('threads', 1)} thread(s)",
    ]
    if config.get("eigen_dirty"):
        parts.append("dirty worktree")
    return [str(part) for part in parts if part]


# ---------------------------------------------------------------------------
# 5. Command line
# ---------------------------------------------------------------------------


class UsageErrorArgumentParser(argparse.ArgumentParser):
    """argparse exits 2 on a usage error; CONTRACTS.md section 4 says 1."""

    def error(self, message: str) -> None:  # type: ignore[override]
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}", file=sys.stderr)
        raise SystemExit(EXIT_USAGE)


def split_list(values: Sequence[str] | None) -> list[str]:
    """Flatten repeated and comma-separated option values, preserving order.

    Duplicates are dropped: every consumer of these lists treats them as sets of
    selectors, and a repeated ``--format`` used to render the same file twice.
    """
    out: list[str] = []
    for value in values or ():
        for item in str(value).split(","):
            item = item.strip()
            if item and item not in out:
                out.append(item)
    return out
