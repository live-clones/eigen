#!/usr/bin/env python3
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

"""Reconcile a comparison binary's actual registrations against ops.toml.

The size grid in the C++ is a hand transcription of the one in `ops.toml`, and
the two are joined only by the benchmark name.  The pytest suite checks that
join against a committed capture of `--benchmark_list_tests`, which is fast and
needs no compiler, but it can only ever describe the binary that was current
when the capture was taken: build a binary whose grid has drifted and the whole
suite still passes.

This runs the binary that was actually built and checks BOTH directions:

  registered -> registry   every name parses, names an op ops.toml calls
                           implemented, uses a declared scalar, carries the
                           declared dimensions in the declared order, and lands
                           on a point of the op's shape family;

  registry -> registered   every point of the grid of every op THIS BINARY
                           registers is registered for the eigen arm.

The second direction is scoped to the ops the binary claims because each binary
carries a subset of the registry: checking all of ops.toml here would flag every
op that lives in a sibling source file.  The complementary check -- an op that
ops.toml calls implemented and that NO binary registers -- needs the listings of
all of them at once, and lives in tests/test_ops_registry.py against the
committed capture.

The point-level direction is the one a snapshot cannot give you.  Without it, adding
a point to `ops.toml` and forgetting the matching entry in the C++ SIZES macro
passes every gate, and the omission surfaces only much later as a cell the
harness reports as `not_implemented` -- which reads on a published page as
"Eigen does not implement this" rather than "the grid drifted".

Usage:
    reconcile_registrations.py <benchmark-executable> [--ops-toml PATH]
"""

import argparse
import subprocess
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
COMPARISON_DIR = TESTS_DIR.parent
if str(COMPARISON_DIR) not in sys.path:
    sys.path.insert(0, str(COMPARISON_DIR))

from _common import load_ops_registry, parse_benchmark_name  # noqa: E402

EXIT_OK = 0
EXIT_DRIFT = 1
EXIT_USAGE = 2


def _all_points(registry, op: str):
    """Every point of every group the op's family declares.

    Reconciliation is about the whole grid, not the subset a particular run
    selects, so this deliberately ignores default_groups.
    """
    return registry.shape_points(op, registry.group_names(op))


def list_registrations(executable: Path) -> list[str]:
    proc = subprocess.run(
        [str(executable), "--benchmark_list_tests=true"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(f"{executable} --benchmark_list_tests failed:\n{proc.stderr}")
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def reconcile(names: list[str], registry) -> list[str]:
    problems: list[str] = []
    registered_points: dict[str, set[tuple]] = {}
    registered_scalars: dict[str, set[str]] = {}

    for name in names:
        try:
            parsed = parse_benchmark_name(name)
        except ValueError as exc:
            problems.append(f"unparseable registration {name!r}: {exc}")
            continue
        op = parsed["op"]
        entry = registry.ops.get(op)
        if entry is None:
            problems.append(f"{name}: op {op!r} is registered but absent from ops.toml")
            continue
        if entry.get("status") != "implemented":
            problems.append(f"{name}: registered although ops.toml calls {op!r} {entry.get('status')!r}")
        if parsed["scalar"] not in (entry.get("scalars") or []):
            problems.append(f"{name}: scalar {parsed['scalar']!r} is not declared for {op}")
        declared = list(registry.shape_dims(op))
        if list(parsed["shape_dims"]) != declared:
            problems.append(
                f"{name}: dimensions {list(parsed['shape_dims'])} but ops.toml declares {declared} for {op}"
            )
            continue
        point = tuple(parsed["shape"][dim] for dim in declared)
        if list(point) not in [list(values) for _, values in _all_points(registry, op)]:
            problems.append(f"{name}: shape {list(point)} is not a point of {op}'s shape family")
        if parsed["arm"] == "eigen":
            registered_points.setdefault(op, set()).add(point)
            registered_scalars.setdefault(op, set()).add(parsed["scalar"])

    # registry -> registered, for the ops this binary actually claims
    for op in sorted(registered_points):
        entry = registry.ops.get(op) or {}
        expected = {tuple(values) for _, values in _all_points(registry, op)}
        missing = expected - registered_points[op]
        if missing:
            shown = ", ".join(str(list(p)) for p in sorted(missing)[:6])
            more = "" if len(missing) <= 6 else f" (+{len(missing) - 6} more)"
            problems.append(
                f"{op}: ops.toml's grid has {len(missing)} point(s) no benchmark registers: {shown}{more}"
            )
        missing_scalars = set(entry.get("scalars") or []) - registered_scalars[op]
        if missing_scalars:
            problems.append(
                f"{op}: ops.toml declares scalar(s) {sorted(missing_scalars)} that no benchmark registers"
            )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("--ops-toml", type=Path, default=COMPARISON_DIR / "ops.toml")
    args = parser.parse_args(argv)

    if not args.executable.is_file():
        print(f"no such executable: {args.executable}", file=sys.stderr)
        return EXIT_USAGE

    registry = load_ops_registry(args.ops_toml)
    names = list_registrations(args.executable)
    if not names:
        print(f"{args.executable} registered no benchmarks at all", file=sys.stderr)
        return EXIT_DRIFT

    problems = reconcile(names, registry)
    if problems:
        print(
            f"{args.executable.name}: the C++ registrations and {args.ops_toml} have diverged:",
            file=sys.stderr,
        )
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return EXIT_DRIFT
    print(f"{args.executable.name}: {len(names)} registrations reconcile with {args.ops_toml.name}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
