# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

"""`ops.toml` against itself, and against what the binaries actually register.

The expensive failure this guards is a quietly empty column: an operation added
to the table but never registered in C++ renders as a blank row that reads like
a slow result, and an operation registered but absent from the table is measured
for hours and then dropped on the floor by the reducer.  Both directions are
checked, and both are hard failures.

The registered-name side comes from a committed `--benchmark_list_tests` capture
(`fixtures/raw/benchmark_list_tests.txt`), so the check runs with no compiler,
no BLAS and no build directory.  Refresh it with

    ./build-comparison/bench_gemm_compare --benchmark_list_tests \\
        > benchmarks/comparison/tests/fixtures/raw/benchmark_list_tests.txt
"""

import ast
import operator

import pytest

import harness_support as support

OP_KEY = __import__("re").compile(r"^[A-Z][A-Z0-9_]*$")
ARM_KEY = __import__("re").compile(r"^[a-z][a-z0-9_]*$")
DIM_KEY = __import__("re").compile(r"^[a-z][a-z0-9_]*$")


# --------------------------------------------------------------------------
# A restricted evaluator for ops.<OP>.flops.real
# --------------------------------------------------------------------------

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}


def eval_flops(formula, shape):
    """Evaluate a flops formula over a shape. Rejects anything but arithmetic."""

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id not in shape:
                raise KeyError(node.id)
            return float(shape[node.id])
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        raise ValueError(f"unsupported node {type(node).__name__} in {formula!r}")

    return visit(ast.parse(formula, mode="eval"))


# --------------------------------------------------------------------------
# ops.toml against itself
# --------------------------------------------------------------------------


def test_registry_header_fields(ops):
    assert ops["schema_version"] == "1.0.0"
    assert ops["flop_counter_name"] == "GFLOPS"


def test_every_op_key_matches_the_grammar(ops):
    for key in ops["ops"]:
        assert OP_KEY.match(key), f"op key {key!r} cannot appear in a benchmark name"


def test_every_shape_family_dim_matches_the_grammar(ops):
    for name, family in ops["shape_families"].items():
        assert family["dims"], f"{name} declares no dimensions"
        for dim in family["dims"]:
            assert DIM_KEY.match(dim), f"{name}: dimension {dim!r} cannot appear in a name"
            assert dim not in support.RESERVED_DIMS, f"{name}: {dim!r} collides with a Google Benchmark suffix"


def test_every_family_group_point_has_the_family_arity(ops):
    for name, family in ops["shape_families"].items():
        arity = len(family["dims"])
        known = {group["name"] for group in family["groups"]}
        for group in family["groups"]:
            for point in group["points"]:
                assert len(point) == arity, f"{name}/{group['name']}: {point} is not {arity}-dimensional"
                assert all(isinstance(v, int) and v > 0 for v in point), f"{name}/{group['name']}: {point}"
        for group_name in family["default_groups"]:
            assert group_name in known, f"{name}: default group {group_name!r} does not exist"


def test_every_op_points_at_a_real_family_and_real_scalars(ops):
    for key, op in ops["ops"].items():
        assert op["shape_family"] in ops["shape_families"], f"{key}: unknown shape family"
        assert op["scalars"], f"{key}: no scalars"
        for tag in op["scalars"]:
            assert tag in ops["scalars"], f"{key}: undeclared scalar {tag!r}"
        assert op["status"] in ("implemented", "planned"), f"{key}: status {op['status']!r}"


def test_implemented_ops_name_their_source(ops):
    for key, op in ops["ops"].items():
        if op["status"] != "implemented":
            continue
        source = op.get("source")
        assert source, f"{key} is implemented but names no benchmark source"
        assert (support.REPO_ROOT / source).is_file(), f"{key}: source {source} does not exist"


def test_reference_kind_none_states_why(ops):
    for key, op in ops["ops"].items():
        reference = op["reference"]
        assert reference["kind"] in ("blas", "lapack", "none"), f"{key}: {reference['kind']!r}"
        if reference["kind"] == "none":
            assert reference.get("reason"), (
                f"{key}: reference.kind is 'none' with no reason; the table would render a blank "
                f"that reads as a missing measurement"
            )
        else:
            assert reference.get("routine"), f"{key}: no reference routine named"


def test_flops_formula_uses_only_the_family_dimensions(ops):
    for key, op in ops["ops"].items():
        dims = ops["shape_families"][op["shape_family"]]["dims"]
        shape = {dim: 7 for dim in dims}
        value = eval_flops(op["flops"]["real"], shape)
        assert value > 0, f"{key}: flops formula is not positive at the 7-cube"


def test_flops_formula_is_positive_over_the_whole_grid(ops):
    """A formula that goes negative or zero on a real grid point makes the rate
    meaningless there, and `flops_per_iteration` has exclusiveMinimum 0."""
    for key, op in ops["ops"].items():
        dims = ops["shape_families"][op["shape_family"]]["dims"]
        for point in support.op_grid(ops, key):
            shape = dict(zip(dims, point))
            value = eval_flops(op["flops"]["real"], shape)
            assert value > 0, f"{key}: flops <= 0 at {shape}"


def test_known_flop_counts_are_the_textbook_ones(ops):
    assert eval_flops(ops["ops"]["GEMM"]["flops"]["real"], {"m": 100, "n": 200, "k": 300}) == 2 * 100 * 200 * 300
    assert eval_flops(ops["ops"]["GEMV"]["flops"]["real"], {"m": 100, "n": 200}) == 2 * 100 * 200
    # The Cholesky closed form must reproduce the summation loop the existing
    # benchmarks report, or adopting the shared helper moves a published number.
    for n in (2, 5, 64, 513):
        loop = sum(2 * ((n - 1 - j) * j + (n - 1 - j) + j) for j in range(n))
        assert eval_flops(ops["ops"]["POTRF"]["flops"]["real"], {"n": n}) == pytest.approx(loop)


# The closed forms benchmarks/bench_common.h defines and tests/test_flops.cpp
# pins numerically. `flops.real` and `flops.helper` are two statements of the same
# quantity -- CONTRACTS.md section 2 makes the C++ counter and the reducer's
# flops_per_iteration agree -- and nothing else cross-checks them, so an op that
# names one helper and spells out a different formula publishes two flop figures
# for the same cell.
HELPER_FORMULAS = {
    "gemmFlops": "2*m*n*k",
    "gemvFlops": "2*m*n",
    "trsmFlops": "n*n*nrhs",
    "symmetricFactorizationFlops": "n*(n-1)*(n-2)/3 + 2*n*(n-1)",
    "getrfFlops": "m*n*n - n*n*n/3",
    "geqrfFlops": "2*m*n*n - 2*n*n*n/3",
    "gesddFlops": "8*m*n*n + 4*n*n*n/3",
    "syevFlops": "9*n*n*n",
}


def helper_function_name(helper):
    """`eigen_bench::getrfFlops<Scalar>(m, n)` -> `getrfFlops`."""
    head = helper.split("(", 1)[0]
    return head.split("<", 1)[0].rsplit("::", 1)[-1]


def test_every_flops_helper_is_a_known_closed_form(ops):
    for key, op in ops["ops"].items():
        helper = op["flops"].get("helper")
        assert helper, f"{key}: flops names no helper, so the C++ counter is unconstrained"
        name = helper_function_name(helper)
        assert name in HELPER_FORMULAS, (
            f"{key}: helper {name!r} has no closed form here; add it alongside the one in "
            f"benchmarks/bench_common.h so flops.real can be checked against it"
        )


def test_flops_real_agrees_with_the_named_helper(ops):
    """The defect this catches is silent: the benchmark emits GFLOPS from the
    helper while the reducer records flops_per_iteration from the formula, and a
    published cell then carries two flop counts that differ by a constant."""
    for key, op in ops["ops"].items():
        dims = ops["shape_families"][op["shape_family"]]["dims"]
        formula = HELPER_FORMULAS[helper_function_name(op["flops"]["helper"])]
        # Distinct extents: a square-only check cannot see a swapped argument.
        for point in ([1000, 100, 37], [512, 512, 512], [4096, 128, 16]):
            shape = dict(zip(dims, point))
            assert eval_flops(op["flops"]["real"], shape) == pytest.approx(
                eval_flops(formula, shape)
            ), f"{key}: flops.real and {op['flops']['helper']} disagree at {shape}"


def test_nominal_flop_counts_are_flagged(ops):
    for key in ("GESDD", "SYEV", "FULLPIVLU", "RANDCOD"):
        assert ops["ops"][key]["flops"]["nominal"] is True, f"{key}: a convention must be labelled as one"
    for key in ("GEMM", "GEMV", "POTRF", "GETRF"):
        assert ops["ops"][key]["flops"]["nominal"] is False


def test_variant_ops_point_at_a_base_mnemonic(ops):
    for key, op in ops["ops"].items():
        if "variant" in op:
            assert op.get("base_mnemonic"), f"{key}: a variant with no base_mnemonic cannot be grouped"


def test_scalar_flop_scale_matches_complexity(ops):
    for tag, scalar in ops["scalars"].items():
        assert scalar["tag"] == tag
        assert scalar["flop_scale"] == (8.0 if scalar["is_complex"] else 2.0), tag


# --------------------------------------------------------------------------
# Reconciliation with what the binaries register
# --------------------------------------------------------------------------


def parse_name(name):
    """The grammar of CONTRACTS.md section 1.3, independent of any implementation."""
    fields = name.split("/")
    if len(fields) < 4:
        raise ValueError(f"too few fields in {name!r}")
    op, arm, scalar = fields[0], fields[1], fields[2]
    threads, shape, order = 1, {}, []
    for field in fields[3:]:
        key, sep, value = field.partition(":")
        if not sep or not value.isdigit():
            raise ValueError(f"unparseable field {field!r} in {name!r}")
        if key == "threads":
            threads = int(value)
        elif key in support.RESERVED_DIMS:
            raise ValueError(f"unsupported registration form {key!r} in {name!r}")
        else:
            order.append(key)
            shape[key] = int(value)
    return {"op": op, "arm": arm, "scalar": scalar, "shape": shape, "shape_dims": order, "threads": threads}


def reconcile(names, ops):
    """Problems in BOTH directions between a registration listing and ops.toml."""
    problems = []
    registered_ops = set()
    registered_points_by_op = set()
    for name in names:
        try:
            parsed = parse_name(name)
        except ValueError as exc:
            problems.append(f"unparseable: {exc}")
            continue
        op = parsed["op"]
        entry = ops["ops"].get(op)
        if entry is None:
            problems.append(f"{name}: op {op!r} is registered but absent from ops.toml")
            continue
        registered_ops.add(op)
        if entry["status"] != "implemented":
            problems.append(f"{name}: op {op!r} is registered but ops.toml calls it {entry['status']!r}")
        if not ARM_KEY.match(parsed["arm"]):
            problems.append(f"{name}: arm {parsed['arm']!r} is not a legal arm key")
        if parsed["scalar"] not in entry["scalars"]:
            problems.append(f"{name}: scalar {parsed['scalar']!r} is not declared for {op}")
        dims = ops["shape_families"][entry["shape_family"]]["dims"]
        if parsed["shape_dims"] != dims:
            problems.append(f"{name}: dimensions {parsed['shape_dims']} != registry order {dims}")
            continue
        point = [parsed["shape"][d] for d in dims]
        if point not in support.op_grid(ops, op):
            problems.append(f"{name}: shape {point} is not a point of family {entry['shape_family']!r}")
        if parsed["arm"] == "eigen":
            registered_points_by_op.add((op, tuple(point)))
    for op, entry in ops["ops"].items():
        if entry["status"] == "implemented" and op not in registered_ops:
            problems.append(f"{op}: ops.toml calls it implemented but no benchmark registers it")
    # Point level, the other direction.  The loop above only asks whether an op is
    # registered at all, so adding a point to a shape family and forgetting the
    # matching entry in the C++ SIZES macro used to pass silently: run.py would
    # plan a cell the binary never emits and file it as `not_implemented`, which
    # reads on the published page as "Eigen does not implement this" rather than
    # "the grid drifted".  The C++ grid is a hand transcription of ops.toml, so
    # this is the only thing standing between the two.
    for op, entry in ops["ops"].items():
        if entry["status"] != "implemented":
            continue
        registered_points = {tuple(p) for o, p in registered_points_by_op if o == op}
        for point in support.op_grid(ops, op):
            if tuple(point) not in registered_points:
                problems.append(f"{op}: ops.toml grid has point {point} but no benchmark registers it")
    return problems


def test_canned_listing_reconciles_with_the_registry(listed_benchmarks, ops):
    assert listed_benchmarks, "the --benchmark_list_tests capture is empty"
    problems = reconcile(listed_benchmarks, ops)
    assert not problems, "registry/registration mismatch:\n  " + "\n  ".join(problems)


def test_listing_covers_the_eigen_arm_for_every_implemented_op(listed_benchmarks, ops):
    by_op = {}
    for name in listed_benchmarks:
        parsed = parse_name(name)
        by_op.setdefault(parsed["op"], set()).add(parsed["arm"])
    for op in support.implemented_ops(ops):
        assert "eigen" in by_op.get(op, set()), f"{op}: no eigen arm is registered, so there is nothing to compare"


def test_an_op_implemented_in_the_table_but_never_registered_fails(listed_benchmarks, ops):
    """Direction 1: the quietly empty column."""
    doctored = support.deep_copy(ops)
    doctored["ops"]["GEMV"]["status"] = "implemented"
    problems = reconcile(listed_benchmarks, doctored)
    assert any("GEMV" in p and "no benchmark registers it" in p for p in problems), problems


def test_an_op_registered_but_absent_from_the_table_fails(listed_benchmarks, ops):
    """Direction 2: hours of measurement the reducer would drop."""
    problems = reconcile(listed_benchmarks + ["NOSUCHOP/eigen/f64/m:8/n:8/k:8"], ops)
    assert any("NOSUCHOP" in p and "absent from ops.toml" in p for p in problems), problems


def test_a_registration_of_a_planned_op_fails(listed_benchmarks, ops):
    problems = reconcile(listed_benchmarks + ["POTRF/eigen/f64/n:512"], ops)
    assert any("POTRF" in p and "planned" in p for p in problems), problems


def test_a_shape_outside_the_family_grid_fails(listed_benchmarks, ops):
    problems = reconcile(listed_benchmarks + ["GEMM/eigen/f64/m:65/n:65/k:65"], ops)
    assert any("not a point of family" in p for p in problems), problems


def test_a_transposed_dimension_order_fails(listed_benchmarks, ops):
    problems = reconcile(listed_benchmarks + ["GEMM/eigen/f64/n:64/m:64/k:64"], ops)
    assert any("registry order" in p for p in problems), problems


def test_an_undeclared_scalar_fails(listed_benchmarks, ops):
    problems = reconcile(listed_benchmarks + ["GEMM/eigen/f16/m:64/n:64/k:64"], ops)
    assert any("is not declared" in p for p in problems), problems


def test_a_registration_without_argnames_fails(listed_benchmarks, ops):
    problems = reconcile(listed_benchmarks + ["GEMM/eigen/f64/64/64/64"], ops)
    assert any("unparseable" in p for p in problems), problems


def test_the_stub_reproduces_a_listing_that_reconciles(tmp_path, ops, raw_gbench):
    """`--benchmark_list_tests` really is the join key's other end."""
    installed, _ = support.stub_executable(tmp_path / "bin", ["bench_gemm_compare"], raw_gbench)
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, str(installed[0]), "--benchmark_list_tests"],
        capture_output=True,
        text=True,
        check=True,
    )
    names = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    assert names, "the stub listed nothing"
    for name in names:
        parsed = parse_name(name)
        assert parsed["op"] in ops["ops"]
        assert not any(name.endswith(suffix) for suffix in ("_mean", "_median", "_stddev", "_cv")), (
            f"{name}: a listing must carry run_name, never an aggregate 'name'"
        )
