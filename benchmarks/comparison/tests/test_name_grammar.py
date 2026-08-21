# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

"""The benchmark-name parser in `reduce.py`.

The name is the join key between the C++ registrations and every Python
consumer, so a parser bug does not produce a visible error: it produces a table
whose rows are attributed to the wrong shape.  These tests pin the exact
behaviour CONTRACTS.md section 1.3 specifies, including the two traps -- the
`_mean` suffix on aggregate rows, and the reserved Google Benchmark suffixes.
"""

import pytest

import harness_support as support


@pytest.fixture(scope="module")
def parse():
    reduce_module = support.import_impl("reduce.py")
    return support.resolve_callable(reduce_module, "parse_benchmark_name", "parse_name", "parse_run_name")


CASES = [
    ("GEMM/eigen/f64/m:1024/n:1024/k:1024", "GEMM", "eigen", "f64", {"m": 1024, "n": 1024, "k": 1024}, ["m", "n", "k"], 1),
    ("GEMM/openblas/f64/m:1024/n:1024/k:1024", "GEMM", "openblas", "f64", {"m": 1024, "n": 1024, "k": 1024}, ["m", "n", "k"], 1),
    ("GEMM/eigen/c64/m:4096/n:96/k:96", "GEMM", "eigen", "c64", {"m": 4096, "n": 96, "k": 96}, ["m", "n", "k"], 1),
    ("GEMV/accelerate/f32/m:10000/n:100", "GEMV", "accelerate", "f32", {"m": 10000, "n": 100}, ["m", "n"], 1),
    ("POTRF/eigen/f64/n:512", "POTRF", "eigen", "f64", {"n": 512}, ["n"], 1),
    ("TRSM_LLNN/mkl/f64/n:2048/nrhs:16", "TRSM_LLNN", "mkl", "f64", {"n": 2048, "nrhs": 16}, ["n", "nrhs"], 1),
    ("GESDD/eigen/f64/m:10000/n:1000", "GESDD", "eigen", "f64", {"m": 10000, "n": 1000}, ["m", "n"], 1),
    ("FULLPIVLU/eigen/f64/m:512/n:512", "FULLPIVLU", "eigen", "f64", {"m": 512, "n": 512}, ["m", "n"], 1),
    ("GEMM/eigen/f64/m:2048/n:2048/k:2048/threads:8", "GEMM", "eigen", "f64", {"m": 2048, "n": 2048, "k": 2048}, ["m", "n", "k"], 8),
]


@pytest.mark.impl
@pytest.mark.parametrize("name,op,arm,scalar,shape,dims,threads", CASES)
def test_every_documented_name_parses(parse, name, op, arm, scalar, shape, dims, threads):
    parsed = parse(name)
    assert parsed["op"] == op
    assert parsed["arm"] == arm
    assert parsed["scalar"] == scalar
    assert parsed["shape"] == shape
    assert parsed["shape_dims"] == dims
    assert parsed["threads"] == threads


@pytest.mark.impl
def test_absent_threads_field_means_one(parse):
    assert parse("GEMM/eigen/f64/m:8/n:8/k:8")["threads"] == 1


@pytest.mark.impl
def test_threads_is_never_a_shape_dimension(parse):
    parsed = parse("GEMM/eigen/f64/m:8/n:8/k:8/threads:4")
    assert "threads" not in parsed["shape"]
    assert "threads" not in parsed["shape_dims"]


@pytest.mark.impl
def test_dimension_order_is_preserved_not_sorted(parse):
    """`shape_dims` is what catches a transposed table; sorting it destroys the check."""
    assert parse("SOLVE/eigen/f64/n:8/nrhs:2")["shape_dims"] == ["n", "nrhs"]
    assert parse("SOLVE/eigen/f64/nrhs:2/n:8")["shape_dims"] == ["nrhs", "n"]


@pytest.mark.impl
@pytest.mark.parametrize("field", sorted(support.RESERVED_DIMS))
def test_reserved_google_benchmark_suffixes_are_hard_errors(parse, field):
    name = f"GEMM/eigen/f64/m:8/n:8/k:8/{field}:3"
    with pytest.raises(Exception) as excinfo:
        parse(name)
    assert name in str(excinfo.value) or field in str(excinfo.value)


@pytest.mark.impl
@pytest.mark.parametrize(
    "name",
    [
        "GEMM/eigen/f64/1024/1024/1024",
        "GEMM/eigen/f64/m1024/n:8/k:8",
        "GEMM/eigen/f64/m:/n:8/k:8",
        "GEMM/eigen/f64/m:eight/n:8/k:8",
        "GEMM/eigen/f64/m:-8/n:8/k:8",
        "GEMM/eigen/f64/m:8.5/n:8/k:8",
    ],
)
def test_a_dimension_without_a_name_is_an_error_that_quotes_the_string(parse, name):
    with pytest.raises(Exception) as excinfo:
        parse(name)
    assert name in str(excinfo.value), (
        "the diagnostic must quote the offending name; a bare 'invalid field' "
        "sends the reader back to the binary to find which registration it was"
    )


@pytest.mark.impl
def test_an_aggregate_suffix_is_not_stripped(parse):
    """`_` is legal inside an op key (TRSM_LLNN) and an arm key, so stripping
    suffixes from `name` corrupts real names. Consumers must join on `run_name`."""
    parsed = parse("TRSM_LLNN/mkl/f64/n:2048/nrhs:16")
    assert parsed["op"] == "TRSM_LLNN"
    with pytest.raises(Exception):
        parse("GEMM/eigen/f64/m:8/n:8/k:8_mean")


@pytest.mark.impl
def test_underscored_arm_keys_survive(parse):
    parsed = parse("GEMM/my_vendor_blas/f64/m:8/n:8/k:8")
    assert parsed["arm"] == "my_vendor_blas"


@pytest.mark.impl
def test_names_that_differ_only_in_arity_do_not_collide(parse):
    """Each dimension carries its name, so GEMV m:512/n:512 and POTRF n:512
    cannot be confused even though both are 512-sized."""
    keys = set()
    for name in ("GEMV/eigen/f64/m:512/n:512", "POTRF/eigen/f64/n:512", "GEMM/eigen/f64/m:512/n:512/k:512"):
        parsed = parse(name)
        keys.add((parsed["op"], parsed["arm"], parsed["scalar"], tuple(sorted(parsed["shape"].items())), parsed["threads"]))
    assert len(keys) == 3
