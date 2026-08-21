# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

"""`result_schema.json` is the gate a published number has to pass.

The tests below are all of the form "delete one fact and the document must stop
validating", because the failure mode being prevented is a result file that
looks fine, merges fine, renders fine, and cannot be defended a year later
because nobody recorded which build of the vendor library produced it.
"""

import copy

import pytest

import harness_support as support

RESULTS = support.FIXTURES / "results"


@pytest.fixture(scope="module")
def valid(validator):
    document = support.read_json(RESULTS / "gemm_eigen_accelerate.json")
    validator.validate(document)
    return document


def invalid_reasons(validator, document):
    return [error.message for error in validator.iter_errors(document)]


def assert_rejected(validator, document, what):
    errors = invalid_reasons(validator, document)
    assert errors, f"schema accepted a document with {what}"
    return errors


# --------------------------------------------------------------------------
# The schema itself, and the committed fixtures
# --------------------------------------------------------------------------


def test_schema_compiles_and_declares_2020_12(schema):
    assert schema["$schema"].endswith("2020-12/schema")
    support.schema_validator(schema)


def test_every_committed_result_fixture_validates(validator):
    for path in sorted(RESULTS.glob("*.json")):
        errors = list(validator.iter_errors(support.read_json(path)))
        assert not errors, f"{path.name}: " + "; ".join(e.message for e in errors[:3])


def test_unknown_op_fixture_is_schema_valid_but_not_registry_valid(validator, ops):
    """The schema cannot police ops.toml; the reducer must.

    If this ever starts failing because the schema learned the op list, the
    reducer's unknown-op path has become untestable through a result file.
    """
    document = support.read_json(RESULTS / "unknown_op.json")
    validator.validate(document)
    assert document["measurements"][0]["op"] not in ops["ops"]


# --------------------------------------------------------------------------
# Required fields
# --------------------------------------------------------------------------


def test_every_required_top_level_field_is_enforced(validator, schema, valid):
    for name in schema["required"]:
        document = copy.deepcopy(valid)
        del document[name]
        assert_rejected(validator, document, f"no top-level {name!r}")


def test_every_required_provenance_field_is_enforced(validator, schema, valid):
    pointers = support.required_pointers(schema, schema["$defs"], schema["properties"]["provenance"], "/provenance")
    assert len(pointers) > 30, f"expected the provenance contract to be broad, enumerated {len(pointers)}"
    for pointer in pointers:
        document = copy.deepcopy(valid)
        if not support.pointer_delete(document, pointer):
            continue
        assert_rejected(validator, document, f"no {pointer}")


def test_every_required_measurement_field_is_enforced(validator, schema, valid):
    pointers = support.required_pointers(schema, schema["$defs"], schema["$defs"]["measurement"], "")
    for pointer in pointers:
        document = copy.deepcopy(valid)
        if not support.pointer_delete(document, "/measurements/0" + pointer):
            continue
        assert_rejected(validator, document, f"a measurement with no {pointer}")


def test_scope_is_required_so_an_absent_row_is_distinguishable(validator, valid):
    document = copy.deepcopy(valid)
    del document["scope"]
    assert_rejected(validator, document, "no scope")
    for key in ("ops", "arms", "scalars", "threads"):
        document = copy.deepcopy(valid)
        del document["scope"][key]
        assert_rejected(validator, document, f"a scope with no {key}")


# --------------------------------------------------------------------------
# The library version -- the field provenance blocks habitually forget
# --------------------------------------------------------------------------


@pytest.mark.parametrize("arm", ["eigen", "accelerate"])
def test_arm_library_version_is_mandatory(validator, valid, arm):
    document = copy.deepcopy(valid)
    del document["provenance"]["arms"][arm]["library_version"]
    assert_rejected(validator, document, f"arm {arm!r} carrying no library_version")


@pytest.mark.parametrize("value", [None, "", 5])
def test_arm_library_version_may_not_be_null_or_empty(validator, valid, value):
    document = copy.deepcopy(valid)
    document["provenance"]["arms"]["accelerate"]["library_version"] = value
    assert_rejected(validator, document, f"library_version = {value!r}")


def test_library_version_unknown_is_a_string_not_an_omission(validator, valid):
    """'unknown' is allowed, but only as a value -- it can then be paired with a
    provenance_gaps entry, which an omission cannot."""
    document = copy.deepcopy(valid)
    document["provenance"]["arms"]["accelerate"]["library_version"] = "unknown"
    validator.validate(document)


def test_arm_kind_and_library_name_are_mandatory(validator, valid):
    for field in ("kind", "library_name"):
        document = copy.deepcopy(valid)
        del document["provenance"]["arms"]["accelerate"][field]
        assert_rejected(validator, document, f"an arm with no {field}")


def test_arms_must_contain_eigen(validator, valid):
    document = copy.deepcopy(valid)
    del document["provenance"]["arms"]["eigen"]
    assert_rejected(validator, document, "no eigen arm")


# --------------------------------------------------------------------------
# The join key
# --------------------------------------------------------------------------


LEGAL_NAMES = [
    "GEMM/eigen/f64/m:1024/n:1024/k:1024",
    "GEMM/openblas/f64/m:1024/n:1024/k:1024",
    "GEMM/eigen/c64/m:4096/n:96/k:96",
    "GEMV/accelerate/f32/m:10000/n:100",
    "POTRF/eigen/f64/n:512",
    "TRSM_LLNN/mkl/f64/n:2048/nrhs:16",
    "GESDD/eigen/f64/m:10000/n:1000",
    "FULLPIVLU/eigen/f64/m:512/n:512",
    "GEMM/eigen/f64/m:2048/n:2048/k:2048/threads:8",
]

ILLEGAL_NAMES = [
    "GEMM/eigen/f64/1024/1024/1024",           # dimensions not self-describing
    "GEMM/eigen/f64",                          # no dimensions at all
    "BM_Gemm<double>/1024",                    # raw BENCHMARK_TEMPLATE output
    "GEMM/eigen/f64/m:1024, n:1024",           # comma and space
    "GEMM/Eigen/f64/m:8/n:8/k:8",              # arm must be lowercase
    "gemm/eigen/f64/m:8/n:8/k:8",              # op must be uppercase
    "GEMM/eigen/float/m:8/n:8/k:8",            # scalar not in the tag set
    "GEMM/eigen/f64/m:-8/n:8/k:8",             # negative dimension
    "GEMM/eigen/f64/m:8.5/n:8/k:8",            # non-integer dimension
    "GEMM/eigen/f64/m:1024/n:1024/k:1024_mean",  # an aggregate 'name', not 'run_name'
]


@pytest.mark.parametrize("name", LEGAL_NAMES)
def test_name_pattern_accepts_the_grammar(validator, valid, name):
    document = copy.deepcopy(valid)
    document["measurements"][0]["name"] = name
    errors = [e for e in validator.iter_errors(document) if "name" in list(e.path)]
    assert not errors, f"schema rejected the legal name {name!r}: {[e.message for e in errors]}"


@pytest.mark.parametrize("name", ILLEGAL_NAMES)
def test_name_pattern_rejects_everything_else(validator, valid, name):
    document = copy.deepcopy(valid)
    document["measurements"][0]["name"] = name
    assert_rejected(validator, document, f"the illegal name {name!r}")


# --------------------------------------------------------------------------
# Values that must not be fudged
# --------------------------------------------------------------------------


def test_timestamp_must_be_utc(validator, valid):
    document = copy.deepcopy(valid)
    document["provenance"]["timestamp_utc"] = "2026-08-01T12:00:00-07:00"
    assert_rejected(validator, document, "a local-offset timestamp")


def test_eigen_commit_must_be_a_full_sha(validator, valid):
    for value in ("e2a2fda", "", "not-a-sha", support.EIGEN_COMMIT.upper()):
        document = copy.deepcopy(valid)
        document["provenance"]["eigen"]["commit"] = value
        assert_rejected(validator, document, f"commit {value!r}")


def test_flops_per_iteration_must_be_strictly_positive(validator, valid):
    for value in (0, -1):
        document = copy.deepcopy(valid)
        document["measurements"][0]["flops_per_iteration"] = value
        assert_rejected(validator, document, f"flops_per_iteration = {value}")


def test_a_stat_must_carry_dispersion_and_a_count(validator, valid):
    for field in ("median", "mad", "min", "max", "count"):
        document = copy.deepcopy(valid)
        del document["measurements"][0]["stats"]["flop_rate"][field]
        assert_rejected(validator, document, f"a stat with no {field}")


def test_repetition_count_may_not_be_zero(validator, valid):
    document = copy.deepcopy(valid)
    document["measurements"][0]["stats"]["flop_rate"]["count"] = 0
    assert_rejected(validator, document, "a stat summarising zero repetitions")


def test_not_measured_reason_is_a_closed_enum(validator, valid, schema):
    allowed = schema["$defs"]["not_measured_entry"]["properties"]["reason"]["enum"]
    assert "no_reference_equivalent" in allowed and "not_implemented" in allowed
    document = copy.deepcopy(valid)
    document["not_measured"] = [{"op": "GEMM", "arm": "accelerate", "reason": "because"}]
    assert_rejected(validator, document, "an ad-hoc not_measured reason")


def test_not_measured_entry_requires_a_reason(validator, valid):
    document = copy.deepcopy(valid)
    document["not_measured"] = [{"op": "GEMM", "arm": "accelerate"}]
    assert_rejected(validator, document, "a not_measured entry with no reason")


def test_provenance_gaps_entry_needs_a_pointer_and_a_reason(validator, valid):
    for entry in ({"field": "cpu/frequency_governor", "reason": "x"}, {"field": "/a"}, {"reason": "x"}):
        document = copy.deepcopy(valid)
        document["provenance_gaps"] = [entry]
        assert_rejected(validator, document, f"provenance_gaps entry {entry!r}")


def test_unknown_keys_are_rejected_on_the_join_surface(validator, valid):
    document = copy.deepcopy(valid)
    document["extra"] = 1
    assert_rejected(validator, document, "an unknown top-level key")
    document = copy.deepcopy(valid)
    document["measurements"][0]["gflops"] = 118.4
    assert_rejected(validator, document, "an unknown measurement key")


def test_new_machine_facts_are_allowed_inside_provenance_subobjects(validator, valid):
    document = copy.deepcopy(valid)
    document["provenance"]["cpu"]["l3_inclusive"] = False
    document["provenance"]["toolchain"]["lto"] = "thin"
    validator.validate(document)


def test_schema_version_and_kind_are_pinned(validator, valid):
    for field, value in (("schema_version", "2.0.0"), ("kind", "something-else")):
        document = copy.deepcopy(valid)
        document[field] = value
        assert_rejected(validator, document, f"{field} = {value!r}")


def test_partial_is_always_true(validator, valid):
    document = copy.deepcopy(valid)
    document["partial"] = False
    assert_rejected(validator, document, "partial = False")


def test_harness_version_is_semantic(validator, valid):
    document = copy.deepcopy(valid)
    document["provenance"]["harness"]["version"] = "1.0"
    assert_rejected(validator, document, "a non-semantic harness version")


def test_cxx_standard_is_one_of_the_supported_values(validator, valid):
    document = copy.deepcopy(valid)
    document["provenance"]["toolchain"]["cxx_standard"] = 11
    assert_rejected(validator, document, "cxx_standard = 11")
    document = copy.deepcopy(valid)
    document["provenance"]["toolchain"]["cxx_standard"] = 17
    validator.validate(document)


def test_cxx_flags_is_a_token_list_not_a_string(validator, valid):
    document = copy.deepcopy(valid)
    document["provenance"]["toolchain"]["cxx_flags"] = "-O3 -DNDEBUG"
    assert_rejected(validator, document, "cxx_flags as one string")


def test_thread_env_values_are_strings(validator, valid):
    document = copy.deepcopy(valid)
    document["provenance"]["threading"]["env"]["OMP_NUM_THREADS"] = 1
    assert_rejected(validator, document, "an integer thread-count env value")
