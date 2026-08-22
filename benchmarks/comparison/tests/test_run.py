# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

"""`run.py`, driven against a stub benchmark executable.

A measurement campaign is expensive; a scripting bug that is only visible after
it costs the whole campaign.  The cases below are the ones that cost a rerun:

* a vendor thread-count variable that was never set, so a "single-threaded"
  comparison quietly used every core;
* CPU pinning that the platform could not honour, recorded as though it had been;
* a benchmark row that failed, turned into a measurement with a zero in it
  instead of an explicit negative;
* the vendor's own aggregate rows counted as extra repetitions;
* the flop-rate counter divided by 1e9 twice, or not at all.

Everything runs against `fixtures/stub_benchmark.py` and a machine profile in
`fixtures/machines/`: no compiler, no BLAS, no quiet machine required.
"""

import json
import os
import platform
import re
import shutil
import subprocess
import sys

import pytest

import harness_support as support

pytestmark = pytest.mark.impl

MACHINES = support.FIXTURES / "machines"


@pytest.fixture(scope="module")
def run_module():
    return support.import_impl("run.py")


@pytest.fixture(scope="module")
def registry(run_module):
    load = support.resolve_callable(run_module, "load_ops_registry", "load_ops_toml")
    return load(support.OPS_TOML)


@pytest.fixture
def machine(run_module):
    load = support.resolve_callable(run_module, "load_machine_profile", "load_machine")
    return load(MACHINES / "testmachine.toml")


@pytest.fixture
def pinned_machine(run_module):
    load = support.resolve_callable(run_module, "load_machine_profile", "load_machine")
    return load(MACHINES / "testmachine-pinned.toml")


# --------------------------------------------------------------------------
# Thread environment -- the regression that costs a whole campaign
# --------------------------------------------------------------------------


def test_every_documented_vendor_thread_variable_is_set(run_module):
    build = support.resolve_callable(run_module, "build_thread_env")
    env = build(1)
    missing = [name for name in support.THREAD_ENV_VARS if name not in env]
    assert not missing, (
        f"run.py leaves {missing} unset. A vendor whose thread count is left to its own default turns a "
        f"single-threaded comparison into a multi-threaded one, and nothing in the output says so."
    )
    assert all(env[name] == "1" for name in support.THREAD_ENV_VARS), env


@pytest.mark.parametrize("threads", [1, 2, 8, 64])
def test_thread_variables_carry_the_requested_count(run_module, threads):
    env = support.resolve_callable(run_module, "build_thread_env")(threads)
    for name in support.THREAD_ENV_VARS:
        assert env[name] == str(threads), f"{name} = {env[name]!r} for a {threads}-thread run"


def test_thread_env_values_are_all_strings(run_module):
    env = support.resolve_callable(run_module, "build_thread_env")(4)
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()), env


def test_thread_env_rejects_a_nonsensical_thread_count(run_module):
    build = support.resolve_callable(run_module, "build_thread_env")
    for bad in (0, -1):
        with pytest.raises(Exception):
            build(bad)


def test_a_machine_profile_can_pin_one_library_sequential(run_module, machine):
    """`VECLIB_MAXIMUM_THREADS = "1"` in the profile must survive an 8-thread run."""
    build = support.resolve_callable(run_module, "build_thread_env")
    arm = machine.arms["accelerate"]
    env = build(8, arm=arm)
    assert env["VECLIB_MAXIMUM_THREADS"] == "1"
    assert env["OMP_NUM_THREADS"] == "8"


def test_a_machine_profile_thread_env_can_track_the_thread_count(run_module, pinned_machine):
    build = support.resolve_callable(run_module, "build_thread_env")
    env = build(6, arm=pinned_machine.arms["openblas"])
    assert env["OPENBLAS_NUM_THREADS"] == "6"


# --------------------------------------------------------------------------
# Pinning: honestly recorded, never silently skipped
# --------------------------------------------------------------------------


def test_pinning_request_on_macos_is_reported_unavailable(run_module, pinned_machine):
    plan = support.resolve_callable(run_module, "plan_pinning")(
        pinned_machine, system="Darwin", tool_exists=lambda name: True
    )
    assert plan.applied is False
    assert plan.command_prefix == ()
    assert plan.unavailable_reason, "macOS cannot pin, and the run must say so rather than stay silent"
    assert "macos" in plan.unavailable_reason.lower()


def test_pinning_request_on_linux_with_the_tool_present_is_applied(run_module, pinned_machine):
    plan = support.resolve_callable(run_module, "plan_pinning")(
        pinned_machine, system="Linux", tool_exists=lambda name: True
    )
    assert plan.applied is True
    assert plan.command_prefix[0] == "taskset"
    assert plan.unavailable_reason is None


def test_a_missing_pinning_tool_is_reported_not_assumed(run_module, pinned_machine):
    plan = support.resolve_callable(run_module, "plan_pinning")(
        pinned_machine, system="Linux", tool_exists=lambda name: False
    )
    assert plan.applied is False
    assert plan.unavailable_reason and "taskset" in plan.unavailable_reason


def test_a_machine_with_no_pinning_request_is_not_flagged(run_module, machine):
    plan = support.resolve_callable(run_module, "plan_pinning")(
        machine, system="Darwin", tool_exists=lambda name: False
    )
    assert plan.applied is False
    assert plan.unavailable_reason is None, "a single-node machine that asked for nothing has nothing to disclose"


@pytest.mark.skipif(platform.system() != "Darwin", reason="the macOS pinning path is what this asserts")
def test_macos_run_records_the_pinning_gap_end_to_end(stub_run):
    document = stub_run(machine="testmachine")
    pointers = {gap["field"] for gap in document["provenance_gaps"]}
    assert "/provenance/cpu/frequency_governor" in pointers, (
        "macOS exposes no governor; CONTRACTS.md section 7 requires the gap to be stated"
    )
    assert document["provenance"]["cpu"]["frequency_governor"] is None
    assert document["provenance"]["numa"]["cpu_binding"] is None
    assert "/provenance/numa/cpu_binding" in pointers, (
        "the run was not bound to any CPU set; that has to be disclosed, not omitted"
    )
    for gap in document["provenance_gaps"]:
        assert gap["reason"].strip(), f"empty reason for {gap['field']}"


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def test_median_and_mad_of_a_known_sample(run_module):
    summarize = support.resolve_callable(run_module, "summarize")
    stat = summarize([10.0, 12.0, 14.0, 16.0, 100.0])
    assert stat["median"] == 14.0
    assert stat["mad"] == 2.0, "MAD must resist the outlier that a stddev would absorb"
    assert stat["count"] == 5
    assert stat["min"] == 10.0 and stat["max"] == 100.0


def test_a_single_repetition_has_zero_dispersion_not_an_error(run_module):
    stat = support.resolve_callable(run_module, "summarize")([7.5])
    assert stat["count"] == 1
    assert stat["mad"] == 0.0
    assert stat["median"] == 7.5
    assert stat["stddev"] in (0.0, None)


def test_summarize_matches_the_schema(run_module, validator, schema):
    stat = support.resolve_callable(run_module, "summarize")([1.0, 2.0, 3.0])
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(stat, {**schema["$defs"]["stat"], "$defs": schema["$defs"]})


# --------------------------------------------------------------------------
# Distilling Google Benchmark JSON
# --------------------------------------------------------------------------


@pytest.fixture
def distilled(run_module, registry, raw_gbench):
    distill = support.resolve_callable(run_module, "distill_benchmark_json")
    return distill(raw_gbench, registry)


def test_aggregate_rows_are_discarded(distilled, raw_gbench):
    """Counting the vendor's own _mean/_median/_stddev/_cv rows as repetitions
    inflates the sample and changes the dispersion the harness reports."""
    aggregates = [r for r in raw_gbench["benchmarks"] if r.get("run_type") == "aggregate"]
    assert aggregates, "the fixture must contain aggregate rows for this to mean anything"
    for measurement in distilled.measurements:
        assert measurement["repetitions"] == 3, measurement["name"]
        assert measurement["stats"]["flop_rate"]["count"] == 3, measurement["name"]


def test_measurements_join_on_run_name_never_on_name(distilled):
    for measurement in distilled.measurements:
        assert not measurement["name"].endswith(("_mean", "_median", "_stddev", "_cv")), measurement["name"]


def test_the_flop_counter_is_stored_as_flops_per_second(distilled):
    """CONTRACTS.md section 1.6: the counter is named GFLOPS and holds flops per
    second. The single division by 1e9 happens in reduce.py, not here."""
    for measurement in distilled.measurements:
        rate = measurement["stats"]["flop_rate"]["median"]
        assert rate > 1e6, f"{measurement['name']}: {rate} looks like it was already divided by 1e9"


def test_distilled_rates_match_the_canned_medians(distilled):
    expected = {
        "GEMM/eigen/f64/m:64/n:64/k:64": 45.0e9,
        "GEMM/eigen/f64/m:128/n:128/k:128": 96.0e9,
        "GEMM/accelerate/f64/m:64/n:64/k:64": 60.0e9,
        "GEMM/accelerate/f64/m:128/n:128/k:128": 95.0e9,
    }
    got = {m["name"]: m["stats"]["flop_rate"]["median"] for m in distilled.measurements}
    assert set(got) == set(expected), got.keys()
    for name, value in expected.items():
        assert got[name] == pytest.approx(value, rel=1e-9), name


def test_times_are_converted_to_seconds(distilled):
    for measurement in distilled.measurements:
        seconds = measurement["stats"]["real_time_s"]["median"]
        assert 0 < seconds < 1.0, f"{measurement['name']}: {seconds}s is not a plausible per-iteration time"
        expected = measurement["flops_per_iteration"] / measurement["stats"]["flop_rate"]["median"]
        assert seconds == pytest.approx(expected, rel=1e-6)


def test_a_failed_row_becomes_an_error_not_a_measurement(distilled, raw_gbench):
    failed = [r for r in raw_gbench["benchmarks"] if r.get("error_occurred")]
    assert failed, "the fixture must contain a failed row"
    names = {m["name"] for m in distilled.measurements}
    for row in failed:
        assert row["run_name"] not in names, "a row that errored must never appear as a measurement"
    assert distilled.errors, "the failure must be reported, not dropped"
    assert any("info" in e["message"] for e in distilled.errors)


def test_flops_per_iteration_comes_from_the_registry(distilled):
    for measurement in distilled.measurements:
        m, n, k = (measurement["shape"][d] for d in ("m", "n", "k"))
        assert measurement["flops_per_iteration"] == pytest.approx(2.0 * m * n * k)


def test_distilled_measurements_validate_against_the_schema(distilled, schema):
    jsonschema = pytest.importorskip("jsonschema")
    sub = {**schema["$defs"]["measurement"], "$defs": schema["$defs"]}
    for measurement in distilled.measurements:
        jsonschema.validate(measurement, sub)


def test_a_multithreaded_plan_still_matches_the_rows_it_planned(run_module, registry, raw_gbench):
    """`/threads:N` is Google Benchmark's OWN multi-threaded registration form,
    which these benchmarks do not use: the harness controls threading through the
    library environment, so every emitted name parses back as threads=1. Keying
    the plan on the parsed count made `--threads 8` measure the whole grid for
    hours and then discard every row as `not_implemented`."""
    distill = support.resolve_callable(run_module, "distill_benchmark_json")
    plan_cells = support.resolve_callable(run_module, "plan_cells")
    planned = plan_cells(
        registry,
        ops=["GEMM"],
        arms=["eigen", "accelerate"],
        scalars=["f64"],
        groups=["small"],
        threads=8,
    )
    distilled = distill(raw_gbench, registry, planned=planned, threads=8)
    assert distilled.measurements, "a multithreaded plan discarded every measured row"
    assert {m["threads"] for m in distilled.measurements} == {8}
    single = distill(raw_gbench, registry, planned=plan_cells(
        registry, ops=["GEMM"], arms=["eigen", "accelerate"], scalars=["f64"], groups=["small"], threads=1
    ), threads=1)
    assert len(distilled.measurements) == len(single.measurements)


def test_distilled_rows_are_keyed_on_the_configured_thread_count(run_module, registry, raw_gbench):
    distill = support.resolve_callable(run_module, "distill_benchmark_json")
    distilled = distill(raw_gbench, registry, threads=4)
    assert distilled.measurements
    for measurement in distilled.measurements:
        assert measurement["threads"] == 4
    for key in distilled.seen_keys:
        assert key[-1] == 4


def test_a_name_that_contradicts_the_configured_thread_count_is_an_error(run_module, registry, raw_gbench):
    distill = support.resolve_callable(run_module, "distill_benchmark_json")
    document = support.deep_copy(raw_gbench)
    for row in document["benchmarks"]:
        for field in ("name", "run_name"):
            if field in row:
                row[field] = f"{row[field]}/threads:2"
    with pytest.raises(Exception) as excinfo:
        distill(document, registry, threads=8)
    assert "threads" in str(excinfo.value)


# --------------------------------------------------------------------------
# Planning, filtering, and the not-measured diff
# --------------------------------------------------------------------------


def test_planned_cells_are_the_declared_cross_product(run_module, registry, ops):
    cells = support.resolve_callable(run_module, "plan_cells")(
        registry, ops=["GEMM"], arms=["eigen", "accelerate"], scalars=["f64"], groups=["small"]
    )
    points = support.family_points(ops, "square3", ["small"])
    assert len(cells) == 2 * len(points)
    assert {cell.arm for cell in cells} == {"eigen", "accelerate"}


def test_a_planned_cell_reproduces_the_benchmark_name_grammar(run_module, registry):
    cells = support.resolve_callable(run_module, "plan_cells")(
        registry, ops=["GEMM"], arms=["eigen"], scalars=["f64"], groups=["small"]
    )
    pattern = re.compile(r"^[A-Z][A-Z0-9_]*/[a-z][a-z0-9_]*/(f16|bf16|f32|f64|c32|c64)(/[a-z][a-z0-9_]*:[0-9]+)+$")
    for cell in cells:
        assert pattern.match(cell.name), cell.name


def test_an_undeclared_scalar_plans_nothing_rather_than_guessing(run_module, registry):
    cells = support.resolve_callable(run_module, "plan_cells")(
        registry, ops=["GEMM"], arms=["eigen"], scalars=["f16"], groups=["small"]
    )
    assert cells == []


def test_the_generated_filter_uses_only_legal_name_characters(run_module, registry):
    cells = support.resolve_callable(run_module, "plan_cells")(
        registry, ops=["GEMM"], arms=["eigen", "accelerate"], scalars=["f64"], groups=["small"]
    )
    expression = support.resolve_callable(run_module, "make_benchmark_filter")(cells)
    compiled = re.compile(expression)
    for cell in cells:
        assert compiled.match(cell.name), f"{expression} does not select {cell.name}"
    for other in ("GEMM/eigen/f32/m:64/n:64/k:64", "GEMV/eigen/f64/m:64/n:64", "POTRF/eigen/f64/n:64"):
        assert not compiled.match(other), f"{expression} wrongly selects {other}"


def test_an_empty_plan_produces_a_filter_that_matches_nothing(run_module):
    expression = support.resolve_callable(run_module, "make_benchmark_filter")([])
    compiled = re.compile(expression)
    assert not compiled.match("GEMM/eigen/f64/m:8/n:8/k:8")


def test_unmeasured_planned_cells_become_explicit_negatives(run_module, registry):
    plan_cells = support.resolve_callable(run_module, "plan_cells")
    diff = support.resolve_callable(run_module, "diff_not_measured")
    cells = plan_cells(registry, ops=["GEMM"], arms=["eigen"], scalars=["f64"], groups=["small"])
    entries = diff(cells, [cells[0].key], registry)
    assert len(entries) == len(cells) - 1
    assert all(entry["reason"] for entry in entries)
    assert {entry["op"] for entry in entries} == {"GEMM"}


def test_an_eigen_only_op_says_the_reference_cannot_exist(run_module, registry):
    """`no_reference_equivalent` is a permanent fact about the APIs;
    `not_implemented` is a temporary fact about us. Conflating them makes a
    table read as though somebody forgot to run something."""
    plan_cells = support.resolve_callable(run_module, "plan_cells")
    diff = support.resolve_callable(run_module, "diff_not_measured")
    cells = plan_cells(registry, ops=["FULLPIVLU"], arms=["accelerate"], scalars=["f64"], groups=["small"])
    entries = diff(cells, [], registry)
    assert entries and all(e["reason"] == "no_reference_equivalent" for e in entries)
    assert all(e["detail"] for e in entries), "the reason text from ops.toml must be carried through"


# --------------------------------------------------------------------------
# Identifiers
# --------------------------------------------------------------------------


def test_config_id_splits_on_everything_that_invalidates_a_comparison(run_module):
    make = support.resolve_callable(run_module, "make_config_id", "config_id_for")
    base = dict(
        machine_id="m4pro",
        isa_target="aarch64-neon",
        compiler_id="AppleClang",
        compiler_version="17.0.0.17000013",
        commit_short="e2a2fda17",
        threads=1,
    )
    reference = make(**base)
    assert reference == "m4pro__aarch64-neon__appleclang17__e2a2fda17__t1"
    for field, value in (
        ("machine_id", "m4max"),
        ("isa_target", "aarch64-sve"),
        ("commit_short", "deadbeef1"),
        ("threads", 8),
    ):
        assert make(**{**base, field: value}) != reference, f"{field} must split the config"
    # A compiler MINOR bump deliberately does not split (CONTRACTS.md 5.2).
    assert make(**{**base, "compiler_version": "17.2.5"}) == reference


def test_run_id_matches_the_schema_pattern(run_module, schema):
    from datetime import datetime, timezone

    moment = datetime(2026, 8, 20, 16, 1, 11, tzinfo=timezone.utc)
    stamp = support.resolve_callable(run_module, "compact_timestamp")(moment)
    assert stamp == "20260820T160111Z"
    make = support.resolve_callable(run_module, "make_run_id")
    run_id = make("m4pro", "aarch64-neon", "accelerate", "e2a2fda17", stamp)
    assert re.match(schema["properties"]["run_id"]["pattern"], run_id), run_id
    assert stamp in run_id
    assert make("m4pro", "aarch64-neon", "accelerate", "e2a2fda17", stamp) == run_id


def test_rfc3339_timestamp_is_utc_with_a_z(run_module, schema):
    from datetime import datetime, timedelta, timezone

    render = support.resolve_callable(run_module, "rfc3339_timestamp")
    local = datetime(2026, 8, 20, 9, 1, 11, tzinfo=timezone(timedelta(hours=-7)))
    assert render(local) == "2026-08-20T16:01:11Z", (
        "Google Benchmark's context 'date' is local time; a result file that copies it "
        "silently mislabels every run made outside UTC"
    )


def test_a_schema_failure_lands_in_a_dot_invalid_file(run_module, tmp_path):
    make = support.resolve_callable(run_module, "make_invalid_output_path")
    path = make(tmp_path / "result.json")
    assert str(path).endswith(".invalid.json")


# --------------------------------------------------------------------------
# End to end against the stub executable
# --------------------------------------------------------------------------


@pytest.fixture
def stub_environment(tmp_path, raw_gbench, ops):
    """A build tree containing the stub under every name run.py may look for."""
    document = support.deep_copy(raw_gbench)
    # Keep only rows inside the `small` group the tests select, and move the
    # failed row onto a size that IS in that group so it is a planned cell.
    rows = []
    for row in document["benchmarks"]:
        name = row.get("run_name", "")
        if row.get("error_occurred"):
            row = dict(row, name=name.replace("m:256/n:256/k:256", "m:96/n:96/k:96"),
                       run_name=name.replace("m:256/n:256/k:256", "m:96/n:96/k:96"))
            rows.append(row)
        elif "m:64/" in name or "m:128/" in name:
            rows.append(row)
    document["benchmarks"] = rows

    build_dir = tmp_path / "build-comparison"
    targets = ["bench_gemm_compare", "bench_comparison"]
    payloads = []
    for isa in ("aarch64-neon", "x86-64-avx2"):
        for arm in ("eigen", "accelerate", "openblas"):
            _, payload = support.stub_executable(build_dir / f"{isa}__{arm}", targets, document)
            payloads.append(payload)
    trace = tmp_path / "stub-trace.jsonl"
    return {
        "build_dir": build_dir,
        "trace": trace,
        "payload": payloads[0],
        "env": {"STUB_BENCH_TRACE": str(trace), "STUB_BENCH_JSON": str(payloads[0])},
    }


@pytest.fixture
def stub_run(stub_environment, validator, tmp_path):
    def invoke(machine="testmachine", extra=(), expect=0, env=None):
        args = [
            "--machine", machine,
            "--machines-dir", str(MACHINES),
            "--build-dir", str(stub_environment["build_dir"]),
            "--no-configure", "--no-build",
            "--ops", "GEMM",
            "--scalars", "f64",
            "--groups", "small",
            "--repetitions", "3",
            "--allow-dirty", "--allow-noisy",
            "--out", "-",
            *extra,
        ]
        merged_env = dict(stub_environment["env"])
        merged_env.update(env or {})
        proc = support.run_cli("run.py", args, env=merged_env)
        assert proc.returncode == expect, (
            f"run.py exited {proc.returncode}, expected {expect}\n"
            f"--- stderr ---\n{proc.stderr[-4000:]}"
        )
        if expect != 0:
            return proc
        document = json.loads(proc.stdout)
        errors = list(validator.iter_errors(document))
        assert not errors, "run.py emitted a document its own schema rejects: " + "; ".join(
            e.message for e in errors[:3]
        )
        return document

    return invoke


def test_a_stub_run_produces_a_schema_valid_result(stub_run):
    document = stub_run()
    assert document["kind"] == "eigen-benchmark-comparison-result"
    assert document["partial"] is True
    assert document["measurements"], "the stub emitted rows; none of them reached the result file"


def test_the_child_process_really_received_every_thread_variable(stub_run, stub_environment):
    """The strongest form of the thread-env regression test: it inspects the
    environment the benchmark process actually saw, not the dictionary the
    harness built."""
    stub_run()
    trace_path = stub_environment["trace"]
    assert trace_path.is_file(), "the stub was never invoked; the harness did not find an executable"
    records = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
    assert records
    for record in records:
        missing = [name for name in support.THREAD_ENV_VARS if name not in record["env"]]
        assert not missing, f"the benchmark process ran without {missing}"
        assert all(record["env"][name] == "1" for name in support.THREAD_ENV_VARS)


def test_the_recorded_env_is_exactly_what_the_child_received(stub_run, stub_environment):
    document = stub_run()
    recorded = document["provenance"]["threading"]["env"]
    records = [json.loads(line) for line in stub_environment["trace"].read_text().splitlines() if line.strip()]
    child = records[-1]["env"]
    for name, value in recorded.items():
        assert child.get(name) == value, (
            f"provenance records {name}={value!r} but the process saw {child.get(name)!r}; "
            f"a recorded value that was not set is worse than no record"
        )


def test_the_benchmark_was_asked_for_raw_repetitions(stub_run, stub_environment):
    stub_run()
    records = [json.loads(line) for line in stub_environment["trace"].read_text().splitlines() if line.strip()]
    argv = " ".join(records[-1]["argv"])
    assert "--benchmark_repetitions=3" in argv
    assert "--benchmark_report_aggregates_only=false" in argv, (
        "the harness computes its own dispersion; it needs the per-repetition rows"
    )
    assert "--benchmark_out_format=json" in argv


def test_unmeasured_cells_are_stated_never_dropped(stub_run, ops):
    document = stub_run()
    planned = len(support.family_points(ops, "square3", ["small"]))
    measured = {(m["op"], m["arm"], m["scalar"], tuple(sorted(m["shape"].items()))) for m in document["measurements"]}
    assert document["not_measured"], "most of the small grid was not measured and the file says nothing"
    covered_shapes = {tuple(sorted(m["shape"].items())) for m in document["measurements"]}
    assert len(covered_shapes) < planned, "the fixture is supposed to be a partial run"
    assert measured, "no measurement survived"


def test_a_failed_row_is_recorded_as_a_runtime_error(stub_run):
    document = stub_run()
    reasons = {entry["reason"] for entry in document["not_measured"]}
    assert "runtime_error" in reasons, (
        "the stub emitted an error_occurred row; it must become an explicit negative with a reason, "
        f"not vanish. Reasons present: {sorted(reasons)}"
    )
    for entry in document["not_measured"]:
        assert entry["reason"] != "runtime_error" or entry.get("detail"), "a runtime error must say what happened"


def test_the_result_file_holds_flops_per_second_not_gflops(stub_run):
    document = stub_run()
    for measurement in document["measurements"]:
        assert measurement["stats"]["flop_rate"]["median"] > 1e6, measurement["name"]


def test_scope_records_what_was_attempted(stub_run):
    document = stub_run()
    scope = document["scope"]
    assert scope["ops"] == ["GEMM"]
    assert "eigen" in scope["arms"]
    assert scope["scalars"] == ["f64"]
    assert scope["threads"] == [1]
    assert scope.get("shape_groups", {}).get("GEMM") == ["small"]


def test_provenance_gaps_accompany_every_null_it_claims(stub_run, schema):
    document = stub_run()
    pointers = {gap["field"] for gap in document["provenance_gaps"]}
    for pointer in pointers:
        try:
            value = support.pointer_get(document, pointer)
        except (KeyError, IndexError):
            continue
        assert value is None or value == "unknown", (
            f"{pointer} is declared a provenance gap but holds {value!r}"
        )


def test_the_skip_envelope_matches_the_header(run_module):
    """One grammar, two languages, nothing but this linking them.

    bench_compare.h writes the envelope; run.py parses the reason out of it to
    file the cell as a machine fact rather than a library defect. If they drift,
    every structured skip silently becomes a runtime_error and the published page
    reports "too large for this machine" as a failure of Eigen.
    """
    header = (support.COMPARISON_DIR / "bench_compare.h").read_text()
    opener = re.search(r'#define EIGEN_BENCH_SKIP_ENVELOPE_OPEN "([^"]*)"', header)
    closer = re.search(r'#define EIGEN_BENCH_SKIP_ENVELOPE_CLOSE "([^"]*)"', header)
    assert opener and closer, "bench_compare.h no longer defines the skip envelope"

    classify = support.resolve_callable(run_module, "classify_skip")
    emitted = f"{opener.group(1)}out_of_memory{closer.group(1)}operands need 6.75 GiB, budget is 4.00 GiB"
    reason, detail = classify(emitted)
    assert reason == "out_of_memory", f"run.py did not parse the envelope the header emits: {emitted!r}"
    assert detail == "operands need 6.75 GiB, budget is 4.00 GiB"


@pytest.mark.parametrize(
    "message, expected_reason",
    [
        ("[eigen-bench:skip:out_of_memory] operands need 6.75 GiB, budget is 4.00 GiB", "out_of_memory"),
        ("[eigen-bench:skip:shape_unsupported] dimension 3000000000 does not fit", "shape_unsupported"),
        # No envelope: a plain SkipWithError, which is what every other benchmark
        # in the tree already uses for a genuine failure.
        ("gemm result disagrees with Eigen at m:8 n:8 k:8", "runtime_error"),
        ("", "runtime_error"),
        (None, "runtime_error"),
        # A reason the binary is not allowed to claim, and one that is not in the
        # schema at all. Both degrade rather than inventing a category that would
        # be rejected at write time.
        ("[eigen-bench:skip:machine_unavailable] nice try", "runtime_error"),
        ("[eigen-bench:skip:not_a_reason] typo", "runtime_error"),
    ],
)
def test_classify_skip(run_module, message, expected_reason):
    classify = support.resolve_callable(run_module, "classify_skip")
    assert classify(message)[0] == expected_reason


def test_benchmark_skip_reasons_are_a_subset_of_the_schema(run_module):
    """A reason the binary may claim must be one the result file can carry."""
    import json

    schema = json.loads((support.COMPARISON_DIR / "result_schema.json").read_text())
    allowed = set(schema["$defs"]["not_measured_entry"]["properties"]["reason"]["enum"])
    claimable = set(getattr(run_module, "BENCHMARK_SKIP_REASONS"))
    assert claimable <= allowed, f"{sorted(claimable - allowed)} would fail schema validation at write time"


def test_the_memory_budget_is_passed_to_the_binary_but_not_called_threading(run_module, tmp_path):
    """The budget must reach the child, and must not be recorded as a thread control.

    provenance.threading.env is documented as exactly the variables that control
    threading -- an absent variable there means something different from one set
    to its default. A memory budget in that mapping would be a false statement
    about what the run controlled.
    """
    build_thread_env = support.resolve_callable(run_module, "build_thread_env")
    env = build_thread_env(1)
    assert "EIGEN_BENCH_MEMORY_BUDGET_BYTES" not in env

    load_machine_profile = support.resolve_callable(run_module, "load_machine_profile")
    text = (MACHINES / "testmachine.toml").read_text()
    text += "\n[memory]\nbenchmark_budget_bytes = 4294967296\n"
    path = tmp_path / "budgeted" / "testmachine.toml"
    path.parent.mkdir()
    path.write_text(text)
    assert load_machine_profile(path).memory_budget_bytes == 4294967296

    # And a profile that declares none leaves the budget unset, so a machine with
    # no measured ceiling enforces nothing rather than guessing one.
    assert load_machine_profile(MACHINES / "testmachine.toml").memory_budget_bytes is None


SHIPPED_MACHINES = support.COMPARISON_DIR / "machines"


def test_an_isa_targets_notes_reach_the_result_file(stub_run, tmp_path):
    """A caveat in a TOML comment is a caveat the reader never sees.

    An opt-in backend rarely covers every operation -- Eigen's SME has a GEMM
    kernel and nothing else -- so a page can carry a row labelled with one
    instruction set and measured with another. That is something the run
    established and proceeded under, so it belongs in run.notes with the rest.
    """
    machines = tmp_path / "machines"
    machines.mkdir()
    for source in MACHINES.glob("*.toml"):
        text = source.read_text()
        isa_target = re.search(r'default_isa_target\s*=\s*"([^"]+)"', text).group(1)
        text += f'\n[isa."{isa_target}"]\nnotes = "only GEMM has a kernel here"\n'
        (machines / source.name).write_text(text)

    document = stub_run(extra=["--machines-dir", str(machines)])
    notes = document["provenance"]["run"]["notes"] or ""
    assert "only GEMM has a kernel here" in notes, f"the ISA target's caveat never reached the run\n{notes!r}"
    # Named, or a reader of a multi-ISA merge cannot tell which target it applies to.
    assert "ISA target" in notes

    assert "only GEMM" not in (stub_run()["provenance"]["run"]["notes"] or ""), (
        "a target with no notes must add none"
    )



@pytest.mark.parametrize("path", sorted(SHIPPED_MACHINES.glob("*.toml")), ids=lambda p: p.stem)
def test_every_shipped_machine_profile_loads(run_module, path):
    """The profiles in the tree, not the fixtures the rest of this module uses.

    Every other test here points --machines-dir at tests/fixtures/machines, so
    nothing exercised machines/*.toml at all -- and a validation rule added to
    parse_machine_profile invalidated one of them with no test able to see it.
    A profile that does not load is a machine that cannot be measured.
    """
    load_machine_profile = support.resolve_callable(run_module, "load_machine_profile")
    profile = load_machine_profile(path)
    assert profile.id == path.stem
    assert profile.default_isa_target in profile.isa_targets
    for target in profile.isa_targets:
        assert target in profile.isa, f"isa_targets names {target!r} with no [isa.{target!r}] block"


def test_isa_flags_reach_the_compiler(run_module, tmp_path):
    """Otherwise the page names an instruction set the binary never used.

    An ISA target's `flags` are what select the instruction set -- Eigen's SME
    backend is opt-in behind -DEIGEN_ARM64_USE_SME, and without it the headers
    compile to the same NEON kernels. They were recorded in provenance and never
    passed to CMake, so a profile could publish "built with -mcpu=apple-m4
    -DEIGEN_ARM64_USE_SME" over a binary compiled with neither: an SME page
    produced by a NEON build, with no symptom except numbers that look like the
    other ISA target's.
    """
    configure_command = support.resolve_callable(run_module, "configure_command")
    load_machine_profile = support.resolve_callable(run_module, "load_machine_profile")

    source = MACHINES / "testmachine.toml"
    text = source.read_text()
    isa_target = load_machine_profile(source).default_isa_target
    text += f'\n[isa."{isa_target}"]\nflags = ["-mcpu=apple-m4", "-DEIGEN_ARM64_USE_SME"]\n'
    # The loader requires id == filename stem.
    path = tmp_path / "flagged" / "testmachine.toml"
    path.parent.mkdir()
    path.write_text(text)

    command = configure_command(
        source_dir=tmp_path / "src",
        build_dir=tmp_path / "build",
        machine=load_machine_profile(path),
        isa_target=isa_target,
        arm=None,
        cxx_standard=17,
    )
    flags = [arg for arg in command if arg.startswith("-DCMAKE_CXX_FLAGS")]
    assert flags, f"the ISA target's flags never reached the configure line:\n{command}"
    assert "-mcpu=apple-m4" in flags[0] and "-DEIGEN_ARM64_USE_SME" in flags[0], flags


def test_an_isa_target_that_sets_cxx_flags_twice_is_refused(run_module, tmp_path):
    """`flags` and a cmake_options -DCMAKE_CXX_FLAGS are two ways to say one thing.

    The cmake_options entry comes last on the command line and would silently
    overwrite `flags`, so the run would record compile flags the binary was not
    built with. That is an authoring mistake with no symptom, so it is refused
    at profile load rather than at measurement time.
    """
    load_machine_profile = support.resolve_callable(run_module, "load_machine_profile")
    harness_error = support.resolve_callable(run_module, "HarnessError")

    source = MACHINES / "testmachine.toml"
    text = source.read_text()
    isa_target = load_machine_profile(source).default_isa_target
    text += (
        f'\n[isa."{isa_target}"]\nflags = ["-mcpu=apple-m4"]\n'
        'cmake_options = ["-DCMAKE_CXX_FLAGS=-mcpu=generic"]\n'
    )
    path = tmp_path / "conflicting" / "testmachine.toml"
    path.parent.mkdir()
    path.write_text(text)

    with pytest.raises(harness_error) as excinfo:
        load_machine_profile(path)
    assert "flags" in str(excinfo.value) and isa_target in str(excinfo.value)


def test_a_load_average_above_the_profile_is_stated_with_the_numbers(stub_run, tmp_path):
    """--allow-noisy must not make the noise invisible.

    Breaching the profile's max_load_avg is only reachable by asking for it, and
    run.py warns on stderr -- but stderr is not the published page. The raw
    load averages are recorded either way; what a reader cannot reconstruct from
    them is what THIS machine declares as quiet, so the exceedance itself has to
    travel with the numbers.
    """
    machines = tmp_path / "machines"
    machines.mkdir()
    for source in MACHINES.glob("*.toml"):
        text = source.read_text()
        assert "max_load_avg" in text, source
        text = re.sub(r"^max_load_avg\s*=.*$", "max_load_avg = 0.0", text, flags=re.M)
        (machines / source.name).write_text(text)

    document = stub_run(extra=["--machines-dir", str(machines)])
    notes = document["provenance"]["run"]["notes"] or ""
    assert "load average" in notes, f"the breached threshold never reached the result file\n{notes!r}"
    assert "0.00" in notes, "the note must state the threshold that was breached, not just that one was"
    assert "--allow-noisy" in notes, "the note must say the run was allowed to proceed"

    # And the unbreached case stays quiet, or the note is decoration rather than
    # a caveat: the committed fixtures declare max_load_avg = 1000.0.
    clean = stub_run()
    assert "load average" not in (clean["provenance"]["run"]["notes"] or "")


UNPINNABLE_REASON = "this fixture machine exposes no CPU affinity API, so the run could not be confined"


def test_no_caveat_is_recorded_in_both_channels(stub_run, tmp_path):
    """One caveat, one channel.

    `provenance_gaps` says "could not establish"; `run.notes` says "established
    and proceeded anyway". They render on the published page under separate
    headings that mean different things, so a caveat written to both is stated
    twice and the second statement is the wrong one. The unpinnable-CPU
    paragraph used to be in both.

    The committed fixtures declare no [pinning] block at all, so the machine is
    doctored here to actually reach the condition -- without that this test
    passes against the duplication it exists to forbid.
    """
    machines = tmp_path / "machines"
    machines.mkdir()
    for source in MACHINES.glob("*.toml"):
        text = source.read_text()
        text = re.sub(r"^\[pinning\]\n(?:(?!\[).*\n)*", "", text, flags=re.M)
        text += f'\n[pinning]\ntool = "none"\nunavailable_reason = "{UNPINNABLE_REASON}"\n'
        (machines / source.name).write_text(text)

    document = stub_run(extra=["--machines-dir", str(machines)])
    notes = document["provenance"]["run"]["notes"] or ""
    reasons = {str(gap.get("reason") or "") for gap in document["provenance_gaps"]}
    assert UNPINNABLE_REASON in reasons, (
        "the doctored profile did not reach the unpinnable case, so this test would prove nothing"
    )
    for gap in document["provenance_gaps"]:
        reason = str(gap.get("reason") or "")
        if not reason:
            continue
        assert reason not in notes, (
            f"{gap['field']} is recorded as a provenance gap AND repeated in run.notes; "
            f"the page would state it twice under two headings that do not mean the same thing"
        )


def test_argv_is_recorded_so_the_run_can_be_repeated(stub_run):
    document = stub_run()
    argv = document["provenance"]["harness"]["argv"]
    assert "--machine" in argv and "testmachine" in argv


def test_dry_run_measures_nothing(stub_environment):
    proc = support.run_cli(
        "run.py",
        [
            "--machine", "testmachine",
            "--machines-dir", str(MACHINES),
            "--build-dir", str(stub_environment["build_dir"]),
            "--no-configure", "--no-build", "--ops", "GEMM", "--scalars", "f64", "--groups", "small",
            "--allow-dirty", "--allow-noisy", "--dry-run",
        ],
        env=stub_environment["env"],
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip(), "--dry-run must print the plan to stdout"
    assert not stub_environment["trace"].exists(), "--dry-run invoked the benchmark"


def test_list_cells_prints_one_self_describing_line_per_cell(stub_environment, ops):
    proc = support.run_cli(
        "run.py",
        [
            "--machine", "testmachine",
            "--machines-dir", str(MACHINES),
            "--build-dir", str(stub_environment["build_dir"]),
            "--no-configure", "--no-build", "--ops", "GEMM", "--scalars", "f64", "--groups", "small",
            "--allow-dirty", "--allow-noisy", "--list-cells",
        ],
        env=stub_environment["env"],
    )
    assert proc.returncode == 0, proc.stderr
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    assert lines
    for line in lines:
        fields = line.split()
        assert len(fields) >= 4, line
        assert fields[0] in ops["ops"], line
        for dimension in fields[3:]:
            assert ":" in dimension, f"{line}: dimensions must stay self-describing"
    assert not stub_environment["trace"].exists(), "--list-cells invoked the benchmark"


def test_an_unknown_machine_is_a_configuration_error(stub_environment):
    proc = support.run_cli(
        "run.py",
        ["--machine", "nosuchmachine", "--machines-dir", str(MACHINES), "--no-configure", "--no-build", "--dry-run"],
        env=stub_environment["env"],
    )
    assert proc.returncode == 2, proc.stderr
    assert "nosuchmachine" in (proc.stderr + proc.stdout)


def test_an_unknown_op_is_a_configuration_error(stub_environment):
    proc = support.run_cli(
        "run.py",
        [
            "--machine", "testmachine", "--machines-dir", str(MACHINES),
            "--no-configure", "--no-build", "--dry-run", "--ops", "NOSUCHOP",
        ],
        env=stub_environment["env"],
    )
    assert proc.returncode == 2, proc.stderr


def test_an_unknown_shape_group_is_a_configuration_error(stub_environment):
    proc = support.run_cli(
        "run.py",
        [
            "--machine", "testmachine", "--machines-dir", str(MACHINES),
            "--no-configure", "--no-build", "--dry-run", "--ops", "GEMM", "--groups", "enormous",
        ],
        env=stub_environment["env"],
    )
    assert proc.returncode == 2, proc.stderr


def test_every_executable_failing_at_runtime_exits_six(stub_run, stub_environment):
    stub_run(expect=6, env={"STUB_BENCH_FAIL_RC": "1"})


def test_a_nonzero_exit_with_a_complete_output_file_still_yields_its_measurements(stub_run):
    """The comparison binaries exit 1 whenever ANY shape disagreed with Eigen --
    that is what ErrorTrackingReporter is for -- and Google Benchmark has written
    every row by then.  Discarding the file on the status alone threw away every
    shape that was fine because one was not, and made the per-row failure
    handling below it unreachable for a real binary."""
    proc = stub_run(expect=6, env={"STUB_BENCH_ERROR_RC": "1"})
    document = json.loads(proc.stdout)
    assert document["measurements"], (
        "the benchmark wrote a complete output file and exited 1 for a failed shape; "
        "every measurement in that file was discarded"
    )
    reasons = {entry["reason"] for entry in document["not_measured"]}
    assert "runtime_error" in reasons, "the row the binary flagged must still be a runtime_error"
    measured = {(m["arm"], tuple(sorted(m["shape"].items()))) for m in document["measurements"]}
    assert ("eigen", (("k", 64), ("m", 64), ("n", 64))) in measured
    # And the unit is still a failure: exit 6 above, not 0.


def test_a_nonzero_exit_with_no_output_file_marks_every_cell(stub_run):
    """The other half of the same branch: nothing was written, so nothing can be
    distilled and every planned cell has to be stated as a runtime error."""
    proc = stub_run(expect=6, env={"STUB_BENCH_FAIL_RC": "1"})
    document = json.loads(proc.stdout)
    assert not document["measurements"]
    assert document["not_measured"]
    assert {entry["reason"] for entry in document["not_measured"]} == {"runtime_error"}


# --------------------------------------------------------------------------
# What the build recorded about itself
# --------------------------------------------------------------------------


def _write_vendor_info(build_dir, **reference):
    """A minimal vendor_info.json beside the stub binaries."""
    directory = build_dir / "comparison"
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0.0",
        "build_target": "bench_comparison_all",
        "reference": {"available": bool(reference.get("arm")), **reference},
        "targets": [],
    }
    (directory / "vendor_info.json").write_text(json.dumps(payload, indent=2))
    return directory / "vendor_info.json"


def test_a_reference_arm_requested_against_an_eigen_only_build_is_refused(stub_environment):
    """With --no-configure the build tree is whatever is already there.

    The guard used to be skipped whenever the build recorded NO reference
    library, which is the dangerous half: an Eigen-only tree accepted a request
    for Accelerate, ran its binaries, found no reference row for anything, and
    filed the whole reference column as not_implemented -- under provenance
    naming Accelerate. That publishes "the library does not implement this" about
    a library that was never linked."""
    build_dir = stub_environment["build_dir"] / "aarch64-neon__accelerate"
    _write_vendor_info(build_dir, arm="")
    proc = support.run_cli(
        "run.py",
        [
            "--machine", "testmachine", "--machines-dir", str(MACHINES),
            "--build-dir", str(stub_environment["build_dir"]),
            "--no-configure", "--no-build", "--ops", "GEMM", "--scalars", "f64",
            "--groups", "small", "--arms", "accelerate", "--allow-dirty", "--allow-noisy",
            "--out", "-",
        ],
        env=stub_environment["env"],
    )
    assert proc.returncode == 2, f"expected a configuration error, got {proc.returncode}\n{proc.stderr}"
    assert "no reference library" in proc.stderr


def test_a_mismatched_reference_arm_is_still_refused(stub_environment):
    build_dir = stub_environment["build_dir"] / "aarch64-neon__accelerate"
    _write_vendor_info(build_dir, arm="openblas")
    proc = support.run_cli(
        "run.py",
        [
            "--machine", "testmachine", "--machines-dir", str(MACHINES),
            "--build-dir", str(stub_environment["build_dir"]),
            "--no-configure", "--no-build", "--ops", "GEMM", "--scalars", "f64",
            "--groups", "small", "--arms", "accelerate", "--allow-dirty", "--allow-noisy",
            "--out", "-",
        ],
        env=stub_environment["env"],
    )
    assert proc.returncode == 2, proc.stderr
    assert "openblas" in proc.stderr


def test_an_eigen_only_unit_accepts_a_build_that_names_no_reference(stub_environment):
    """The other side of the same guard: an Eigen-only unit has nothing to
    attribute to a vendor, so an Eigen-only build is exactly right for it."""
    build_dir = stub_environment["build_dir"] / "aarch64-neon__eigen"
    _write_vendor_info(build_dir, arm="")
    proc = support.run_cli(
        "run.py",
        [
            "--machine", "testmachine", "--machines-dir", str(MACHINES),
            "--build-dir", str(stub_environment["build_dir"]),
            "--no-configure", "--no-build", "--ops", "GEMM", "--scalars", "f64",
            "--groups", "small", "--arms", "eigen", "--allow-dirty", "--allow-noisy",
            "--out", "-",
        ],
        env=stub_environment["env"],
    )
    assert proc.returncode == 0, proc.stderr


def test_the_families_a_build_can_call_beat_the_vendors_declaration(run_module):
    """`provides` is what a vendor row declares; `families` is that reconciled
    with what configuring found. A vendor shipping no LAPACK still gains the
    family when find_package(LAPACK) supplied one, and reading `provides` alone
    would report ?potrf absent on a build that links it perfectly well."""
    provides = support.resolve_callable(run_module, "vendor_info_provides")
    assert provides({"reference": {"provides": ["blas", "cblas"]}}) == ["blas", "cblas"]
    assert provides(
        {"reference": {"provides": ["blas", "cblas"], "families": ["blas", "cblas", "lapack"]}}
    ) == ["blas", "cblas", "lapack"]
    assert provides(None) == []


def test_a_separately_found_lapack_is_not_attributed_to_the_blas_vendor(run_module):
    """A POTRF row is headed with the reference arm, and a reader takes that to
    be what computed it. When the vendor ships no LAPACK, the ?potrf in that row
    belongs to a different package and the page has to say so."""
    note = support.resolve_callable(run_module, "vendor_info_lapack_note")
    assert note({"reference": {"arm": "netlib", "lapack": {"available": True, "provider": "vendor"}}}) is None
    assert note({"reference": {"arm": "netlib", "lapack": {"available": False, "provider": ""}}}) is None
    separate = note(
        {
            "reference": {
                "arm": "netlib",
                "lapack": {"available": True, "provider": "separate", "libraries": ["/usr/lib/liblapack.so"]},
            }
        }
    )
    assert separate and "netlib" in separate and "liblapack" in separate


# --------------------------------------------------------------------------
# Explicit negatives: a reason belongs to the shape it was recorded for
# --------------------------------------------------------------------------


def test_a_whole_grid_still_collapses_to_one_entry(run_module, registry):
    """The other side of the same rule.  `shape: null` exists so a wholly
    unimplemented operation writes one row instead of hundreds of identical ones,
    and refusing to collapse anything would trade one bug for that one."""
    collapse = support.resolve_callable(run_module, "collapse_not_measured")
    plan = support.resolve_callable(run_module, "plan_cells")
    cells = plan(registry, ops=["GEMM"], arms=["accelerate"], scalars=["f64"], groups=["small"])
    assert len(cells) > 1
    entries = [
        {"op": c.op, "arm": c.arm, "scalar": c.scalar, "shape": c.shape,
         "threads": c.threads, "reason": "not_implemented", "detail": "d"}
        for c in cells
    ]
    whole = collapse(entries, attempted=cells)
    assert len(whole) == 1 and whole[0]["shape"] is None, whole

    # Drop one point and the claim is no longer about the whole grid.
    part = collapse(entries[:-1], attempted=cells)
    assert len(part) == len(entries) - 1, part
    assert all(entry["shape"] is not None for entry in part)


def test_a_partial_run_keeps_its_per_shape_reasons(stub_run, ops):
    """`shape: null` claims the WHOLE grid, so a partial run must not emit it.

    The stub measures two shapes of the six-point `small` grid and fails a third.
    Collapsing every remaining entry that shares a reason into one whole-grid row
    makes the reducer expand it back over all six points, on top of the
    shape-specific reasons the same run recorded."""
    document = stub_run()
    grid = {tuple(sorted(dict(zip(["m", "n", "k"], point)).items()))
            for point in support.family_points(ops, "square3", ["small"])}
    measured = {(m["arm"], tuple(sorted(m["shape"].items()))) for m in document["measurements"]}
    assert len(measured) < 2 * len(grid), "the fixture is supposed to be a partial run"

    blanket = [e for e in document["not_measured"] if e.get("shape") is None]
    assert not blanket, (
        "a partial run emitted a whole-grid entry; the reducer expands it across every point "
        f"of the grid: {json.dumps(blanket, sort_keys=True)}"
    )

    failed = [
        e for e in document["not_measured"]
        if e["reason"] == "runtime_error" and e["arm"] == "accelerate"
    ]
    assert [e["shape"] for e in failed] == [{"m": 96, "n": 96, "k": 96}], (
        "the failed shape must stay attached to the shape that failed"
    )


def test_a_shape_specific_reason_survives_reduction(stub_run, tmp_path):
    """End to end: run.py -> reduce.py, on the reason that a blanket entry ate.

    The reducer resolves same-key contributions by timestamp, and every entry in
    one result file carries the run's timestamp, so a whole-grid row appended
    after a shape-specific one simply replaces it -- `runtime_error` at m:96
    became `not_implemented`, which on a published page reads as "Eigen does not
    implement this" about a shape that was measured and failed."""
    document = stub_run()
    result = support.write_json(tmp_path / "partial-run.json", document)
    merged = support.cli_json("reduce.py", [str(result)])

    def cell(shape):
        for candidate in merged["cells"]:
            if candidate["op"] == "GEMM" and candidate["scalar"] == "f64" and candidate["shape"] == shape:
                return candidate
        raise AssertionError(f"no cell for {shape}")

    failed = cell({"m": 96, "n": 96, "k": 96})["arms"]["accelerate"]
    assert failed["state"] == "not_measured"
    assert failed["reason"] == "runtime_error", (
        "the shape the binary reported an error for was overwritten during reduction: "
        f"{json.dumps(failed, sort_keys=True)}"
    )
    # The shapes that really were never registered keep saying so.
    assert cell({"m": 24, "n": 24, "k": 24})["arms"]["accelerate"]["reason"] == "not_implemented"
    # And a measured shape is untouched by any of it.
    assert cell({"m": 64, "n": 64, "k": 64})["arms"]["accelerate"]["state"] == "measured"


def test_a_run_that_measured_none_of_its_runnable_cells_is_a_failure(stub_run, tmp_path, raw_gbench):
    """A binary that exits 0 having produced nothing for the plan is a broken
    run, not a coverage manifest reporting the operation unimplemented. Left as
    exit 0 it publishes `not_implemented` for a whole grid on a machine that has
    the operation."""
    outside_the_plan = support.deep_copy(raw_gbench)
    # `small` stops at 128; every row here is a `medium` shape, so the plan keys
    # none of them and the distillation is empty.
    outside_the_plan["benchmarks"] = [
        row for row in outside_the_plan["benchmarks"] if "m:256/" in str(row.get("run_name", ""))
    ]
    assert outside_the_plan["benchmarks"], "the fixture no longer carries an out-of-group row"
    canned = tmp_path / "outside-the-plan.json"
    canned.write_text(json.dumps(outside_the_plan))

    proc = stub_run(expect=6, env={"STUB_BENCH_JSON": str(canned)})
    assert "none was measured" in proc.stderr


def test_version_and_help_exit_zero():
    for flag in ("--version", "--help"):
        proc = support.run_cli("run.py", [flag])
        assert proc.returncode == 0, f"run.py {flag}: {proc.stderr}"
        assert proc.stdout.strip()


def test_a_missing_required_option_is_a_usage_error():
    proc = support.run_cli("run.py", [])
    assert proc.returncode == 1, f"argparse's default exit 2 collides with the config-error code\n{proc.stderr}"


def test_probe_git_reports_dirtiness(tmp_path, run_module):
    """Detection itself, against a purpose-built repo rather than ambient state.

    The end-to-end refusal below can only run when the checkout it measures is
    dirty.  This one builds both states explicitly, so the logic behind the
    refusal is covered in a clean checkout and in CI too.
    """
    probe_git = support.resolve_callable(run_module, "probe_git")
    repo = tmp_path / "repo"
    repo.mkdir()
    run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, check=True)
    run("init", "-q")
    run("config", "user.email", "t@example.invalid")
    run("config", "user.name", "t")
    (repo / "a.txt").write_text("one\n")
    run("add", "a.txt")
    run("commit", "-qm", "initial")

    clean = probe_git(repo)
    assert clean.available and not clean.dirty, "a freshly committed repo must read clean"

    (repo / "untracked.txt").write_text("two\n")
    assert probe_git(repo).dirty, "an untracked file makes a measurement unreproducible"
    (repo / "untracked.txt").unlink()

    (repo / "a.txt").write_text("modified\n")
    assert probe_git(repo).dirty, "a modified tracked file makes a measurement unreproducible"


def test_the_runs_own_build_tree_does_not_make_the_worktree_dirty(tmp_path, run_module):
    """The guard must not fire on its own output.

    --build-dir defaults to a relative path, so the harness's build trees land
    inside the worktree, and Eigen's .gitignore does not cover `build*/`.  Before
    this was scoped, the very first `run.py` invocation in a clean checkout was
    refused as dirty -- and the only way past it, --allow-dirty, marks the result
    unreproducible.  Everyone would have learned to pass it always.
    """
    probe_git = support.resolve_callable(run_module, "probe_git")
    repo = tmp_path / "repo"
    repo.mkdir()
    run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, check=True)
    run("init", "-q")
    run("config", "user.email", "t@example.invalid")
    run("config", "user.name", "t")
    (repo / "a.txt").write_text("one\n")
    run("add", "a.txt")
    run("commit", "-qm", "initial")

    build = repo / "build-comparison" / "aarch64-neon__openblas"
    build.mkdir(parents=True)
    (build / "CMakeCache.txt").write_text("x\n")
    results = repo / "benchmarks" / "comparison" / "results" / "m4"
    results.mkdir(parents=True)
    (results / "run.json").write_text("{}\n")
    assert probe_git(repo).dirty, "unscoped, an untracked build tree reads as dirty"
    assert not probe_git(repo, repo / "build-comparison", results.parent).dirty, (
        "the run's own build tree and results are outputs, not inputs to the measurement"
    )
    # Each is excluded independently: results alone still leaves the build tree
    # visible, which is what proves the exclusion is not a blanket one.
    assert probe_git(repo, results.parent).dirty
    assert probe_git(repo, repo / "build-comparison").dirty, (
        "the first successful measurement used to make the next run refuse as dirty"
    )
    shutil.rmtree(repo / "benchmarks")

    # Scoping must not become a hole: a real source change alongside the build
    # tree still has to be caught.
    (repo / "a.txt").write_text("modified\n")
    assert probe_git(repo, repo / "build-comparison").dirty, (
        "excluding the build tree must not stop a modified source from reading dirty"
    )
    (repo / "a.txt").write_text("one\n")
    (repo / "benchmarks").mkdir(exist_ok=True)
    (repo / "benchmarks" / "sneaky.cpp").write_text("int main(){}\n")
    assert probe_git(repo, repo / "build-comparison").dirty, (
        "an untracked source outside the build tree still makes the measurement unreproducible"
    )

    # --build-dir pointing at the repo root would exclude everything and silently
    # disable the guard, so it is refused as an exclusion and the source is still
    # seen.
    assert probe_git(repo, repo).dirty, "--build-dir '.' must not disable the guard"
    (repo / "benchmarks" / "sneaky.cpp").unlink()

    # A build tree outside the worktree never appears in status, so the exclusion
    # has nothing to do; it must degrade to a no-op rather than raising on the
    # relative_to.  Checked against an otherwise clean tree, so a leftover would
    # be visible as a failure rather than masked by another source of dirt.
    shutil.rmtree(repo / "build-comparison")
    assert not probe_git(repo, tmp_path / "elsewhere").dirty


def test_a_dirty_worktree_is_refused_unless_allowed(stub_environment):
    # Make the checkout dirty rather than hoping it already is: this used to skip
    # whenever the tree was clean, which is precisely the case CI runs in.
    sentinel = support.REPO_ROOT / ".eigen-bench-dirty-probe.tmp"
    sentinel.write_text("transient fixture; removed by this test\n")
    try:
        proc = support.run_cli(
            "run.py",
            [
                "--machine", "testmachine", "--machines-dir", str(MACHINES),
                "--build-dir", str(stub_environment["build_dir"]),
                "--no-configure", "--no-build", "--ops", "GEMM", "--scalars", "f64", "--groups", "small",
                "--allow-noisy", "--out", "-",
            ],
            env=stub_environment["env"],
        )
    finally:
        sentinel.unlink(missing_ok=True)
    assert proc.returncode == 3, (
        "a dirty Eigen worktree makes the measurement unreproducible and must be refused without "
        f"--allow-dirty\n{proc.stderr[-2000:]}"
    )


def _provenance_inputs(run_module, machine, *, load_before, load_after):
    """A ProvenanceInputs carrying the two load readings and nothing else of interest.

    Every other field is filled with the least interesting legal value, so a
    failure in a test using this is about the load averages and not about
    whatever else the provenance block happens to contain.
    """
    import datetime as _dt

    host = run_module.HostFacts(
        hostname="host",
        system="Linux",
        os_name="Linux",
        os_release="test",
        kernel=None,
        arch="x86_64",
        cpu_model=machine.cpu_model,
        logical_cpus=1,
        sockets=1,
        cores_per_socket=1,
        threads_per_core=1,
        performance_cores=None,
        efficiency_cores=None,
        frequency_governor="performance",
        memory_total_bytes=1 << 30,
        transparent_huge_pages="never",
        load_avg=tuple(load_before) if load_before else None,
    )
    return run_module.ProvenanceInputs(
        timestamp=_dt.datetime(2026, 1, 1, tzinfo=_dt.timezone.utc),
        machine=machine,
        host=host,
        git=run_module.GitFacts("a" * 40, "a" * 9, False, "main", "a" * 9, True),
        eigen_version={},
        isa_target=machine.default_isa_target,
        isa_flags=(),
        isa_notes=None,
        arm_key=None,
        arm_profile=None,
        arm_context={},
        compiler_id="GNU",
        compiler_version="13.3.0",
        compiler_path="/usr/bin/c++",
        cxx_standard=17,
        cxx_flags=(),
        cmake_build_type="Release",
        benchmark_library_version="v1.9.5",
        threads=1,
        thread_env={},
        eigen_nb_threads=None,
        pinning=run_module.PinningPlan(tool="none"),
        caches=(),
        cpu_scaling_enabled=False,
        argv=("--machine", machine.id),
        benchmark_argv=(),
        executable=None,
        repetitions=1,
        min_time="0.5s",
        benchmark_filter=None,
        memory_budget_bytes=None,
        load_avg_before=load_before,
        load_avg_after=load_after,
        duration_s=1.0,
        notes=None,
    )


def test_the_harness_own_build_does_not_forge_an_allow_noisy_caveat(run_module):
    """The noise caveat must describe the machine, not the harness running on it.

    run.py checks the load average BEFORE the run and refuses without
    --allow-noisy; it then builds with `cmake --build --parallel N` and measures.
    A 1-minute load average sampled after all that reflects the harness itself.
    Deciding the caveat on the peak of the two readings therefore stamped clean
    runs with "the run was allowed to proceed with --allow-noisy" on command
    lines that never passed it -- a result file contradicting its own
    provenance.harness.argv, and, once merged, a coverage page attributing an
    override to an operator who never made one.

    Observed on three hosts before the fix, most starkly at load_avg_before 0.11
    on an idle 72-core machine against a 0.50 threshold.
    """
    import dataclasses

    load_machine_profile = support.resolve_callable(run_module, "load_machine_profile")
    machine = dataclasses.replace(
        load_machine_profile(MACHINES / "testmachine.toml"), max_load_avg=0.5
    )

    # The guard passed cleanly: 0.37 is well under 0.50. 0.81 is what the
    # harness's own -j build left in the 1-minute average by the time the run
    # ended.
    provenance, _ = run_module.assemble_provenance(
        _provenance_inputs(run_module, machine, load_before=[0.37, 0.3, 0.2], load_after=[0.81, 0.4, 0.3])
    )
    notes = provenance["run"]["notes"] or ""
    assert "--allow-noisy" not in notes, (
        "a run that never breached the threshold claimed it was forced through anyway:\n" + notes
    )
    assert "load average" not in notes, (
        "the harness's own build raised a noise caveat against the machine:\n" + notes
    )
    # ... and both readings are still on the record, so nothing was hidden.
    assert provenance["run"]["load_avg_before"][0] == pytest.approx(0.37)
    assert provenance["run"]["load_avg_after"][0] == pytest.approx(0.81)

    # The genuine case is unaffected: a breach at guard time is only reachable
    # with --allow-noisy, and still has to travel with the numbers.
    provenance, _ = run_module.assemble_provenance(
        _provenance_inputs(run_module, machine, load_before=[0.87, 0.5, 0.4], load_after=[0.9, 0.5, 0.4])
    )
    notes = provenance["run"]["notes"] or ""
    assert "0.87" in notes and "0.50" in notes, (
        "a breached threshold must state both the reading and what the profile calls quiet:\n" + notes
    )
    assert "--allow-noisy" in notes
