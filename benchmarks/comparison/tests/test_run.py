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
