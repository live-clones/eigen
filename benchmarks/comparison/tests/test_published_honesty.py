# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

"""Regressions for defects that put a WRONG number on a published page.

Each test here corresponds to a defect that shipped: the pipeline ran, exited 0,
validated against the schema and rendered a well-formed page that said something
untrue.  A crash is recoverable; a plausible wrong number under Eigen's name is
not, so these assert the specific dishonesty rather than merely that the tools
run.
"""

import json

import pytest

import harness_support as support


@pytest.fixture(scope="module")
def run_module():
    return support.import_impl("run.py")


@pytest.fixture(scope="module")
def registry(run_module):
    load = support.resolve_callable(run_module, "load_ops_registry", "load_ops_toml")
    return load(support.OPS_TOML)


def _reduce_to(tmp_path, result, name="r.json"):
    path = tmp_path / name
    path.write_text(json.dumps(result))
    out = tmp_path / "merged.json"
    proc = support.run_cli("reduce.py", [str(path), "--out", str(out)])
    assert proc.returncode == 0, proc.stderr
    return json.loads(out.read_text())


# --------------------------------------------------------------------------
# A presentation-time --baseline must not relabel a precomputed ratio
# --------------------------------------------------------------------------


@pytest.mark.parametrize("script", ["render.py", "plots.py"])
def test_a_mismatched_baseline_is_refused_rather_than_relabelled(tmp_path, script):
    """`ratio` is computed once, by reduce.py, against one arm.

    Nothing downstream can recompute it -- a cell keeps only the arms' rates and
    that single number -- so honouring a different `--baseline` at render time
    retitled the page and the ratio column while the ratios underneath still
    divided by the original arm.  The observed output was a page headed "Eigen
    versus openblas" whose own neighbouring column read "Apple Accelerate
    GFLOP/s", over Eigen/Accelerate ratios, at exit 0.
    """
    merged = json.loads((support.FIXTURES / "merged" / "gemm_merged.json").read_text())
    recorded = merged.get("baseline")
    assert recorded, "fixture must record a baseline for this test to mean anything"

    out = tmp_path / "out"
    args = [str(support.FIXTURES / "merged" / "gemm_merged.json"), "--baseline", "openblas"]
    args += ["--out-dir", str(out)] if script == "plots.py" else ["--format", "markdown", "--out", str(out / "t.md")]
    out.mkdir(parents=True, exist_ok=True)
    proc = support.run_cli(script, args)

    assert proc.returncode != 0, (
        f"{script} accepted a --baseline naming an arm the ratios were not computed against; "
        "that publishes a ratio against one library under another library's name\n" + proc.stdout[-2000:]
    )
    assert "openblas" in proc.stderr and recorded in proc.stderr, (
        "the refusal must name both the requested and the recorded baseline so the operator knows "
        f"to re-run reduce.py\n{proc.stderr}"
    )


@pytest.mark.parametrize("script", ["render.py", "plots.py"])
def test_the_recorded_baseline_is_still_accepted(tmp_path, script):
    """The guard must not break the legitimate case it exists to protect."""
    merged_path = support.FIXTURES / "merged" / "gemm_merged.json"
    recorded = json.loads(merged_path.read_text())["baseline"]
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    args = [str(merged_path), "--baseline", recorded]
    args += ["--out-dir", str(out)] if script == "plots.py" else ["--format", "markdown", "--out", str(out / "t.md")]
    proc = support.run_cli(script, args)
    assert proc.returncode == 0, proc.stderr


# --------------------------------------------------------------------------
# A multi-threaded run must not be published as single-threaded
# --------------------------------------------------------------------------


def test_measured_cells_take_the_runs_configured_thread_count(tmp_path):
    """Google Benchmark appends /threads:N only when ->Threads(n) was used.

    These registrations do not, so every name parses back as 1.  Keying measured
    cells on that filed an 8-thread campaign under a "1 thread(s)" heading while
    the explicit negatives, which come from scope.threads, built a second,
    entirely empty 8-thread configuration beside it -- the same run appearing
    twice, once wrong and once hollow.
    """
    result = support.make_result(
        measurements=[support.make_measurement(shape={"m": 64, "n": 64, "k": 64})],
        scope={
            "ops": ["GEMM"],
            "arms": ["eigen", "accelerate"],
            "scalars": ["f64"],
            "threads": [8],
            "shape_groups": {"GEMM": ["small"]},
        },
    )
    merged = _reduce_to(tmp_path, result)

    configs = merged["configs"]
    ids = list(configs) if isinstance(configs, dict) else [c["config_id"] for c in configs]
    assert len(ids) == 1, f"one measurement unit must yield one configuration, got {ids}"
    assert ids[0].endswith("__t8"), f"the run configured 8 threads; published as {ids[0]!r}"
    assert all(cell["threads"] == 8 for cell in merged["cells"]), "cells disagree with the run's thread count"
    # The defect's signature was a SECOND configuration holding no measurements at
    # all: the measured rows went to __t1 and the explicit negatives to __t8, so
    # the same run appeared twice, once mislabelled and once hollow.
    measured_per_config = {}
    for cell in merged["cells"]:
        arms = cell.get("arms", {}) or {}
        measured = any(a.get("state") == "measured" for a in arms.values())
        measured_per_config.setdefault(cell["config_id"], False)
        measured_per_config[cell["config_id"]] |= measured
    hollow = [cid for cid, has in measured_per_config.items() if not has]
    assert not hollow, f"configurations with no measurements at all: {hollow}"


def test_a_thread_count_in_the_name_still_wins(tmp_path):
    """If a registration ever does encode threads, that is the more specific fact."""
    parse = support.resolve_callable(support.import_impl("_common.py"), "parse_benchmark_name")
    named = parse("GEMM/eigen/f64/m:64/n:64/k:64/threads:4")
    assert named["threads"] == 4 and named["threads_in_name"] is True
    plain = parse("GEMM/eigen/f64/m:64/n:64/k:64")
    assert plain["threads"] == 1 and plain["threads_in_name"] is False, (
        "an absent /threads: must be reported as absent, not as an observed 1"
    )


# --------------------------------------------------------------------------
# A measured zero is a broken measurement, not an absent one
# --------------------------------------------------------------------------


def test_a_measured_zero_rate_is_not_reported_as_unmeasured(tmp_path):
    """`and other.get("gflops")` was a truthiness test that a measured 0.0 fails.

    The cell then carried ratio_state "not_measured" and a footnote asserting an
    arm had not been measured, on a page whose coverage manifest counted both
    arms as measured -- two artifacts of the same run contradicting each other.
    """
    result = support.make_result(
        measurements=[
            support.make_measurement(arm="eigen", shape={"m": 64, "n": 64, "k": 64}, gflops=45.0),
            support.make_measurement(arm="accelerate", shape={"m": 64, "n": 64, "k": 64}, gflops=0.0, gflops_mad=0.0),
        ]
    )
    merged = _reduce_to(tmp_path, result)
    cell = next(c for c in merged["cells"] if c["shape"]["m"] == 64)

    assert cell["ratio"] is None, "a zero rate cannot be a divisor"
    assert cell["ratio_state"] != "not_measured", (
        "both arms reported a measurement, so claiming the comparison was not measured is false; "
        f"got {cell['ratio_state']!r}"
    )
    assert cell["ratio_state"] == "degenerate"
    assert merged["coverage"]["totals"]["measured"] > 0, "coverage must still count the arms it measured"


# --------------------------------------------------------------------------
# Caveats must survive the pipeline
# --------------------------------------------------------------------------


def _result(run_id, timestamp, *, dirty=False, gaps=None, lib_version=None, gflops=45.0):
    result = support.make_result(
        run_id=run_id,
        measurements=[support.make_measurement(shape={"m": 64, "n": 64, "k": 64}, gflops=gflops)],
    )
    result["run_id"] = run_id
    result["provenance"]["timestamp_utc"] = timestamp
    result["provenance"]["eigen"]["dirty"] = dirty
    if gaps is not None:
        result["provenance_gaps"] = gaps
    if lib_version is not None:
        for arm, meta in result["provenance"].get("arms", {}).items():
            if arm != "eigen":
                meta["library_version"] = lib_version
    return result


def _reduce_paths(tmp_path, results, extra_args=()):
    paths = []
    for name, payload in results:
        path = tmp_path / name
        path.write_text(json.dumps(payload))
        paths.append(str(path))
    out = tmp_path / "merged.json"
    proc = support.run_cli("reduce.py", [*paths, "--out", str(out), *extra_args])
    assert proc.returncode == 0, proc.stderr
    return json.loads(out.read_text())


def _sole_config(merged):
    configs = merged["configs"]
    values = list(configs.values()) if isinstance(configs, dict) else list(configs)
    assert len(values) == 1, f"expected one configuration, got {len(values)}"
    return values[0]


def test_a_dirty_run_marks_the_configuration_whatever_the_input_order(tmp_path):
    """`eigen_dirty` is not part of config_id, so a clean and a dirty run of the
    same commit share a configuration. Taking the first-read run's value made the
    "not reproducible" warning appear or vanish according to which file sorted
    first."""
    clean = ("a_clean.json", _result("run-clean", "2026-08-01T12:00:00Z", dirty=False))
    dirty = ("b_dirty.json", _result("run-dirty", "2026-08-01T12:00:01Z", dirty=True))
    for order in ([clean, dirty], [dirty, clean]):
        merged = _reduce_paths(tmp_path, order)
        assert _sole_config(merged)["eigen_dirty"] is True, (
            f"a dirty contribution stopped marking the configuration when read in order "
            f"{[name for name, _ in order]}"
        )


def test_provenance_gaps_reach_the_published_coverage_manifest(tmp_path):
    """A gap is the run stating what it could not establish about its own
    environment. Discarding them at the reducer meant no published artifact ever
    mentioned, for instance, that the CPU could not be pinned."""
    gap = {"field": "/provenance/numa/cpu_binding", "reason": "this platform exposes no CPU-affinity API"}
    merged = _reduce_paths(tmp_path, [("r.json", _result("run-a", "2026-08-01T12:00:00Z", gaps=[gap]))])
    assert _sole_config(merged)["provenance_gaps"], "the reducer dropped the run's recorded gaps"

    merged_path = tmp_path / "merged.json"
    out = tmp_path / "out"
    proc = support.run_cli("render.py", [str(merged_path), "--format", "coverage", "--out-dir", str(out)])
    assert proc.returncode == 0, proc.stderr
    text = (out / "coverage.md").read_text()
    assert gap["field"] in text and gap["reason"] in text, (
        "the coverage manifest omits a caveat the run recorded, overstating how controlled the "
        f"measurement was:\n{text}"
    )


def test_on_conflict_first_means_oldest_not_first_filename(tmp_path):
    """Inputs are processed in sorted-pathname order, so "first" used to mean
    "whichever file was named earlier" -- renaming a contribution changed the
    published number."""
    old = _result("run-old", "2026-08-01T12:00:00Z", gflops=10.0)
    new = _result("run-new", "2026-08-05T12:00:00Z", gflops=99.0)
    for names in (("a.json", "b.json"), ("b.json", "a.json")):
        merged = _reduce_paths(tmp_path, [(names[0], old), (names[1], new)], extra_args=("--on-conflict", "first"))
        cell = next(c for c in merged["cells"] if c["shape"]["m"] == 64)
        kept = cell["arms"]["eigen"]["gflops"]
        assert kept == 10.0, (
            f"--on-conflict first kept {kept} with the older run named {names[0]!r}; the winner must be "
            "the oldest measurement, not the first filename"
        )


def test_arm_metadata_ranks_by_parsed_timestamp_not_string_order(tmp_path):
    """Fractional seconds are schema-legal and '.' < 'Z', so a raw string compare
    ranks 12:00:00.500Z BEFORE the whole second it follows -- publishing the older
    reference-library version. reduce.py's own _parse_timestamp docstring warns
    about exactly this."""
    older = _result("run-older", "2026-08-01T12:00:00Z", lib_version="OLD 1.0")
    newer = _result("run-newer", "2026-08-01T12:00:00.500Z", lib_version="NEW 2.0")
    merged = _reduce_paths(tmp_path, [("a.json", older), ("b.json", newer)])
    reference = {arm: meta for arm, meta in merged["arms"].items() if arm != "eigen"}
    assert reference, "fixture must carry a reference arm"
    for arm, meta in reference.items():
        assert meta["library_version"] == "NEW 2.0", (
            f"arm {arm!r} published {meta['library_version']!r}; the run 0.5s later is newer"
        )


# --------------------------------------------------------------------------
# The rate and the elapsed time in one cell must share a clock
# --------------------------------------------------------------------------


def test_the_published_rate_is_wall_clock_not_cpu_time(run_module, registry, raw_gbench):
    """Google Benchmark's kIsIterationInvariantRate counters divide by CPU time
    while the reported real_time is wall clock.

    Taking the counter verbatim put two numbers measured against different clocks
    in one published cell. They agree only while a run is single-threaded and
    undisturbed: a threaded vendor BLAS accumulates CPU time across its workers
    and would be published as several times slower than it is, and a descheduled
    run reports a rate faster than anything that happened.
    """
    rows = [r for r in raw_gbench["benchmarks"] if r.get("run_type") != "aggregate"]
    assert rows, "the fixture must carry iteration rows"
    # Simulate what a real binary emits when CPU time diverges sharply from wall
    # time, as a threaded or contended run does: real_time is unchanged, cpu_time
    # is a quarter of it, and the counter -- which Google Benchmark divides by CPU
    # time -- is correspondingly four times the wall-clock rate.
    for row in rows:
        row["cpu_time"] = float(row["real_time"]) / 4.0
        if "GFLOPS" in row:
            row["GFLOPS"] = float(row["GFLOPS"]) * 4.0

    distill = support.resolve_callable(run_module, "distill_benchmark_json")
    distilled = distill(raw_gbench, registry)
    assert distilled.measurements

    for measurement in distilled.measurements:
        flops = measurement["flops_per_iteration"]
        rate = measurement["stats"]["flop_rate"]["median"]
        seconds = measurement["stats"]["real_time_s"]["median"]
        assert seconds > 0 and rate > 0, measurement["name"]
        implied = flops / seconds
        assert abs(rate - implied) / implied < 1e-6, (
            f"{measurement['name']}: the cell publishes {rate:.6g} flop/s beside an elapsed time of "
            f"{seconds:.6g}s for {flops:.6g} flops, which implies {implied:.6g} flop/s. The rate and "
            "the time are being measured against different clocks."
        )
