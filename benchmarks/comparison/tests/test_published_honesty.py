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
