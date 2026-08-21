# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

"""`reduce.py`, on the inputs that actually bite.

Every fixture here encodes a way a real result store goes wrong: a reference arm
that covers only part of the grid, a library that was not installed, the same
cell measured twice with different answers, a machine that measured nothing, a
row carrying NaN, a single-repetition run with no dispersion to report, and a
benchmark name that contradicts the registry.  None of these may produce a
number that looks fine.

The reducer is also the only place the flop-rate counter is divided by 1e9
(CONTRACTS.md section 1.6), so the unit tests below pin that boundary from both
sides: raw files hold flops per second, merged files hold GFLOP/s.
"""

import json
import math
import pathlib

import pytest

import harness_support as support

pytestmark = pytest.mark.impl

RESULTS = support.FIXTURES / "results"


def reduce_to(args, **kwargs):
    proc = support.run_cli("reduce.py", args, **kwargs)
    return proc


def merged_from(paths, extra=(), **kwargs):
    proc = reduce_to([*[str(p) for p in paths], *extra], **kwargs)
    assert proc.returncode == 0, f"reduce.py exited {proc.returncode}\n{proc.stderr[-3000:]}"
    return json.loads(proc.stdout)


def cell_of(merged, op="GEMM", shape=None, scalar="f64"):
    for cell in merged["cells"]:
        if cell["op"] == op and cell["scalar"] == scalar and (shape is None or cell["shape"] == shape):
            return cell
    raise AssertionError(f"no {op}/{scalar} cell for shape {shape} among {[c['shape'] for c in merged['cells']]}")


# --------------------------------------------------------------------------
# Shape of the merged intermediate
# --------------------------------------------------------------------------


def test_merged_document_has_the_documented_shape():
    merged = merged_from([RESULTS / "gemm_eigen_accelerate.json"])
    assert merged["kind"] == "eigen-benchmark-comparison-merged"
    assert merged["schema_version"] == "1.0.0"
    for key in ("configs", "arms", "cells", "coverage", "conflicts", "baseline", "reducer_version"):
        assert key in merged, key
    assert merged["baseline"] == "accelerate"
    assert set(merged["arms"]) == {"eigen", "accelerate"}
    assert list(merged["configs"]) == [support.CONFIG_ID], merged["configs"]


def test_every_cell_carries_its_dimension_names_and_size_key():
    merged = merged_from([RESULTS / "gemm_eigen_accelerate.json"])
    for cell in merged["cells"]:
        assert cell["shape_dims"] == ["m", "n", "k"], (
            "shape_dims is what lets a consumer read the shape without relying on JSON key order"
        )
        assert set(cell["shape"]) == set(cell["shape_dims"])
        assert cell["size_key"] == support.size_key(cell["shape"])


def test_arm_versions_survive_the_merge():
    merged = merged_from([RESULTS / "gemm_eigen_accelerate.json"])
    for arm, record in merged["arms"].items():
        assert record["library_version"], f"{arm}: a rate without a library version cannot be defended"


# --------------------------------------------------------------------------
# The single division by 1e9
# --------------------------------------------------------------------------


def test_flop_rate_is_divided_by_1e9_exactly_once():
    raw = support.read_json(RESULTS / "gemm_eigen_accelerate.json")
    merged = merged_from([RESULTS / "gemm_eigen_accelerate.json"])
    expected = {}
    for measurement in raw["measurements"]:
        key = (measurement["arm"], tuple(sorted(measurement["shape"].items())))
        expected[key] = measurement["stats"]["flop_rate"]["median"] / 1e9
    for cell in merged["cells"]:
        for arm, entry in cell["arms"].items():
            if entry["state"] != "measured":
                continue
            key = (arm, tuple(sorted(cell["shape"].items())))
            assert entry["gflops"] == pytest.approx(expected[key], rel=1e-12), (
                f"{arm} {cell['shape']}: {entry['gflops']} is not flop_rate/1e9"
            )


def test_merged_rates_are_physically_plausible():
    """A double division shows up as micro-GFLOP/s; a missing one as gigaGFLOP/s."""
    merged = merged_from([RESULTS / "gemm_eigen_accelerate.json"])
    for cell in merged["cells"]:
        for entry in cell["arms"].values():
            if entry["state"] == "measured":
                assert 1e-3 < entry["gflops"] < 1e6, entry["gflops"]


def test_time_stays_in_seconds():
    merged = merged_from([RESULTS / "gemm_eigen_accelerate.json"])
    for cell in merged["cells"]:
        for entry in cell["arms"].values():
            if entry["state"] == "measured":
                assert entry["time_s"] == pytest.approx(
                    cell["flops_per_iteration"] / (entry["gflops"] * 1e9), rel=1e-9
                )


def test_a_differing_source_time_unit_does_not_rescale_anything():
    """`source_time_unit` is informational; every time in a result file is
    already seconds. Re-applying it is a 1000x error that looks plausible."""
    merged = merged_from([RESULTS / "gemm_time_unit_ns.json", RESULTS / "gemm_time_unit_us.json"])
    by_config = {}
    for cell in merged["cells"]:
        if cell["shape"] == {"m": 64, "n": 64, "k": 64}:
            by_config[cell["config_id"]] = cell["arms"]["eigen"]["gflops"]
    assert len(by_config) == 2, f"expected one cell per machine, got {by_config}"
    for config_id, gflops in by_config.items():
        assert gflops == pytest.approx(45.0), f"{config_id}: {gflops}"


# --------------------------------------------------------------------------
# Ratios and inconclusiveness
# --------------------------------------------------------------------------


def test_ratio_greater_than_one_means_eigen_is_faster():
    merged = merged_from([RESULTS / "gemm_eigen_accelerate.json"])
    faster = cell_of(merged, shape={"m": 24, "n": 24, "k": 24})
    assert faster["ratio"] == pytest.approx(8.0 / 5.0)
    assert faster["ratio"] > 1.0
    slower = cell_of(merged, shape={"m": 64, "n": 64, "k": 64})
    assert slower["ratio"] == pytest.approx(45.0 / 60.0)
    assert slower["ratio"] < 1.0


def test_a_difference_inside_the_dispersion_is_inconclusive():
    """`.agents/benchmarking.md` rule 5: a change smaller than run-to-run
    variation is not a win or a regression."""
    merged = merged_from([RESULTS / "gemm_eigen_accelerate.json"])
    overlapping = cell_of(merged, shape={"m": 128, "n": 128, "k": 128})
    assert overlapping["ratio_state"] == "inconclusive", (
        f"96 +/- 1 against 95 +/- 1 overlaps; got {overlapping['ratio_state']}"
    )
    separated = cell_of(merged, shape={"m": 24, "n": 24, "k": 24})
    assert separated["ratio_state"] == "ok"


def test_the_inconclusive_rule_can_be_switched_off():
    merged = merged_from([RESULTS / "gemm_eigen_accelerate.json"], ["--inconclusive-rule", "none"])
    assert all(cell["ratio_state"] != "inconclusive" for cell in merged["cells"])


def test_the_inconclusive_rule_survives_a_merge(tmp_path):
    """`--merge` re-finalises every cell so that a folded-in contribution is
    rated against the baseline actually chosen. That re-rating has to apply the
    rule the caller asked for; hard-coding `mad-overlap` there made the same
    dataset come out `inconclusive` through `--merge` and `ok` through a
    single-shot reduce of the same files."""
    base = tmp_path / "base.json"
    proc = reduce_to([str(RESULTS / "gemm_eigen_accelerate.json"), "--inconclusive-rule", "none", "--out", str(base)])
    assert proc.returncode == 0, proc.stderr

    merged = merged_from(
        [RESULTS / "gemm_eigen_accelerate.json"],
        ["--inconclusive-rule", "none", "--merge", str(base)],
    )
    assert all(cell["ratio_state"] != "inconclusive" for cell in merged["cells"])


def test_a_single_repetition_run_reports_no_dispersion_and_does_not_crash():
    merged = merged_from([RESULTS / "gemm_single_repetition.json"])
    cell = cell_of(merged, shape={"m": 128, "n": 128, "k": 128})
    for arm, entry in cell["arms"].items():
        assert entry["state"] == "measured", arm
        assert entry["reps"] == 1
        assert entry["gflops_mad"] == 0.0, "one sample yields no dispersion; it must be 0, not invented"
    assert cell["ratio"] == pytest.approx(96.0 / 95.0)
    assert cell["ratio_state"] == "ok", (
        "with zero dispersion the intervals are points and do not overlap; calling it inconclusive "
        "would make every single-repetition run unusable"
    )


# --------------------------------------------------------------------------
# Partial data
# --------------------------------------------------------------------------


def test_a_partial_reference_arm_leaves_stated_negatives_never_zeros():
    merged = merged_from([RESULTS / "gemm_partial_shapes.json"])
    states = {}
    for cell in merged["cells"]:
        states[tuple(sorted(cell["shape"].items()))] = cell["arms"]["accelerate"]["state"]
    assert len(states) == 6, "the whole `small` group is in scope"
    assert sorted(states.values()).count("not_measured") == 4
    for cell in merged["cells"]:
        entry = cell["arms"]["accelerate"]
        if entry["state"] == "not_measured":
            assert "gflops" not in entry or entry["gflops"] is None, "a missing cell may never carry a number"
            assert entry["reason"], "a negative without a reason is indistinguishable from a bug"
            assert cell["ratio"] is None
            assert cell["ratio_state"] == "not_measured"


def test_a_missing_reference_library_is_stated_for_the_whole_grid():
    merged = merged_from([RESULTS / "gemm_reference_unavailable.json"])
    assert merged["cells"], "the eigen arm was measured; the cells must still exist"
    for cell in merged["cells"]:
        entry = cell["arms"].get("accelerate")
        assert entry is not None, "an arm in scope may not be dropped from the cell"
        assert entry["state"] == "not_measured"
        assert entry["reason"] == "reference_library_unavailable"
        assert cell["arms"]["eigen"]["state"] == "measured"


def test_an_eigen_only_op_says_the_comparison_cannot_exist(ops, tmp_path):
    """`no_reference_equivalent` is a permanent fact about the APIs, and reads
    differently from a measurement somebody forgot to make."""
    document = support.read_json(RESULTS / "minimal_valid.json")
    document["measurements"][0]["op"] = "FULLPIVLU"
    document["measurements"][0]["shape"] = {"m": 512, "n": 512}
    document["measurements"][0]["shape_dims"] = ["m", "n"]
    document["measurements"][0]["name"] = "FULLPIVLU/eigen/f64/m:512/n:512"
    document["scope"]["ops"] = ["FULLPIVLU"]
    document["scope"]["arms"] = ["eigen", "accelerate"]
    document["provenance"]["arms"]["accelerate"] = {
        "kind": "reference",
        "library_name": "Apple Accelerate",
        "library_version": "macOS 15.6",
    }
    document["not_measured"] = [
        {
            "op": "FULLPIVLU",
            "arm": "accelerate",
            "scalar": None,
            "shape": None,
            "reason": "no_reference_equivalent",
            "detail": ops["ops"]["FULLPIVLU"]["reference"]["reason"],
        }
    ]
    path = support.write_json(tmp_path / "fullpivlu.json", document)
    merged = merged_from([path])
    cell = cell_of(merged, op="FULLPIVLU", shape={"m": 512, "n": 512})
    assert cell["ratio"] is None
    assert cell["ratio_state"] == "no_reference_equivalent"
    assert cell["arms"]["accelerate"]["reason"] == "no_reference_equivalent"


def test_a_machine_that_measured_nothing_still_appears():
    merged = merged_from([RESULTS / "zero_ops_measured.json"])
    assert merged["configs"], "a machine with zero measurements is a fact about the store, not an absence"
    assert any(config["machine_config_id"] == "m4max" for config in merged["configs"].values())
    totals = merged["coverage"]["totals"]
    assert totals["measured"] == 0
    assert totals["not_measured"] > 0
    assert totals["unaccounted"] == 0, "everything in scope must be classified"


def test_coverage_totals_account_for_every_arm_cell():
    merged = merged_from([RESULTS / "gemm_partial_shapes.json"])
    counted = 0
    for cell in merged["cells"]:
        counted += len(cell["arms"])
    totals = merged["coverage"]["totals"]
    assert totals["measured"] + totals["not_measured"] + totals["unaccounted"] == counted, totals


# --------------------------------------------------------------------------
# Conflicts
# --------------------------------------------------------------------------


def test_duplicate_contributions_keep_the_newest_and_record_the_drop():
    merged = merged_from([RESULTS / "gemm_eigen_accelerate.json", RESULTS / "gemm_rerun_newer.json"])
    cell = cell_of(merged, shape={"m": 128, "n": 128, "k": 128})
    assert cell["arms"]["eigen"]["gflops"] == pytest.approx(96.0 * 1.10), "latest must win"
    assert merged["conflicts"], (
        "a config whose numbers move between runs is information about the machine; "
        "resolving the conflict silently throws it away"
    )
    conflict = next(
        c for c in merged["conflicts"] if c["shape"] == {"m": 128, "n": 128, "k": 128} and c["arm"] == "eigen"
    )
    assert conflict["policy"] == "latest"
    assert conflict["kept"]["gflops"] == pytest.approx(96.0 * 1.10)
    assert conflict["dropped"] and conflict["dropped"][0]["gflops"] == pytest.approx(96.0)


def test_input_order_does_not_decide_the_winner():
    forward = merged_from([RESULTS / "gemm_eigen_accelerate.json", RESULTS / "gemm_rerun_newer.json"])
    backward = merged_from([RESULTS / "gemm_rerun_newer.json", RESULTS / "gemm_eigen_accelerate.json"])
    shape = {"m": 128, "n": 128, "k": 128}
    assert cell_of(forward, shape=shape)["arms"]["eigen"]["gflops"] == pytest.approx(
        cell_of(backward, shape=shape)["arms"]["eigen"]["gflops"]
    ), "`latest` must mean the newest timestamp, not the last file on the command line"


def test_on_conflict_first_keeps_the_oldest():
    merged = merged_from(
        [RESULTS / "gemm_eigen_accelerate.json", RESULTS / "gemm_rerun_newer.json"], ["--on-conflict", "first"]
    )
    assert cell_of(merged, shape={"m": 128, "n": 128, "k": 128})["arms"]["eigen"]["gflops"] == pytest.approx(96.0)


def test_on_conflict_error_refuses_with_exit_three():
    proc = reduce_to(
        [str(RESULTS / "gemm_eigen_accelerate.json"), str(RESULTS / "gemm_rerun_newer.json"), "--on-conflict", "error"]
    )
    assert proc.returncode == 3, proc.stderr


def _restamped(path, tmp_path, name, run_id, timestamp):
    """A copy of a result fixture under a new run id and provenance timestamp."""
    document = json.loads(pathlib.Path(path).read_text())
    document["run_id"] = run_id
    document["provenance"]["timestamp_utc"] = timestamp
    target = tmp_path / name
    target.write_text(json.dumps(document))
    return target


def test_a_fractional_second_stamp_is_newer_than_the_whole_second_it_follows(tmp_path):
    """RFC 3339 fractional seconds are schema-legal and `'.' < 'Z'`, so ranking
    the stamps as strings discards a re-measurement in favour of the stale number
    it was taken to replace."""
    older = _restamped(
        RESULTS / "gemm_eigen_accelerate.json", tmp_path, "older.json",
        "older-run", "2026-08-01T12:00:00Z",
    )
    newer = _restamped(
        RESULTS / "gemm_rerun_newer.json", tmp_path, "newer.json",
        "newer-run", "2026-08-01T12:00:00.500Z",
    )
    merged = merged_from([older, newer])
    cell = cell_of(merged, shape={"m": 128, "n": 128, "k": 128})
    assert cell["arms"]["eigen"]["run_id"] == "newer-run", (
        "the half-second-newer re-run lost to the whole-second stamp it follows"
    )


def test_an_unparsable_timestamp_never_outranks_a_real_one(tmp_path):
    broken = _restamped(
        RESULTS / "gemm_rerun_newer.json", tmp_path, "broken.json", "broken-run", "not-a-timestamp"
    )
    stamped = _restamped(
        RESULTS / "gemm_eigen_accelerate.json", tmp_path, "stamped.json",
        "stamped-run", "2020-01-01T00:00:00Z",
    )
    merged = merged_from([stamped, broken], ["--no-validate"])
    cell = cell_of(merged, shape={"m": 128, "n": 128, "k": 128})
    assert cell["arms"]["eigen"]["run_id"] == "stamped-run", (
        "a stamp that cannot be parsed must not win the tie-break"
    )


def test_merging_a_second_vendor_refuses_an_ambiguous_baseline(tmp_path):
    """`--merge` must apply the same ambiguity rule as a single-shot reduce.
    Inheriting the base's reference arm gave every cell of the second vendor
    `ratio_state: "not_measured"` with both arms measured, under a column header
    naming the library that was not used."""
    base = tmp_path / "base.json"
    proc = reduce_to([str(RESULTS / "gemm_eigen_accelerate.json"), "--out", str(base)])
    assert proc.returncode == 0, proc.stderr

    second_vendor = json.loads((RESULTS / "gemm_eigen_accelerate.json").read_text())
    second_vendor["run_id"] = "second-vendor-run"
    for row in second_vendor["measurements"]:
        if row["arm"] == "accelerate":
            row["arm"] = "openblas"
            row["name"] = row["name"].replace("/accelerate/", "/openblas/")
    for row in second_vendor.get("not_measured", []):
        if row.get("arm") == "accelerate":
            row["arm"] = "openblas"
    arms = second_vendor["provenance"]["arms"]
    arms["openblas"] = arms.pop("accelerate")
    arms["openblas"]["library_name"] = "OpenBLAS"
    second_vendor["scope"]["arms"] = ["eigen", "openblas"]
    other = tmp_path / "openblas.json"
    other.write_text(json.dumps(second_vendor))

    ambiguous = reduce_to([str(other), "--merge", str(base)])
    assert ambiguous.returncode == 1, ambiguous.stdout[:2000]
    assert "--baseline" in ambiguous.stderr

    resolved = reduce_to([str(other), "--merge", str(base), "--baseline", "openblas"])
    assert resolved.returncode == 0, resolved.stderr
    merged = json.loads(resolved.stdout)
    assert merged["baseline"] == "openblas"
    for cell in merged["cells"]:
        arm_states = {arm: entry.get("state") for arm, entry in cell["arms"].items()}
        if arm_states.get("eigen") == "measured" and arm_states.get("openblas") == "measured":
            assert cell["ratio_state"] != "not_measured", cell["shape"]


def test_reducing_a_file_with_itself_is_not_a_conflict():
    merged = merged_from([RESULTS / "gemm_eigen_accelerate.json", RESULTS / "gemm_eigen_accelerate.json"])
    assert merged["conflicts"] == [], "identical numbers for one cell are not a disagreement"


# --------------------------------------------------------------------------
# Hostile input
# --------------------------------------------------------------------------


def test_a_name_that_contradicts_the_registry_is_a_hard_error_quoting_the_name():
    proc = reduce_to([str(RESULTS / "gemm_bad_dim_order.json")])
    assert proc.returncode != 0, (
        "a benchmark name whose dimension order contradicts ops.toml means the C++ and the registry "
        "have diverged; every number in the file is suspect and it may not be silently skipped"
    )
    assert "GEMM/eigen/f64/n:512/m:512/k:512" in (proc.stderr + proc.stdout), (
        f"the diagnostic must name the offending string. stderr was:\n{proc.stderr}"
    )


def test_an_op_absent_from_the_registry_is_an_input_error():
    proc = reduce_to([str(RESULTS / "unknown_op.json")])
    assert proc.returncode == 2, proc.stderr
    assert "NOSUCHOP" in proc.stderr


def test_a_nan_never_reaches_the_merged_output():
    proc = reduce_to([str(RESULTS / "gemm_nan_row.json")])
    if proc.returncode == 0:
        bad = support.find_nonfinite(json.loads(proc.stdout))
        assert not bad, (
            "a NaN rendered as a number is worse than a stated gap; it becomes a table cell nobody "
            f"questions. Non-finite values at: {bad}"
        )
    else:
        assert proc.returncode in (2, 3), proc.stderr
        assert proc.stderr.strip(), "rejecting the file is fine, but it has to say why"


def test_a_schema_invalid_input_is_refused(tmp_path):
    # reduce.py deliberately continues with "jsonschema is unavailable; inputs
    # were not validated" when the optional package is missing, so this case is
    # about the validator, not about the reducer.
    pytest.importorskip("jsonschema")
    broken = support.read_json(RESULTS / "minimal_valid.json")
    del broken["provenance"]["arms"]["eigen"]["library_version"]
    path = support.write_json(tmp_path / "broken.json", broken)
    proc = reduce_to([str(path)])
    assert proc.returncode == 2, proc.stderr


def test_skip_invalid_continues_with_a_warning(tmp_path):
    broken = support.read_json(RESULTS / "minimal_valid.json")
    del broken["provenance"]["arms"]["eigen"]["library_version"]
    path = support.write_json(tmp_path / "broken.json", broken)
    proc = reduce_to([str(path), str(RESULTS / "gemm_eigen_accelerate.json"), "--skip-invalid"])
    assert proc.returncode == 0, proc.stderr
    assert proc.stderr.strip(), "skipping an input silently is how a partial dataset becomes a wrong one"
    assert json.loads(proc.stdout)["cells"]


def test_no_input_at_all_is_exit_four():
    proc = reduce_to([], stdin="")
    assert proc.returncode == 4, proc.stderr


def test_an_ambiguous_baseline_is_refused(tmp_path):
    second = support.read_json(RESULTS / "gemm_eigen_accelerate.json")
    second["run_id"] = "m4pro-neon-openblas-e2a2fda17-20260801T130000Z"
    arms = second["provenance"]["arms"]
    arms["openblas"] = {"kind": "reference", "library_name": "OpenBLAS", "library_version": "0.3.29"}
    del arms["accelerate"]
    second["scope"]["arms"] = ["eigen", "openblas"]
    for measurement in second["measurements"]:
        if measurement["arm"] == "accelerate":
            measurement["arm"] = "openblas"
            measurement["name"] = measurement["name"].replace("/accelerate/", "/openblas/")
    path = support.write_json(tmp_path / "openblas.json", second)
    proc = reduce_to([str(RESULTS / "gemm_eigen_accelerate.json"), str(path)])
    assert proc.returncode != 0, "two reference arms make 'the baseline' a guess; it must be asked for"
    ok = merged_from([RESULTS / "gemm_eigen_accelerate.json", path], ["--baseline", "openblas"])
    assert ok["baseline"] == "openblas"


def test_run_notes_reach_the_configuration_and_are_unioned(tmp_path):
    """A caveat recorded by one contributing run describes the merged set.

    `provenance.run.notes` carries what a run established and proceeded under
    anyway -- the operator's own --note, an unverified machine profile, a load
    average above what the profile calls quiet. Before this it stopped at the
    result file: reduce.py dropped it, so nothing the operator wrote about a run
    could ever appear beside its numbers.

    Unioned rather than first-wins, for the same reason eigen_dirty is: two runs
    of one configuration where only one was noisy must not lose the caveat
    according to which filename sorts first.
    """
    first = support.read_json(RESULTS / "gemm_eigen_accelerate.json")
    first["provenance"]["run"]["notes"] = "measured during a thunderstorm"
    second = support.read_json(RESULTS / "gemm_eigen_accelerate.json")
    second["run_id"] = second["run_id"].replace("2026", "2027")
    second["provenance"]["run"]["notes"] = "measured with the lid closed"

    merged = merged_from(
        [
            support.write_json(tmp_path / "a.json", first),
            support.write_json(tmp_path / "b.json", second),
        ],
        ["--on-conflict", "keep-all"],
    )
    notes = [n for config in merged["configs"].values() for n in config.get("notes", [])]
    assert "measured during a thunderstorm" in notes
    assert "measured with the lid closed" in notes, (
        "the second run's caveat was dropped; a merged configuration must carry every "
        "contributing run's stated conditions"
    )


def test_a_run_without_notes_carries_an_empty_list_not_a_null():
    merged = merged_from([RESULTS / "gemm_eigen_accelerate.json"])
    for config in merged["configs"].values():
        assert config.get("notes") == [], "absence of caveats must render as no caveats, not as unknown"


def test_the_eigen_arm_is_refused_as_a_baseline():
    """A page of x1.00 is not a result.

    Ratios are eigen/baseline, so naming the Eigen arm takes every ratio against
    itself: exactly 1.0 everywhere, every cell flagged "inconclusive" because an
    interval always overlaps itself, and the ratio column headed "Eigen vs
    Eigen".  Nothing in that page reads as broken -- it reads as a real
    measurement showing no difference, which is the worst way for this harness
    to be wrong.  Auto-selection has always discarded the Eigen arm; only an
    explicit --baseline could reach it.
    """
    proc = reduce_to([str(RESULTS / "gemm_eigen_accelerate.json"), "--baseline", "eigen"])
    assert proc.returncode != 0, "comparing the Eigen arm against itself must not render a page"
    assert "eigen" in proc.stderr and "itself" in proc.stderr
    # The message has to name what to pass instead, or the user's next move is a guess.
    assert "accelerate" in proc.stderr


def test_a_baseline_no_result_file_carries_is_refused():
    """Otherwise a typo renders the whole page as 'not measured'.

    Every cell would be missing its baseline arm, so every ratio is undefined
    and the coverage manifest reports a dataset that was in fact measured as
    absent.  A dataset genuinely missing one arm on some configs is a different
    thing and stays allowed; this is the case where no file mentions the arm at
    all.
    """
    # A near-miss of the real arm name, spelled by transposition rather than by
    # dropping a letter: `codespell` runs over this tree and would flag the
    # obvious misspelling as a typo to correct, which would silently turn this
    # into a test of a name that is present.
    typo = "acceelrate"
    proc = reduce_to([str(RESULTS / "gemm_eigen_accelerate.json"), "--baseline", typo])
    assert proc.returncode != 0, "a misspelt baseline must not silently blank the page"
    assert typo in proc.stderr, "the message must quote what was asked for"
    assert "accelerate" in proc.stderr, "and name the arm that is actually present"


# --------------------------------------------------------------------------
# Input plumbing and determinism
# --------------------------------------------------------------------------


def test_paths_can_arrive_on_stdin():
    paths = "\n".join(
        str(RESULTS / name) for name in ("gemm_eigen_accelerate.json", "gemm_single_repetition.json")
    )
    proc = reduce_to([], stdin=paths + "\n")
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["cells"]


def test_glob_expansion_finds_the_same_files():
    proc = reduce_to(["--glob", str(RESULTS / "gemm_time_unit_*.json")])
    assert proc.returncode == 0, proc.stderr
    merged = json.loads(proc.stdout)
    assert len(merged["configs"]) == 2, "the two unit fixtures were measured on two machines"
    assert len(merged["cells"]) == 12


def test_two_runs_over_the_same_inputs_agree_byte_for_byte():
    first = reduce_to([str(RESULTS / "gemm_eigen_accelerate.json")])
    second = reduce_to([str(RESULTS / "gemm_eigen_accelerate.json")])
    assert first.returncode == second.returncode == 0
    strip = lambda text: "\n".join(l for l in text.splitlines() if "generated_utc" not in l)
    assert strip(first.stdout) == strip(second.stdout), "the reducer is not deterministic"


def test_pretty_output_is_sorted_so_a_store_diffs_cleanly():
    proc = reduce_to([str(RESULTS / "gemm_eigen_accelerate.json")])
    text = proc.stdout
    assert "\n  " in text, "--pretty must indent"
    document = json.loads(text)
    dumped = json.dumps(document, indent=2, sort_keys=True)
    assert text.strip() == dumped.strip(), "keys must be sorted, or every merge produces a noisy diff"


def test_writing_to_a_file_matches_writing_to_stdout(tmp_path):
    out = tmp_path / "merged.json"
    proc = reduce_to([str(RESULTS / "gemm_eigen_accelerate.json"), "--out", str(out)])
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "", "machine-readable output belongs on stdout only when --out is '-'"
    assert json.loads(out.read_text())["cells"]


def test_help_exits_zero():
    proc = support.run_cli("reduce.py", ["--help"])
    assert proc.returncode == 0 and proc.stdout.strip()


# --------------------------------------------------------------------------
# The registry a result was measured against
# --------------------------------------------------------------------------


@pytest.fixture
def result_from_another_registry(tmp_path):
    """A perfectly valid result that names a different ops.toml.

    Everything about it validates: the rows are well formed, the scope is
    coherent, the names parse.  What is wrong is invisible in the file -- the
    rates in it were divided by a flop count this ops.toml no longer computes,
    and the coverage, shape groups and flops_per_iteration the reducer would
    attach to them all come from the current registry."""
    document = support.read_json(RESULTS / "gemm_eigen_accelerate.json")
    document["scope"]["ops_toml_sha256"] = "f" * 64
    return support.write_json(tmp_path / "other-registry.json", document)


def test_a_result_from_another_registry_is_refused(result_from_another_registry):
    proc = reduce_to([str(result_from_another_registry)])
    assert proc.returncode != 0, (
        "a result measured against a different ops.toml was reduced without comment; its rates keep the "
        "old flop formula while everything around them comes from the new grid"
    )
    assert "ffffffffffff" in proc.stderr, proc.stderr
    assert support.OPS_TOML_SHA256[:12] in proc.stderr, proc.stderr


def test_registry_drift_can_be_published_only_with_the_caveat_attached(result_from_another_registry):
    merged = merged_from([result_from_another_registry], extra=["--allow-registry-drift"])
    notes = [note for record in merged["configs"].values() for note in record["notes"]]
    assert any("ops.toml" in note for note in notes), (
        "--allow-registry-drift published the numbers without the discrepancy reaching the page: "
        f"{notes}"
    )


def test_a_result_that_states_no_registry_is_not_blocked(tmp_path):
    """`scope.ops_toml_sha256` is nullable in the schema -- a hand-built binary
    with no EIGEN_BENCH_OPS_TOML_SHA256 produces one.  Nothing can be reconciled,
    so nothing is claimed, and the contribution is not refused for a fact it
    never asserted."""
    document = support.read_json(RESULTS / "gemm_eigen_accelerate.json")
    document["scope"]["ops_toml_sha256"] = None
    path = support.write_json(tmp_path / "unstated.json", document)
    merged = merged_from([path])
    assert merged["cells"]


def test_a_matching_registry_reduces_silently(tmp_path):
    proc = reduce_to([str(RESULTS / "gemm_eigen_accelerate.json")])
    assert proc.returncode == 0, proc.stderr
    assert "ops.toml" not in proc.stderr, (
        "the committed fixtures state the current registry; nothing should be reported about them"
    )

