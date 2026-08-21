# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

"""Incrementality, asserted in both directions.

This is the property the whole harness is built around: a result store grows one
contribution at a time, from different people on different machines, and it has
to be publishable at every point in between.

(a) A dataset with ONE cell in it renders, and every combination that is not in
    it is stated as not measured -- never dropped, never zero.
(b) Adding a second contribution changes only its own rows.  If merging a new
    machine's numbers perturbs an existing row, every published table has to be
    re-reviewed, which in practice means the store stops being merged at all.
"""

import json

import pytest

import harness_support as support

pytestmark = pytest.mark.impl

RESULTS = support.FIXTURES / "results"
MERGED = support.FIXTURES / "merged"


def reduce_files(paths, extra=()):
    proc = support.run_cli("reduce.py", [*[str(p) for p in paths], *extra])
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def render_markdown(path, extra=()):
    proc = support.run_cli("render.py", ["--format", "markdown", "--out", "-", *extra, str(path)])
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def data_rows(markdown):
    """Table rows only, keyed by their size column, ignoring header separators."""
    rows = {}
    for line in markdown.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if not cells or cells[0].startswith("BLAS") or set("".join(cells)) <= {"-", ":", " "}:
            continue
        rows[(cells[0], cells[2] if len(cells) > 2 else "")] = line
    return rows


@pytest.fixture
def one_cell_result(tmp_path):
    """A contribution whose scope is a whole shape group but which measured one
    point, with the rest stated explicitly."""
    document = support.read_json(RESULTS / "gemm_eigen_accelerate.json")
    keep = {"m": 24, "n": 24, "k": 24}
    document["run_id"] = "m4pro-neon-accelerate-e2a2fda17-20260820T090000Z"
    document["provenance"]["timestamp_utc"] = "2026-08-20T09:00:00Z"
    document["measurements"] = [m for m in document["measurements"] if m["shape"] == keep]
    assert len(document["measurements"]) == 2
    grid = support.family_points(support.load_ops(), "square3", ["small"])
    document["not_measured"] = [
        {
            "op": "GEMM",
            "arm": arm,
            "scalar": "f64",
            "shape": dict(zip(["m", "n", "k"], point)),
            "threads": 1,
            "reason": "excluded_by_filter",
            "detail": "this contribution measured a single point",
        }
        for point in grid
        if dict(zip(["m", "n", "k"], point)) != keep
        for arm in ("eigen", "accelerate")
    ]
    return support.write_json(tmp_path / "one_cell_result.json", document)


# --------------------------------------------------------------------------
# (a) One cell is a publishable dataset
# --------------------------------------------------------------------------


def test_a_single_merged_cell_renders():
    text = render_markdown(MERGED / "one_cell.json")
    assert "GEMM" in text
    rows = data_rows(text)
    assert len(rows) == 1, rows


def test_nothing_outside_a_one_cell_dataset_is_rendered_as_a_number(ops):
    """Every grid point that is not in the dataset must be absent or marked --
    a fabricated 0.00 would be indistinguishable from a real slow result."""
    text = render_markdown(MERGED / "one_cell.json")
    present = support.read_json(MERGED / "one_cell.json")["cells"][0]["shape"]
    for point in support.op_grid(ops, "GEMM"):
        shape = dict(zip(["m", "n", "k"], point))
        if shape == present:
            continue
        label = f"m={shape['m']} n={shape['n']} k={shape['k']}"
        for line in text.splitlines():
            if label in line:
                cells = [c.strip() for c in line.strip("|").split("|")]
                for cell in cells[3:]:
                    assert not cell.replace(".", "").isdigit(), f"fabricated number for {label}: {line}"


def test_a_one_point_contribution_marks_the_rest_of_its_scope(one_cell_result):
    merged = reduce_files([one_cell_result])
    measured = [c for c in merged["cells"] if c["arms"]["eigen"]["state"] == "measured"]
    assert len(measured) == 1, "the contribution measured one point"
    assert len(merged["cells"]) == 6, "the whole scope must survive as cells"
    for cell in merged["cells"]:
        if cell in measured:
            continue
        for arm, entry in cell["arms"].items():
            assert entry["state"] == "not_measured", (arm, cell["shape"])
            assert entry.get("reason"), (arm, cell["shape"])
            assert "gflops" not in entry or entry["gflops"] is None


def test_the_one_point_page_marks_every_other_row(one_cell_result, tmp_path):
    merged = support.write_json(tmp_path / "merged.json", reduce_files([one_cell_result]))
    text = render_markdown(merged)
    rows = data_rows(text)
    assert len(rows) == 6, rows
    assert text.count("n/a") >= 10, "five of six rows have both arms unmeasured"


def test_coverage_of_a_one_point_contribution_is_honest(one_cell_result, tmp_path):
    merged = support.write_json(tmp_path / "merged.json", reduce_files([one_cell_result]))
    proc = support.run_cli("render.py", ["--format", "coverage", "--out-dir", str(tmp_path), str(merged)])
    assert proc.returncode == 0, proc.stderr
    coverage = json.loads((tmp_path / "coverage.json").read_text())
    assert coverage["totals"]["measured"] == 2
    assert coverage["totals"]["not_measured"] == 10
    assert coverage["totals"]["unaccounted"] == 0


# --------------------------------------------------------------------------
# (b) A second contribution disturbs nothing
# --------------------------------------------------------------------------


@pytest.fixture
def second_contribution(tmp_path):
    """The same grid measured for a different scalar: disjoint cells, same config."""
    document = support.read_json(RESULTS / "gemm_eigen_accelerate.json")
    document["run_id"] = "m4pro-neon-accelerate-e2a2fda17-20260821T090000Z"
    document["provenance"]["timestamp_utc"] = "2026-08-21T09:00:00Z"
    document["scope"]["scalars"] = ["f32"]
    for measurement in document["measurements"]:
        measurement["scalar"] = "f32"
        measurement["name"] = measurement["name"].replace("/f64/", "/f32/")
        for stat in measurement["stats"].values():
            for field in ("median", "mean", "min", "max"):
                stat[field] = stat[field] * 1.7
    return support.write_json(tmp_path / "second.json", document)


def test_adding_a_contribution_leaves_existing_cells_identical(second_contribution):
    before = reduce_files([RESULTS / "gemm_eigen_accelerate.json"])
    after = reduce_files([RESULTS / "gemm_eigen_accelerate.json", second_contribution])

    def index(merged):
        return {
            (c["config_id"], c["op"], c["scalar"], tuple(sorted(c["shape"].items())), c.get("threads", 1)): c
            for c in merged["cells"]
        }

    before_cells, after_cells = index(before), index(after)
    assert set(before_cells) < set(after_cells), "the second contribution must add cells"
    for key, cell in before_cells.items():
        assert after_cells[key] == cell, (
            f"merging a new contribution changed an existing cell: {key}\n"
            f"  before: {json.dumps(cell, sort_keys=True)}\n"
            f"  after:  {json.dumps(after_cells[key], sort_keys=True)}"
        )


def test_adding_a_contribution_leaves_existing_rendered_rows_byte_identical(second_contribution, tmp_path):
    before_path = support.write_json(tmp_path / "before.json", reduce_files([RESULTS / "gemm_eigen_accelerate.json"]))
    after_path = support.write_json(
        tmp_path / "after.json", reduce_files([RESULTS / "gemm_eigen_accelerate.json", second_contribution])
    )
    before = data_rows(render_markdown(before_path, ["--scalar", "f64"]))
    after = data_rows(render_markdown(after_path, ["--scalar", "f64"]))
    assert before, "nothing was rendered before"
    for key, line in before.items():
        assert key in after, f"row {key} disappeared when a second contribution was added"
        assert after[key] == line, (
            f"row {key} changed when an unrelated contribution was added:\n  {line}\n  {after[key]}"
        )


def _clean_gemm_merged():
    """The merged fixture reduced to fully measured GEMM f64 cells.

    Fully measured and conclusive on purpose: such a row carries no footnote, so
    a byte comparison of it tests the row's own rendering rather than the
    page-wide footnote numbering, which is a separate coupling.
    """
    merged = support.read_json(MERGED / "gemm_merged.json")
    cells = []
    for cell in merged["cells"]:
        if cell["op"] != "GEMM" or cell["scalar"] != "f64":
            continue
        arms = cell.get("arms") or {}
        if {a.get("state") for a in arms.values()} != {"measured"}:
            continue
        if cell.get("ratio_state") != "ok":
            continue
        cells.append(cell)
    assert len(cells) >= 3, "the fixture must contain several clean GEMM rows"
    merged["cells"] = cells
    merged["ops"] = {k: v for k, v in (merged.get("ops") or {}).items() if k == "GEMM"}
    return merged


def _cell_for_a_second_shape_family(template):
    """A GETRF cell: two dimensions where GEMM has three, same config and scalar."""
    cell = json.loads(json.dumps(template))
    cell["op"] = "GETRF"
    cell["op_family"] = "lapack-factorization"
    cell["shape"] = {"m": 512, "n": 512}
    cell["shape_dims"] = ["m", "n"]
    cell["shape_group"] = "medium"
    cell["size_key"] = 512
    cell["flops_per_iteration"] = 512.0 * 512.0 * 512.0 * 2.0 / 3.0
    return cell


def test_an_op_from_another_shape_family_leaves_the_gemm_rows_byte_identical(tmp_path):
    """Property (b) for the size column. Deciding `m=64 n=64 k=64` versus
    `64x64x64` from the set of shape families present in the table made every
    pre-existing row and the table header depend on which OTHER ops had been
    merged, so adding a factorization re-rendered every GEMM row without a
    single number changing."""
    before = _clean_gemm_merged()
    after = json.loads(json.dumps(before))
    after["cells"].append(_cell_for_a_second_shape_family(before["cells"][0]))

    before_path = support.write_json(tmp_path / "before.json", before)
    after_path = support.write_json(tmp_path / "after.json", after)
    before_text = render_markdown(before_path)
    after_text = render_markdown(after_path)

    before_rows, after_rows = data_rows(before_text), data_rows(after_text)
    assert before_rows, "nothing was rendered before"
    assert len(after_rows) == len(before_rows) + 1, "the second op must add exactly one row"
    for key, line in before_rows.items():
        assert key in after_rows, f"row {key} disappeared when a second shape family was added"
        assert after_rows[key] == line, (
            f"row {key} was re-rendered because an unrelated op was merged:\n  {line}\n  {after_rows[key]}"
        )

    def header(text):
        return [line for line in text.splitlines() if line.startswith("| BLAS/LAPACK")]

    assert header(after_text) == header(before_text), "the table header changed"


def test_a_new_machine_does_not_disturb_the_old_one():
    before = reduce_files([RESULTS / "gemm_time_unit_ns.json"])
    after = reduce_files([RESULTS / "gemm_time_unit_ns.json", RESULTS / "gemm_time_unit_us.json"])
    old_config = list(before["configs"])[0]
    old_cells = sorted(
        (c for c in before["cells"] if c["config_id"] == old_config), key=lambda c: c["size_key"]
    )
    new_cells = sorted(
        (c for c in after["cells"] if c["config_id"] == old_config), key=lambda c: c["size_key"]
    )
    assert old_cells == new_cells, "adding a second machine perturbed the first machine's cells"
    assert before["configs"][old_config] == after["configs"][old_config]


def test_merge_into_an_existing_merged_file_is_additive(second_contribution, tmp_path):
    base = tmp_path / "base.json"
    proc = support.run_cli(
        "reduce.py", [str(RESULTS / "gemm_eigen_accelerate.json"), "--out", str(base)]
    )
    assert proc.returncode == 0, proc.stderr
    before = json.loads(base.read_text())
    proc = support.run_cli("reduce.py", [str(second_contribution), "--merge", str(base)])
    assert proc.returncode == 0, proc.stderr
    after = json.loads(proc.stdout)
    keyed = {
        (c["op"], c["scalar"], tuple(sorted(c["shape"].items()))): c for c in after["cells"]
    }
    for cell in before["cells"]:
        key = (cell["op"], cell["scalar"], tuple(sorted(cell["shape"].items())))
        assert keyed[key] == cell, f"--merge rewrote an existing cell: {key}"
    assert len(after["cells"]) > len(before["cells"])


@pytest.fixture
def caveat_carrying_contribution(tmp_path):
    """A second contribution to the SAME configuration that carries every caveat.

    `eigen_dirty` is deliberately not part of `config_id`, so a clean run and a
    dirty run of the same commit share a configuration and the caveats have to be
    unioned rather than resolved.  The scalar differs so the cells are disjoint
    and only the configuration record is under test."""
    document = support.read_json(RESULTS / "gemm_eigen_accelerate.json")
    document["run_id"] = "m4pro-neon-accelerate-e2a2fda17-20260822T090000Z"
    document["provenance"]["timestamp_utc"] = "2026-08-22T09:00:00Z"
    document["provenance"]["eigen"]["dirty"] = True
    document["provenance"].setdefault("run", {})["notes"] = "measured with an unverified machine profile"
    document["provenance_gaps"] = [
        {"field": "/provenance/cpu/turbo_enabled", "reason": "no way to read the turbo state on this host"}
    ]
    document["scope"]["scalars"] = ["f32"]
    for measurement in document["measurements"]:
        measurement["scalar"] = "f32"
        measurement["name"] = measurement["name"].replace("/f64/", "/f32/")
    return support.write_json(tmp_path / "dirty.json", document)


def _sole_config(merged):
    assert len(merged["configs"]) == 1, sorted(merged["configs"])
    return next(iter(merged["configs"].values()))


def test_merge_preserves_the_caveats_of_the_contribution_folded_in(caveat_carrying_contribution, tmp_path):
    """The two documented routes to one store must agree about what a
    configuration carries.

    `reduce.py a.json b.json` unions the caveats; `reduce.py b.json --merge
    a-merged.json` kept the base record whole and unioned only `provenance_refs`,
    so a dirty, noisy, gap-carrying contribution rendered as clean the moment it
    arrived incrementally.  A caveat that disappears according to how the store
    was assembled is worse than no caveat at all."""
    together = reduce_files([RESULTS / "gemm_eigen_accelerate.json", caveat_carrying_contribution])

    base = tmp_path / "base.json"
    proc = support.run_cli("reduce.py", [str(RESULTS / "gemm_eigen_accelerate.json"), "--out", str(base)])
    assert proc.returncode == 0, proc.stderr
    assert _sole_config(json.loads(base.read_text()))["eigen_dirty"] is False, "the base must start clean"
    incremental = support.cli_json("reduce.py", [str(caveat_carrying_contribution), "--merge", str(base)])

    left, right = _sole_config(together), _sole_config(incremental)
    assert right["eigen_dirty"] is True, (
        "a dirty contribution merged incrementally rendered as a clean, reproducible configuration"
    )
    assert right["notes"] == left["notes"] != [], "the incremental route dropped the run's notes"
    assert right["provenance_gaps"] == left["provenance_gaps"], "the incremental route dropped a provenance gap"
    assert sorted(right["provenance_refs"]) == sorted(left["provenance_refs"])


def test_merge_takes_arm_metadata_from_the_newer_document(caveat_carrying_contribution, tmp_path):
    """Library version is what a published column is headed with.  First-wins
    kept the version that happened to be in the file being merged INTO, so
    re-measuring against an upgraded vendor and merging the result went on
    printing the superseded version string."""
    document = support.read_json(caveat_carrying_contribution)
    document["provenance"]["arms"]["accelerate"]["library_version"] = "macOS 26.0"
    newer = support.write_json(tmp_path / "newer.json", document)

    base = tmp_path / "base.json"
    proc = support.run_cli("reduce.py", [str(RESULTS / "gemm_eigen_accelerate.json"), "--out", str(base)])
    assert proc.returncode == 0, proc.stderr
    merged = support.cli_json("reduce.py", [str(newer), "--merge", str(base)])
    assert merged["arms"]["accelerate"]["library_version"] == "macOS 26.0"


def test_reducing_the_same_contribution_twice_is_a_no_op():
    once = reduce_files([RESULTS / "gemm_eigen_accelerate.json"])
    twice = reduce_files([RESULTS / "gemm_eigen_accelerate.json", RESULTS / "gemm_eigen_accelerate.json"])
    once.pop("generated_utc"), twice.pop("generated_utc")
    assert once == twice, "re-submitting an identical contribution must not change the store"
