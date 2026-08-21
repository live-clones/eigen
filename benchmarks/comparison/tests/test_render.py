# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

"""`render.py`: golden files, plus the properties a golden cannot express.

The golden files under `fixtures/golden/` are byte-compared.  Regenerate them
with exactly one command and review the diff:

    python3 benchmarks/comparison/tests/regenerate.py --goldens

A golden alone would let a renderer drift into printing a plausible number for a
combination nobody measured, so the tests after the golden section assert the
properties directly: a not-measured cell prints the token and never a zero, an
operation with no reference counterpart says so rather than leaving a blank, and
every footnote marker on the page resolves.
"""

import json
import re

import pytest

import harness_support as support

pytestmark = pytest.mark.impl

MERGED = support.FIXTURES / "merged" / "gemm_merged.json"
GOLDEN = support.FIXTURES / "golden"

FORMATS = [
    ("doxygen", "comparison_tables.dox"),
    ("markdown", "comparison_tables.md"),
]


def render(args, expect=0, **kwargs):
    proc = support.run_cli("render.py", args, **kwargs)
    assert proc.returncode == expect, f"render.py {args} exited {proc.returncode}\n{proc.stderr[-3000:]}"
    return proc


def rendered(fmt, extra=(), source=MERGED):
    return render(["--format", fmt, "--out", "-", *extra, str(source)]).stdout


# --------------------------------------------------------------------------
# Golden files
# --------------------------------------------------------------------------


@pytest.mark.golden
@pytest.mark.parametrize("fmt,filename", FORMATS)
def test_golden_table(fmt, filename, regen_golden):
    text = rendered(fmt)
    path = GOLDEN / filename
    if regen_golden:
        path.write_text(text)
        pytest.skip(f"regenerated {path}")
    assert path.is_file(), f"{path} is missing; run `python3 tests/regenerate.py --goldens`"
    assert text == path.read_text(), (
        f"{filename} changed. If the change is intended, run "
        f"`python3 benchmarks/comparison/tests/regenerate.py --goldens` and review the diff."
    )


@pytest.mark.golden
def test_golden_coverage_manifest(tmp_path, regen_golden):
    render(["--format", "coverage", "--out-dir", str(tmp_path), str(MERGED)])
    for filename in ("coverage.md", "coverage.json"):
        produced = (tmp_path / filename).read_text()
        path = GOLDEN / filename
        if regen_golden:
            path.write_text(produced)
            continue
        assert path.is_file(), f"{path} is missing; run `python3 tests/regenerate.py --goldens`"
        assert produced == path.read_text(), (
            f"{filename} changed; regenerate with `tests/regenerate.py --goldens` and review the diff"
        )
    if regen_golden:
        pytest.skip("regenerated the coverage goldens")


@pytest.mark.golden
def test_rendering_is_deterministic():
    for fmt, _ in FORMATS:
        assert rendered(fmt) == rendered(fmt), f"{fmt} output is not reproducible"


# --------------------------------------------------------------------------
# Nothing is ever rendered as zero
# --------------------------------------------------------------------------


def table_rows(markdown):
    rows = []
    for line in markdown.splitlines():
        if not line.startswith("|") or set(line.replace("|", "").replace(":", "").strip()) <= {"-", " "}:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        rows.append(cells)
    return rows


@pytest.mark.parametrize("fmt,_filename", FORMATS)
def test_a_not_measured_cell_prints_the_token_not_a_number(fmt, _filename, merged_gemm):
    text = rendered(fmt)
    missing = [
        cell
        for cell in merged_gemm["cells"]
        for entry in cell["arms"].values()
        if entry["state"] == "not_measured"
    ]
    assert missing, "the fixture must contain a not-measured cell"
    assert "n/a" in text or "—" in text
    for row in table_rows(text):
        for cell in row:
            assert not re.fullmatch(r"0(\.0+)?", cell), f"{fmt}: a cell rendered as zero: {row}"


def test_the_not_measured_token_is_configurable():
    text = rendered("markdown", ["--not-measured-token", "NOT-RUN"])
    assert "NOT-RUN" in text
    assert "| n/a " not in text


def test_an_operation_with_no_reference_counterpart_says_so(ops):
    """`no_reference_equivalent` is a permanent fact about the APIs. A blank
    there reads as a measurement somebody forgot to make."""
    text = rendered("markdown")
    reason = ops["ops"]["FULLPIVLU"]["reference"]["reason"]
    assert reason in text, "the ops.toml reason must be rendered verbatim"
    row = next(row for row in table_rows(text) if "FullPivLU" in row[0])
    assert "—" in " ".join(row), f"expected an em dash for the absent reference arm: {row}"


def test_every_footnote_marker_resolves():
    for fmt, _ in FORMATS:
        text = rendered(fmt)
        used = set(re.findall(r"\[([a-z]{1,2})\]", text))
        defined = set(re.findall(r"\*\*\[([a-z]{1,2})\]\*\*|<b>\[([a-z]{1,2})\]</b>", text))
        defined = {a or b for a, b in defined}
        assert used, f"{fmt}: the fixture has footnote-worthy cells but no markers were emitted"
        assert used <= defined, f"{fmt}: markers with no definition: {sorted(used - defined)}"


def test_an_inconclusive_ratio_is_labelled_not_coloured(merged_gemm):
    inconclusive = [c for c in merged_gemm["cells"] if c["ratio_state"] == "inconclusive"]
    assert inconclusive, "the fixture must contain an inconclusive cell"
    text = rendered("markdown")
    assert "inconclusive" in text.lower()


def test_a_nominal_flop_count_is_labelled(merged_gemm):
    nominal = [c for c in merged_gemm["cells"] if c["flops_nominal"]]
    assert nominal, "the fixture must contain a nominal-flops op"
    text = rendered("markdown")
    assert "convention" in text.lower() or "nominal" in text.lower()


def test_every_measured_number_in_the_table_comes_from_the_data(merged_gemm):
    """No cell may carry a rate that is not in the merged document."""
    expected = set()
    for cell in merged_gemm["cells"]:
        for entry in cell["arms"].values():
            if entry["state"] == "measured":
                expected.add(f"{entry['gflops']:.2f}")
    text = rendered("markdown")
    found = set(re.findall(r"\| (\d+\.\d\d) \|", text))
    assert found, "no rates were rendered at all"
    assert found <= expected, f"rates in the table that are not in the data: {sorted(found - expected)}"


# --------------------------------------------------------------------------
# Coverage manifest
# --------------------------------------------------------------------------


def test_coverage_json_accounts_for_every_arm_cell(tmp_path, merged_gemm):
    render(["--format", "coverage", "--out-dir", str(tmp_path), str(MERGED)])
    coverage = json.loads((tmp_path / "coverage.json").read_text())
    arm_cells = sum(len(cell["arms"]) for cell in merged_gemm["cells"])
    totals = coverage["totals"]
    assert totals["measured"] + totals["not_measured"] + totals["unaccounted"] == arm_cells, totals
    assert totals["unaccounted"] == 0


def test_coverage_lists_every_negative_with_its_reason(tmp_path, merged_gemm):
    render(["--format", "coverage", "--out-dir", str(tmp_path), str(MERGED)])
    coverage = json.loads((tmp_path / "coverage.json").read_text())
    reasons = {entry["reason"] for entry in coverage["not_measured"]}
    expected = {
        entry["reason"]
        for cell in merged_gemm["cells"]
        for entry in cell["arms"].values()
        if entry["state"] == "not_measured"
    }
    assert expected <= reasons, f"missing from the manifest: {sorted(expected - reasons)}"
    markdown = (tmp_path / "coverage.md").read_text()
    for reason in expected:
        assert reason in markdown, f"coverage.md does not mention {reason}"


def test_out_dir_writes_the_four_documented_filenames(tmp_path):
    render(["--out-dir", str(tmp_path), str(MERGED)])
    for filename in ("comparison_tables.dox", "comparison_tables.md", "coverage.md", "coverage.json"):
        assert (tmp_path / filename).is_file(), filename


# --------------------------------------------------------------------------
# Output shape
# --------------------------------------------------------------------------


def test_doxygen_output_uses_markdown_tables_not_raw_html():
    text = rendered("doxygen")
    for tag in ("<table", "<tr", "<td", "<th"):
        assert tag not in text.lower(), f"the Doxygen page contains a raw HTML table ({tag})"
    assert "\\eigenManualPage" in text


def test_markdown_table_has_one_row_per_cell(merged_gemm):
    rows = [row for row in table_rows(rendered("markdown")) if not row[0].startswith("BLAS")]
    assert len(rows) == len(merged_gemm["cells"]), (
        f"{len(rows)} rendered rows for {len(merged_gemm['cells'])} cells; a dropped cell is invisible"
    )


def test_no_forbidden_characters_leak_from_a_benchmark_name():
    text = rendered("markdown")
    assert "BM_" not in text, "an internal benchmark function name reached the published table"


# --------------------------------------------------------------------------
# Selection and metrics
# --------------------------------------------------------------------------


def test_op_selection_narrows_the_page():
    text = rendered("markdown", ["--op", "GEMM"])
    assert "GEMM" in text
    assert "FullPivLU" not in text


def test_scalar_selection_narrows_the_page():
    assert rendered("markdown", ["--scalar", "f64"]).count("|") > 0
    empty = rendered("markdown", ["--scalar", "c32"])
    assert "24" not in empty or "GEMM (zgemm)" not in empty


def test_config_selection_narrows_the_page():
    text = rendered("markdown", ["--config", support.CONFIG_ID])
    assert support.CONFIG_ID in text or "Apple M4 Pro" in text


@pytest.mark.parametrize("metric", ["gflops", "time", "ratio"])
def test_each_metric_renders(metric):
    text = rendered("markdown", ["--metric", metric])
    assert table_rows(text), metric


def test_merged_input_can_arrive_on_stdin():
    proc = render(["--format", "markdown", "--out", "-"], stdin=MERGED.read_text())
    assert "GEMM" in proc.stdout


# --------------------------------------------------------------------------
# Failure modes
# --------------------------------------------------------------------------


def test_out_and_out_dir_are_mutually_exclusive(tmp_path):
    proc = support.run_cli("render.py", ["--out", "-", "--out-dir", str(tmp_path), str(MERGED)])
    assert proc.returncode == 1, proc.stderr


def test_neither_out_nor_out_dir_is_a_usage_error():
    proc = support.run_cli("render.py", [str(MERGED)])
    assert proc.returncode == 1, proc.stderr


def test_out_with_several_formats_is_a_usage_error(tmp_path):
    proc = support.run_cli("render.py", ["--format", "markdown", "--format", "doxygen", "--out", "-", str(MERGED)])
    assert proc.returncode == 1, proc.stderr


def test_unparseable_input_is_exit_two(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json")
    proc = support.run_cli("render.py", ["--format", "markdown", "--out", "-", str(path)])
    assert proc.returncode == 2, proc.stderr


def test_an_op_absent_from_the_registry_is_exit_two(tmp_path, merged_gemm):
    document = support.deep_copy(merged_gemm)
    document["cells"][0]["op"] = "NOSUCHOP"
    path = support.write_json(tmp_path / "unknown.json", document)
    proc = support.run_cli("render.py", ["--format", "markdown", "--out", "-", str(path)])
    assert proc.returncode == 2, proc.stderr
    assert "NOSUCHOP" in proc.stderr


def test_a_dirty_eigen_worktree_is_marked(tmp_path, merged_gemm):
    document = support.deep_copy(merged_gemm)
    for config in document["configs"].values():
        config["eigen_dirty"] = True
    path = support.write_json(tmp_path / "dirty.json", document)
    text = render(["--format", "markdown", "--out", "-", str(path)]).stdout
    assert "dirty" in text.lower(), (
        "a result measured against an uncommitted worktree is not reproducible and must say so"
    )


def test_help_exits_zero():
    proc = support.run_cli("render.py", ["--help"])
    assert proc.returncode == 0 and proc.stdout.strip()
