#!/usr/bin/env python3
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

"""Render the merged comparison intermediate as tables and a coverage manifest.

Three outputs, all driven from the same table model:

* a Doxygen page for ``doc/``,
* a website markdown page,
* the coverage manifest, human readable and machine readable.

The central requirement is correctness from a **partial** dataset.  Rows are
keyed by the BLAS/LAPACK mnemonic with the Eigen spelling as a secondary
column, because the audience navigates by BLAS/LAPACK naming.  A combination
that was not measured renders as an explicit token with a footnote, never as a
blank and never as zero; an operation with no reference counterpart says so,
rather than leaving a gap that would read as a missing measurement.

See ``benchmarks/comparison/CONTRACTS.md`` sections 4.3 and 6.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# This script is run directly and is also imported by file path by the test
# suite, so its own directory is not reliably on sys.path when _common is needed.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _common import (  # noqa: E402
    PipelineError,
    UsageErrorArgumentParser,
    arm_display,
    arms_in,
    compact_shape,
    config_parts,
    load_merged,
    load_ops_registry,
    select_cells,
    split_list,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OPS_TOML = os.path.join(HERE, "ops.toml")

DEFAULT_NOT_MEASURED_TOKEN = "n/a"

#: Rendered in the reference column of an operation that has no counterpart in
#: the reference API.  Kept as a named constant so a project that compares
#: against something other than LAPACK can swap the wording in one place.
NO_REFERENCE_TOKEN = "no LAPACK counterpart"

#: CONTRACTS.md section 6: for ``no_reference_equivalent`` the arm cell is an
#: em dash, so the table states that the comparison cannot exist.
NO_REFERENCE_MARK = "—"

DOXYGEN_PAGE_ID = "EigenBlasComparisonBenchmark"

DEFAULT_FILENAMES = {
    "doxygen": "comparison_tables.dox",
    "markdown": "comparison_tables.md",
    "coverage": ("coverage.md", "coverage.json"),
}

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_INPUT = 2


class RenderError(PipelineError):
    def __init__(self, message: str, code: int = EXIT_INPUT):
        super().__init__(message, code)


# ---------------------------------------------------------------------------
# Table model
# ---------------------------------------------------------------------------


@dataclass
class Cell:
    text: str
    note: Optional[str] = None
    code: bool = False
    muted: bool = False


@dataclass
class Table:
    title: str
    caption: str
    headers: List[str]
    rows: List[List[Cell]]
    anchor: str = "comparison"


@dataclass
class Page:
    title: str
    intro: List[str]
    tables: List[Table]
    footnotes: List[Tuple[str, str]]


@dataclass
class RenderOptions:
    configs: Optional[Sequence[str]] = None
    ops: Optional[Sequence[str]] = None
    scalars: Optional[Sequence[str]] = None
    baseline: Optional[str] = None
    metric: str = "gflops"
    not_measured_token: str = DEFAULT_NOT_MEASURED_TOKEN
    title: Optional[str] = None
    table_style: str = "markdown"


class Footnotes:
    """Page-scoped footnote registry: one marker per distinct note text."""

    def __init__(self) -> None:
        self._order: List[Tuple[str, str]] = []
        self._index: Dict[str, str] = {}

    def marker(self, text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        if text in self._index:
            return self._index[text]
        marker = _marker_at(len(self._order))
        self._index[text] = marker
        self._order.append((marker, text))
        return marker

    def items(self) -> List[Tuple[str, str]]:
        return list(self._order)


def _doxygen_anchor(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in text)
    return "comparison_" + cleaned.strip("_").lower()


def _marker_at(index: int) -> str:
    letters = "abcdefghijklmnopqrstuvwxyz"
    marker = ""
    index += 1
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        marker = letters[remainder] + marker
    return marker


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


def load_ops_toml(path: str) -> Dict[str, Any]:
    """ops.toml as a plain mapping.

    A registry this renderer cannot read degrades to ``{}``: everything it looks
    up there is a fallback for metadata the merged document normally embeds, so
    an absent registry costs display polish rather than a rendered page.
    """
    try:
        return dict(load_ops_registry(path).data)
    except PipelineError:
        return {}


def op_metadata(merged: Mapping[str, Any], registry: Mapping[str, Any], op: str) -> Dict[str, Any]:
    """Op metadata from the merged file, falling back to ops.toml."""
    embedded = (merged.get("ops") or {}).get(op)
    if embedded:
        return dict(embedded)
    entry = (registry.get("ops") or {}).get(op)
    if entry is None:
        raise RenderError(f"merged input names op {op!r}, absent from ops.toml")
    data = dict(entry)
    data.setdefault("display_name", op)
    return data


def baseline_of(merged: Mapping[str, Any], options: RenderOptions) -> Optional[str]:
    return options.baseline or merged.get("baseline")


def config_display(merged: Mapping[str, Any], config_id: str) -> str:
    """The caption naming a table's configuration; shared with plots.py."""
    return ", ".join(config_parts(merged, config_id, fallback=config_id))


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_rate(value: Optional[float]) -> str:
    if value is None:
        return DEFAULT_NOT_MEASURED_TOKEN
    if value >= 100.0:
        return f"{value:.1f}"
    if value >= 1.0:
        return f"{value:.2f}"
    return f"{value:.3g}"


def format_time(value: Optional[float]) -> str:
    if value is None:
        return DEFAULT_NOT_MEASURED_TOKEN
    return f"{value:.4g}"


def format_ratio(value: Optional[float]) -> str:
    if value is None:
        return DEFAULT_NOT_MEASURED_TOKEN
    return f"x{value:.2f}"


def format_shape(cell: Mapping[str, Any]) -> str:
    """A row's size cell, from that row alone: `m=64 n=64 k=64`.

    Unambiguous without a per-table header, which is what lets a table hold rows
    from several shape families without any of them being re-rendered.
    """
    return " ".join(f"{dim}={cell['shape'][dim]}" for dim in cell["shape_dims"])


def routine_name(meta: Mapping[str, Any], scalar: str, registry: Mapping[str, Any]) -> Optional[str]:
    """The typed BLAS/LAPACK routine, e.g. ?gemm + f64 -> dgemm."""
    reference = meta.get("reference") or {}
    if reference.get("kind") in (None, "none"):
        return None
    routine = reference.get("routine")
    if not routine:
        return None
    complex_routine = reference.get("complex_routine")
    scalars = registry.get("scalars") or {}
    prefix = (scalars.get(scalar) or {}).get("blas_prefix", "")
    is_complex = bool((scalars.get(scalar) or {}).get("is_complex", scalar in ("c32", "c64")))
    if is_complex and complex_routine:
        routine = complex_routine
    if not prefix:
        prefix = {"f32": "s", "f64": "d", "c32": "c", "c64": "z"}.get(scalar, "")
    return routine.replace("?", prefix.lower())


def eigen_spelling(meta: Mapping[str, Any]) -> str:
    return meta.get("eigen_expr") or meta.get("eigen_class") or meta.get("display_name") or ""


# ---------------------------------------------------------------------------
# Table construction
# ---------------------------------------------------------------------------


def build_table(
    merged: Mapping[str, Any],
    registry: Mapping[str, Any],
    config_id: str,
    scalar: str,
    cells: Sequence[Mapping[str, Any]],
    options: RenderOptions,
    notes: Footnotes,
) -> Table:
    baseline = baseline_of(merged, options)
    arms = arms_in(cells, baseline)
    metric = options.metric

    # Named dimensions, always, and a fixed header. Deciding the spelling from the
    # set of shape families present in the table made it a property of the table
    # rather than of the row: merging a contribution for an op with a different
    # shape family rewrote every pre-existing row and the header, byte for byte,
    # without a single number changing -- the exact opposite of the incrementality
    # property this renderer is documented to have.
    headers = ["BLAS/LAPACK", "Eigen", "Size"]
    if metric == "gflops":
        headers += [f"{arm_display(merged, arm)} GFLOP/s" for arm in arms]
    elif metric == "time":
        headers += [f"{arm_display(merged, arm)} time (s)" for arm in arms]
    if baseline:
        headers.append(f"Eigen vs {arm_display(merged, baseline)}")

    rows: List[List[Cell]] = []
    for cell in cells:
        meta = op_metadata(merged, registry, cell["op"])
        reference = meta.get("reference") or {}
        routine = routine_name(meta, scalar, registry)
        if routine:
            label = f"{meta.get('display_name', cell['op'])} ({routine})"
            label_note = None
        else:
            label = str(meta.get("display_name", cell["op"]))
            label_note = reference.get("reason") or "No reference counterpart exists for this operation."
        if cell.get("flops_nominal"):
            nominal_note = (
                f"The flop count for {meta.get('display_name', cell['op'])} is a convention, not a "
                "hardware-truthful operation count. It is applied identically to both arms, so the ratio is "
                "meaningful even though the absolute rate is not a machine-efficiency figure."
            )
            label_note = f"{label_note} {nominal_note}" if label_note else nominal_note

        row = [
            Cell(label, notes.marker(label_note)),
            Cell(eigen_spelling(meta), code=True),
            Cell(format_shape(cell)),
        ]
        if metric in ("gflops", "time"):
            for arm in arms:
                row.append(_arm_cell(cell, arm, reference, options, notes, metric))
        if baseline:
            row.append(_ratio_cell(cell, reference, options, notes))
        rows.append(row)

    caption_bits = [config_display(merged, config_id)]
    scalar_display = ((registry.get("scalars") or {}).get(scalar) or {}).get("display", scalar)
    caption_bits.append(f"scalar type {scalar_display}")
    return Table(
        title=f"{scalar_display} on {config_display(merged, config_id)}",
        caption="; ".join(caption_bits),
        headers=headers,
        rows=rows,
        anchor=_doxygen_anchor(f"{config_id}_{scalar}"),
    )


def _arm_cell(
    cell: Mapping[str, Any],
    arm: str,
    reference: Mapping[str, Any],
    options: RenderOptions,
    notes: Footnotes,
    metric: str,
) -> Cell:
    entry = (cell.get("arms") or {}).get(arm)
    if entry is None:
        return Cell(
            options.not_measured_token,
            notes.marker(f"No contribution recorded this arm for {cell['op']}."),
            muted=True,
        )
    if entry.get("state") == "measured":
        value = entry.get("gflops") if metric == "gflops" else entry.get("time_s")
        return Cell(format_rate(value) if metric == "gflops" else format_time(value))
    if entry.get("reason") == "no_reference_equivalent":
        text = reference.get("reason") or entry.get("detail") or "No reference counterpart exists."
        return Cell(NO_REFERENCE_MARK, notes.marker(text), muted=True)
    detail = entry.get("detail") or _reason_text(entry.get("reason"))
    return Cell(options.not_measured_token, notes.marker(detail), muted=True)


def _ratio_cell(
    cell: Mapping[str, Any],
    reference: Mapping[str, Any],
    options: RenderOptions,
    notes: Footnotes,
) -> Cell:
    state = cell.get("ratio_state")
    if state == "no_reference_equivalent":
        text = reference.get("reason") or "No reference counterpart exists for this operation."
        return Cell(NO_REFERENCE_TOKEN, notes.marker(text), muted=True)
    if state == "ok":
        return Cell(format_ratio(cell.get("ratio")))
    if state == "inconclusive":
        return Cell(
            f"{format_ratio(cell.get('ratio'))} (inconclusive)",
            notes.marker(
                "The two arms' [median - MAD, median + MAD] intervals overlap: the difference is smaller "
                "than the observed run-to-run variation and is neither a win nor a regression."
            ),
            muted=True,
        )
    return Cell(
        options.not_measured_token,
        notes.marker("At least one arm of this comparison was not measured; see the coverage manifest."),
        muted=True,
    )


_REASON_TEXT = {
    "not_implemented": "No benchmark source registers this operation yet.",
    "no_reference_equivalent": "No reference counterpart exists for this operation.",
    "reference_routine_absent": "The reference library exposes no such routine.",
    "reference_library_unavailable": "The reference library was not available in this build.",
    "scalar_unsupported": "This scalar type is not supported for this operation.",
    "shape_unsupported": "This shape is not supported for this operation.",
    "excluded_by_filter": "Excluded by the run's benchmark filter.",
    "build_failed": "The benchmark executable failed to build.",
    "runtime_error": "The benchmark failed at run time.",
    "timeout": "The benchmark exceeded its time budget.",
    "out_of_memory": "The benchmark ran out of memory.",
    "isa_unsupported": "The ISA target does not support this configuration.",
    "machine_unavailable": "The machine was not available for this run.",
    "unaccounted": "In scope but reported by neither measurements nor not_measured; this is a harness bug.",
}


def _reason_text(reason: Optional[str]) -> str:
    return _REASON_TEXT.get(reason or "", f"Not measured ({reason or 'reason not recorded'}).")


def build_page(merged: Mapping[str, Any], registry: Mapping[str, Any], options: RenderOptions) -> Page:
    cells = select_cells(merged, options, by_config=True)
    notes = Footnotes()
    tables: List[Table] = []
    groups: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for cell in cells:
        groups.setdefault((cell["config_id"], cell["scalar"]), []).append(cell)
    for (config_id, scalar) in sorted(groups):
        tables.append(build_table(merged, registry, config_id, scalar, groups[(config_id, scalar)], options, notes))

    baseline = baseline_of(merged, options)
    title = options.title or (
        f"Eigen versus {arm_display(merged, baseline)} on dense linear algebra"
        if baseline
        else "Eigen dense linear algebra benchmarks"
    )
    intro = [
        "Rows are keyed by the BLAS/LAPACK mnemonic, with the equivalent Eigen spelling beside it, so a reader "
        "who navigates by BLAS/LAPACK naming can find the Eigen expression that performs the same work.",
        "Every rate is GFLOP/s computed from the flop count in `ops.toml`, identical for both arms of a "
        "comparison. A ratio above 1 means Eigen is faster.",
        f"This dataset is partial by construction. A combination that was not measured is shown as "
        f"`{options.not_measured_token}` with a footnote giving the reason; an operation with no reference "
        f"counterpart is shown as `{NO_REFERENCE_MARK}` and named as such. Neither is ever rendered as zero. "
        f"The coverage manifest lists every such combination.",
    ]
    if merged.get("generated_utc"):
        intro.append(f"Merged {merged['generated_utc']} by reduce.py {merged.get('reducer_version', '')}.".strip())
    return Page(title=title, intro=intro, tables=tables, footnotes=notes.items())


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def _escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _escape_markdown_cell(text: str) -> str:
    return text.replace("|", "\\|")


def _plain_cell(cell: Cell) -> str:
    text = f"`{cell.text}`" if cell.code and cell.text else cell.text
    if cell.note:
        text = f"{text} [{cell.note}]"
    return _escape_markdown_cell(text)


def markdown_table(table: Table) -> str:
    lines = [f"### {table.title}", "", table.caption, ""]
    lines.append("| " + " | ".join(_escape_markdown_cell(h) for h in table.headers) + " |")
    lines.append("|" + "|".join("---" for _ in table.headers) + "|")
    for row in table.rows:
        lines.append("| " + " | ".join(_plain_cell(cell) for cell in row) + " |")
    lines.append("")
    return "\n".join(lines)


def render_markdown(page: Page) -> str:
    """Format an already-built page as markdown; see `build_page`."""
    out = [f"# {page.title}", ""]
    out.extend(paragraph + "\n" for paragraph in page.intro)
    if not page.tables:
        out.append("No cell in the merged dataset matched the selection.\n")
    for table in page.tables:
        out.append(markdown_table(table))
    if page.footnotes:
        out.append("## Notes\n")
        for marker, text in page.footnotes:
            out.append(f"- **[{marker}]** {text}")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def doxygen_markdown_table(table: Table) -> str:
    lines = [f"\\subsection {table.anchor} {_escape_html(table.title)}", "",
             _escape_html(table.caption), ""]
    lines.append("| " + " | ".join(_escape_markdown_cell(_escape_html(h)) for h in table.headers) + " |")
    lines.append("|" + "|".join(":---" for _ in table.headers) + "|")
    for row in table.rows:
        cells = []
        for cell in row:
            text = _escape_html(cell.text)
            if cell.code and cell.text:
                text = f"<code>{text}</code>"
            if cell.note:
                text = f"{text} [{cell.note}]"
            cells.append(_escape_markdown_cell(text))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def doxygen_html_table(table: Table) -> str:
    """The `<table class="manual">` style used by doc/DenseDecompositionBenchmark.dox."""
    lines = [f"\\subsection {table.anchor} {_escape_html(table.title)}", "",
             _escape_html(table.caption), "", '<table class="manual">']
    lines.append("<tr>" + "".join(f"<th>{_escape_html(h)}</th>" for h in table.headers) + "</tr>")
    for index, row in enumerate(table.rows):
        cells = []
        for cell in row:
            text = _escape_html(cell.text)
            if cell.code and cell.text:
                text = f"<code>{text}</code>"
            if cell.muted:
                text = f"<em>{text}</em>"
            if cell.note:
                text = f'{text} <sup><a href="#note_{cell.note}">{cell.note}</a></sup>'
            cells.append(f"<td>{text}</td>")
        opening = '<tr class="alt">' if index % 2 else "<tr>"
        lines.append(opening + "".join(cells) + "</tr>")
    lines.append("</table>")
    lines.append("")
    return "\n".join(lines)


def render_doxygen(page: Page, table_style: str = "markdown") -> str:
    """Format an already-built page as a Doxygen manual page; see `build_page`."""
    formatter = doxygen_html_table if table_style == "html" else doxygen_markdown_table
    out = ["namespace Eigen {", "", f"/** \\eigenManualPage {DOXYGEN_PAGE_ID} {page.title}", ""]
    out.extend(_escape_html(paragraph) + "\n" for paragraph in page.intro)
    if not page.tables:
        out.append("No cell in the merged dataset matched the selection.\n")
    for table in page.tables:
        out.append(formatter(table))
    if page.footnotes:
        out.append("\\b Notes:")
        out.append("")
        for marker, text in page.footnotes:
            if table_style == "html":
                out.append(f'<a name="note_{marker}"><b>{marker}</b>: </a> {_escape_html(text)}')
                out.append("")
            else:
                out.append(f"- <b>[{marker}]</b> {_escape_html(text)}")
        out.append("")
    out.append("*/")
    out.append("")
    out.append("}")
    return "\n".join(out).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Coverage manifest
# ---------------------------------------------------------------------------


def coverage_model(merged: Mapping[str, Any], options: RenderOptions) -> Dict[str, Any]:
    """Machine-readable coverage manifest for the selected slice."""
    cells = select_cells(merged, options, by_config=True)
    groups: Dict[Tuple[str, str, str, str, str], Dict[str, Any]] = {}
    measured = 0
    not_measured = 0
    unaccounted = 0
    per_op: Dict[str, Dict[str, Any]] = {}
    per_arm: Dict[str, Dict[str, int]] = {}
    for cell in cells:
        op_entry = per_op.setdefault(cell["op"], {"measured": 0, "not_measured": 0, "unaccounted": 0, "arms": []})
        for arm, entry in sorted((cell.get("arms") or {}).items()):
            if arm not in op_entry["arms"]:
                op_entry["arms"].append(arm)
            arm_entry = per_arm.setdefault(arm, {"measured": 0, "not_measured": 0})
            if entry.get("state") == "measured":
                measured += 1
                op_entry["measured"] += 1
                arm_entry["measured"] += 1
                continue
            not_measured += 1
            op_entry["not_measured"] += 1
            arm_entry["not_measured"] += 1
            reason = entry.get("reason") or "unaccounted"
            if reason == "unaccounted":
                unaccounted += 1
                op_entry["unaccounted"] += 1
            key = (cell["config_id"], cell["op"], arm, cell["scalar"], reason)
            group = groups.setdefault(
                key,
                {
                    "config_id": cell["config_id"],
                    "op": cell["op"],
                    "arm": arm,
                    "scalar": cell["scalar"],
                    "reason": reason,
                    "detail": entry.get("detail") or _reason_text(reason),
                    "count": 0,
                    "shapes": [],
                },
            )
            group["count"] += 1
            shape_text = compact_shape(cell)
            if shape_text not in group["shapes"]:
                group["shapes"].append(shape_text)
    for entry in per_op.values():
        entry["arms"].sort()
    return {
        "schema_version": "1.0.0",
        "kind": "eigen-benchmark-comparison-coverage",
        "generated_utc": merged.get("generated_utc"),
        "baseline": baseline_of(merged, options),
        "configs": sorted({cell["config_id"] for cell in cells}),
        "config_details": {
            config_id: (merged.get("configs") or {}).get(config_id, {})
            for config_id in sorted({cell["config_id"] for cell in cells})
        },
        "arms": per_arm,
        "ops": per_op,
        "scalars": sorted({cell["scalar"] for cell in cells}),
        "totals": {"measured": measured, "not_measured": not_measured, "unaccounted": unaccounted},
        "missing_configs": list((merged.get("coverage") or {}).get("missing_configs", [])),
        "not_measured": [groups[key] for key in sorted(groups)],
    }


def render_coverage_markdown(merged: Mapping[str, Any], registry: Mapping[str, Any], model: Mapping[str, Any]) -> str:
    """Render the human-readable half of the manifest from the model, which the
    caller also emits as JSON: deriving it twice from the same inputs is how the
    two halves come to disagree."""
    totals = model["totals"]
    out = ["# Benchmark comparison coverage", ""]
    out.append(
        "This manifest states exactly which machine, library and operation combinations were measured. "
        "Anything absent from the tables is listed here with the reason it is absent, so a partial dataset "
        "is self-describing."
    )
    out.append("")
    out.append(f"- measured arm cells: **{totals['measured']}**")
    out.append(f"- not measured arm cells: **{totals['not_measured']}**")
    out.append(f"- unaccounted (harness bug): **{totals['unaccounted']}**")
    out.append("")
    if totals["unaccounted"]:
        out.append(
            "> **Warning.** A non-zero unaccounted count means cells were in a run's scope yet reported by "
            "neither `measurements` nor `not_measured`. That is a harness bug, not a property of the machine."
        )
        out.append("")

    out.append("## Configurations")
    out.append("")
    if model["configs"]:
        out.append("| Config | Machine | ISA | Compiler | Threads | Eigen commit |")
        out.append("|---|---|---|---|---|---|")
        for config_id in model["configs"]:
            config = model["config_details"].get(config_id, {})
            out.append(
                "| `{}` | {} | {} | {} | {} | `{}` |".format(
                    config_id,
                    _escape_markdown_cell(str(config.get("cpu_model") or config.get("machine_config_id") or "?")),
                    _escape_markdown_cell(str(config.get("isa_target") or "?")),
                    _escape_markdown_cell(str(config.get("compiler") or "?")),
                    config.get("threads", "?"),
                    config.get("eigen_commit_short") or "?",
                )
            )
        for config_id in model["configs"]:
            if (model["config_details"].get(config_id) or {}).get("eigen_dirty"):
                out.append("")
                out.append(f"> `{config_id}` was measured from a **dirty** Eigen worktree and is not reproducible.")
    else:
        out.append("No configuration in the selected slice.")
    out.append("")

    out.append("## Operations")
    out.append("")
    out.append("| BLAS/LAPACK | Eigen | Arms | Measured | Not measured | Unaccounted |")
    out.append("|---|---|---|---|---|---|")
    for op in sorted(model["ops"]):
        entry = model["ops"][op]
        meta = op_metadata(merged, registry, op)
        out.append(
            "| {} | `{}` | {} | {} | {} | {} |".format(
                _escape_markdown_cell(meta.get("display_name", op)),
                _escape_markdown_cell(eigen_spelling(meta)),
                ", ".join(entry["arms"]) or "-",
                entry["measured"],
                entry["not_measured"],
                entry["unaccounted"],
            )
        )
    out.append("")

    out.append("## Not measured, by reason")
    out.append("")
    if not model["not_measured"]:
        out.append("Every combination in the selected slice was measured.")
    else:
        out.append("| Config | BLAS/LAPACK | Arm | Scalar | Reason | Cells | Shapes |")
        out.append("|---|---|---|---|---|---|---|")
        for group in model["not_measured"]:
            shapes = group["shapes"]
            shown = ", ".join(shapes[:8]) + (", ..." if len(shapes) > 8 else "")
            out.append(
                "| `{}` | {} | {} | {} | `{}` | {} | {} |".format(
                    group["config_id"],
                    _escape_markdown_cell(op_metadata(merged, registry, group["op"]).get("display_name", group["op"])),
                    group["arm"],
                    group["scalar"],
                    group["reason"],
                    group["count"],
                    shown,
                )
            )
        out.append("")
        out.append("Details:")
        out.append("")
        seen: List[str] = []
        for group in model["not_measured"]:
            line = f"- `{group['reason']}`: {group['detail']}"
            if line not in seen:
                seen.append(line)
                out.append(line)
    out.append("")
    if model["missing_configs"]:
        out.append("## Machines with no data")
        out.append("")
        for entry in model["missing_configs"]:
            out.append(f"- `{entry.get('machine_config_id')}` / {entry.get('op')}: {entry.get('reason')}")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = UsageErrorArgumentParser(
        prog="render.py",
        description="Render the merged comparison intermediate as Doxygen, markdown and a coverage manifest.",
    )
    parser.add_argument("merged", nargs="?", default="-", metavar="MERGED", help="merged file; '-' is stdin")
    parser.add_argument("--format", action="append", default=[], metavar="LIST",
                        help="doxygen|markdown|coverage|all (repeatable, comma-separated)")
    parser.add_argument("--out-dir", default=None, metavar="DIR")
    parser.add_argument("--out", default=None, metavar="PATH", help="single-file output; '-' is stdout")
    parser.add_argument("--config", action="append", default=[], metavar="ID")
    parser.add_argument("--op", action="append", default=[], metavar="OP")
    parser.add_argument("--scalar", action="append", default=[], metavar="TAG")
    parser.add_argument("--baseline", default=None, metavar="ARM")
    parser.add_argument("--metric", choices=["gflops", "time", "ratio"], default="gflops")
    parser.add_argument("--not-measured-token", default=DEFAULT_NOT_MEASURED_TOKEN, metavar="STR")
    parser.add_argument("--title", default=None, metavar="STR")
    parser.add_argument("--table-style", choices=["markdown", "html"], default="markdown",
                        help="Doxygen table syntax; 'html' emits the <table class=\"manual\"> style used by doc/")
    parser.add_argument("--ops-toml", default=DEFAULT_OPS_TOML, metavar="PATH")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    def warn(message: str) -> None:
        print(f"render.py: {message}", file=sys.stderr)

    formats = split_list(args.format) or ["all"]
    if "all" in formats:
        formats = ["doxygen", "markdown", "coverage"]
    unknown = [name for name in formats if name not in ("doxygen", "markdown", "coverage")]
    if unknown:
        warn(f"unknown --format {unknown[0]!r}")
        return EXIT_USAGE
    if bool(args.out_dir) == bool(args.out):
        warn("exactly one of --out-dir and --out is required")
        return EXIT_USAGE
    if args.out and len(set(formats)) != 1:
        warn("--out requires exactly one --format")
        return EXIT_USAGE

    options = RenderOptions(
        configs=split_list(args.config) or None,
        ops=split_list(args.op) or None,
        scalars=split_list(args.scalar) or None,
        baseline=args.baseline,
        metric=args.metric,
        not_measured_token=args.not_measured_token,
        title=args.title,
        table_style=args.table_style,
    )

    try:
        merged = load_merged(args.merged)
        registry = load_ops_toml(args.ops_toml)
        # The two table formats are two spellings of one page model, and the two
        # halves of the coverage manifest are two spellings of one coverage
        # model. Under the default --format all each was derived twice.
        page = build_page(merged, registry, options) if {"doxygen", "markdown"} & set(formats) else None
        coverage = coverage_model(merged, options) if "coverage" in formats else None
        outputs: List[Tuple[str, str]] = []
        for name in formats:
            if name == "doxygen":
                outputs.append((DEFAULT_FILENAMES["doxygen"], render_doxygen(page, options.table_style)))
            elif name == "markdown":
                outputs.append((DEFAULT_FILENAMES["markdown"], render_markdown(page)))
            else:
                outputs.append(
                    (DEFAULT_FILENAMES["coverage"][0], render_coverage_markdown(merged, registry, coverage))
                )
                outputs.append(
                    (DEFAULT_FILENAMES["coverage"][1], json.dumps(coverage, indent=2, sort_keys=True) + "\n")
                )
    except PipelineError as exc:
        warn(str(exc))
        return exc.code

    if args.out:
        text = outputs[0][1]
        if args.out == "-":
            sys.stdout.write(text)
        else:
            _write(args.out, text)
        return EXIT_OK

    os.makedirs(args.out_dir, exist_ok=True)
    for filename, text in outputs:
        _write(os.path.join(args.out_dir, filename), text)
        if args.verbose:
            warn(f"wrote {os.path.join(args.out_dir, filename)}")
    return EXIT_OK


def _write(path: str, text: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


if __name__ == "__main__":
    sys.exit(main())
