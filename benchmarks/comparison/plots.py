#!/usr/bin/env python3
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

"""Plot the merged comparison intermediate with matplotlib.

Three properties this module exists to guarantee:

* **Deterministic output.** No timestamps, no randomness, stable ordering,
  fixed figure size and DPI, a pinned SVG hash salt.  Two runs over the same
  merged file produce byte-identical files.
* **Missing data is a gap, never a zero.** A zero would read as "Eigen scored
  0 GFLOP/s", which is the worst failure mode a published performance page has.
  Unmeasured points are ``float("nan")`` and the axes are never rescaled as
  though the gap were a zero.
* **Never colour alone.** The palette is colourblind-safe and validated, and
  every series also varies its marker and line style, because these figures get
  printed in greyscale.

Figure construction is a pure function returning a ``Figure``; writing files is
separate, so the figures can be unit-tested without touching disk.
"""

from __future__ import annotations

import argparse
import math
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

from _common import (
    resolve_baseline,  # noqa: E402
    PipelineError,
    UsageErrorArgumentParser,
    arm_display,
    arms_in,
    compact_shape,
    config_parts,
    load_merged,
    select_cells,
    split_list,
)

# A headless backend, guaranteed, without paying for the import: MPLBACKEND is
# what a not-yet-imported matplotlib will read, and an already-imported one is
# switched in place. Importing plots.py must never be able to open a window -- a
# CI job would hang -- but importing matplotlib itself costs about half a second
# on every `--help` and every usage error, so `_require_matplotlib` below defers
# it until a figure is actually drawn.
os.environ.setdefault("MPLBACKEND", "Agg")
if "matplotlib" in sys.modules:  # pragma: no cover - depends on import order
    sys.modules["matplotlib"].use("Agg")

matplotlib: Any = None
plt: Any = None
Figure = Any

#: Categorical palette: Okabe-Ito, re-ordered so that every adjacent pair clears
#: the CVD separation floor.  Validated with the dataviz palette validator
#: against the light surface below (all six checks pass; the sub-3:1 contrast
#: warning is relieved by the always-present legend, the >=8px markers and the
#: tables render.py emits from the same data).  Swap this one constant to
#: retheme every figure.
PALETTE: Tuple[str, ...] = ("#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9", "#CC79A7")

#: Secondary encodings, applied by the same index as the palette, so no series
#: is distinguished by colour alone.
MARKERS: Tuple[str, ...] = ("o", "s", "^", "D", "v", "P")
LINESTYLES: Tuple[Any, ...] = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1)))
HATCHES: Tuple[str, ...] = ("", "///", "...", "xxx", "\\\\\\", "+++")

SURFACE = "#FCFCFB"
INK = "#1A1A1A"
INK_MUTED = "#5C5C5A"
GRID = "#D9D9D6"

KINDS = ("rate-vs-size", "ratio-vs-size", "bar", "roofline")

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_INPUT = 2
EXIT_NO_MATPLOTLIB = 3


class PlotError(PipelineError):
    def __init__(self, message: str, code: int = EXIT_INPUT):
        super().__init__(message, code)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Series:
    """One library's curve.  ``y`` carries NaN wherever nothing was measured."""

    label: str
    x: List[float]
    y: List[float]
    index: int = 0

    @property
    def partial(self) -> bool:
        return any(math.isnan(value) for value in self.y)

    @property
    def empty(self) -> bool:
        return all(math.isnan(value) for value in self.y)

    def legend_label(self) -> str:
        if self.empty:
            return f"{self.label} (not measured)"
        if self.partial:
            return f"{self.label} (partial)"
        return self.label


@dataclass
class PlotOptions:
    configs: Optional[Sequence[str]] = None
    ops: Optional[Sequence[str]] = None
    scalars: Optional[Sequence[str]] = None
    baseline: Optional[str] = None
    width: float = 7.0
    height: float = 4.5
    dpi: int = 150
    log_x: bool = True


def _gflops(cell: Mapping[str, Any], arm: str) -> float:
    entry = (cell.get("arms") or {}).get(arm) or {}
    if entry.get("state") != "measured" or entry.get("gflops") is None:
        # A gap, never a zero.
        return float("nan")
    return float(entry["gflops"])


def size_key_collisions(cells: Sequence[Mapping[str, Any]]) -> List[Tuple[float, List[str]]]:
    """Distinct shapes that share one abscissa, as (size_key, shape labels).

    `size_key` is the rounded geometric mean of the dimensions, so it is not
    injective over a mixed grid: the default GEMM grid maps (64,64,1024),
    (1024,64,64) and (64,1024,64) onto 161, and (512,512,512) onto the same 512
    as (8192,128,128). A line plot keyed on it draws several different shapes at
    one x and connects them, which reads as a vertical collapse of the rate at a
    single "problem size" on an axis labelled as one.
    """
    by_key: Dict[float, List[str]] = {}
    for cell in cells:
        by_key.setdefault(float(cell.get("size_key", 0)), []).append(compact_shape(cell))
    return [
        (key, sorted(set(labels)))
        for key, labels in sorted(by_key.items())
        if len(set(labels)) > 1
    ]


def rate_series(merged: Mapping[str, Any], cells: Sequence[Mapping[str, Any]], baseline: Optional[str]) -> List[Series]:
    arms = arms_in(cells, baseline)
    xs = [float(cell.get("size_key", 0)) for cell in cells]
    return [
        Series(label=arm_display(merged, arm), x=list(xs), y=[_gflops(cell, arm) for cell in cells], index=index)
        for index, arm in enumerate(arms)
    ]


def ratio_series(merged: Mapping[str, Any], cells: Sequence[Mapping[str, Any]], baseline: Optional[str]) -> List[Series]:
    label = f"Eigen / {arm_display(merged, baseline)}" if baseline else "Eigen / reference"
    values = []
    for cell in cells:
        ratio = cell.get("ratio")
        values.append(float("nan") if ratio is None else float(ratio))
    return [Series(label=label, x=[float(cell.get("size_key", 0)) for cell in cells], y=values, index=0)]


# ---------------------------------------------------------------------------
# Figures (pure: no disk access)
# ---------------------------------------------------------------------------


def _require_matplotlib() -> None:
    """Import matplotlib on first use, or say it is missing.

    Every entry point that draws calls this first, so the import is paid for by
    the invocations that produce a figure and by no others.
    """
    global matplotlib, plt
    if plt is not None:
        return
    try:
        import matplotlib as _matplotlib

        _matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        import matplotlib.ticker  # noqa: F401  (attribute access elsewhere)
    except ImportError as exc:  # pragma: no cover - only where matplotlib is absent
        raise PlotError("matplotlib is unavailable; install it to render plots", EXIT_NO_MATPLOTLIB) from exc
    matplotlib, plt = _matplotlib, _plt


def _new_figure(options: PlotOptions):
    figure = plt.figure(figsize=(options.width, options.height), dpi=options.dpi, facecolor="white")
    axes = figure.add_subplot(1, 1, 1)
    axes.set_facecolor(SURFACE)
    axes.grid(True, which="major", color=GRID, linewidth=0.6, zorder=0)
    axes.set_axisbelow(True)
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axes.spines[spine].set_color(GRID)
    axes.tick_params(colors=INK_MUTED, labelsize=9)
    return figure, axes


def _apply_size_axis(axes, series: Sequence["Series"], log_x: bool) -> None:
    """Log-scaled size axis labelled with the sizes themselves, not 2^k."""
    xs = sorted({value for item in series for value in item.x if value > 0})
    if log_x and xs:
        axes.set_xscale("log", base=2)
        axes.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=2.0))
        axes.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _: f"{int(round(value))}"))
        axes.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axes.set_xlim(min(xs) / 1.4, max(xs) * 1.4)
    elif xs:
        axes.set_xlim(min(xs) * 0.95 - 1.0, max(xs) * 1.05 + 1.0)


def _annotate_if_empty(axes, series: Sequence["Series"]) -> None:
    """An all-missing combination states so; it never renders as a zero line."""
    if not series or any(not item.empty for item in series):
        return
    axes.set_ylim(0.0, 1.0)
    axes.set_yticks([])
    axes.text(
        0.5,
        0.5,
        "not measured\nsee the coverage manifest",
        transform=axes.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        color=INK_MUTED,
    )


def _finish(figure, axes, title: str, subtitle: str, xlabel: str, ylabel: str, options: PlotOptions, legend: bool):
    axes.set_title(title, color=INK, fontsize=12, loc="left", pad=14)
    if subtitle:
        axes.text(
            0.0,
            1.02,
            subtitle,
            transform=axes.transAxes,
            color=INK_MUTED,
            fontsize=8.5,
            va="bottom",
            ha="left",
        )
    axes.set_xlabel(xlabel, color=INK_MUTED, fontsize=10)
    axes.set_ylabel(ylabel, color=INK_MUTED, fontsize=10)
    if legend:
        frame = axes.legend(loc="best", fontsize=9, frameon=True, facecolor="white", edgecolor=GRID)
        for text in frame.get_texts():
            text.set_color(INK)
    figure.tight_layout()
    return figure


def build_rate_figure(
    series: Sequence[Series],
    *,
    title: str,
    subtitle: str = "",
    ylabel: str = "GFLOP/s",
    xlabel: str = "problem size (geometric mean of the dimensions)",
    options: Optional[PlotOptions] = None,
) -> "Figure":
    """Rate versus size, one series per library.  Pure; returns the Figure."""
    _require_matplotlib()
    options = options or PlotOptions()
    figure, axes = _new_figure(options)
    for item in series:
        style = item.index % len(PALETTE)
        axes.plot(
            item.x,
            item.y,
            label=item.legend_label(),
            color=PALETTE[style],
            linestyle=LINESTYLES[style % len(LINESTYLES)],
            marker=MARKERS[style % len(MARKERS)],
            markersize=5.0,
            markeredgecolor="white",
            markeredgewidth=0.7,
            linewidth=2.0,
            zorder=3 + item.index,
        )
    _apply_size_axis(axes, series, options.log_x)
    # Bottom the value axis at zero without letting a gap masquerade as one.
    finite = [value for item in series for value in item.y if not math.isnan(value)]
    if finite:
        axes.set_ylim(0.0, max(finite) * 1.12)
    _annotate_if_empty(axes, series)
    return _finish(figure, axes, title, subtitle, xlabel, ylabel, options, legend=True)


def build_ratio_figure(
    series: Sequence[Series],
    *,
    title: str,
    subtitle: str = "",
    ylabel: str = "Eigen / reference (>1 means Eigen is faster)",
    xlabel: str = "problem size (geometric mean of the dimensions)",
    options: Optional[PlotOptions] = None,
) -> "Figure":
    _require_matplotlib()
    options = options or PlotOptions()
    figure, axes = _new_figure(options)
    axes.axhline(1.0, color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
    for item in series:
        style = item.index % len(PALETTE)
        axes.plot(
            item.x,
            item.y,
            label=item.legend_label(),
            color=PALETTE[style],
            linestyle=LINESTYLES[style % len(LINESTYLES)],
            marker=MARKERS[style % len(MARKERS)],
            markersize=5.0,
            markeredgecolor="white",
            markeredgewidth=0.7,
            linewidth=2.0,
            zorder=3 + item.index,
        )
    _apply_size_axis(axes, series, options.log_x)
    finite = [value for item in series for value in item.y if not math.isnan(value) and value > 0.0]
    if finite:
        span = max(finite + [1.0]) / min(finite + [1.0])
        if span >= 2.0:
            # A log axis so that 2x and 0.5x sit symmetrically about parity.
            axes.set_yscale("log", base=2)
            axes.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=2.0))
            axes.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _: f"{value:g}"))
            axes.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        else:
            low, high = min(finite + [1.0]), max(finite + [1.0])
            pad = max((high - low) * 0.15, 0.01)
            axes.set_ylim(low - pad, high + pad)
    _annotate_if_empty(axes, series)
    return _finish(figure, axes, title, subtitle, xlabel, ylabel, options, legend=True)


def build_bar_figure(
    categories: Sequence[str],
    series: Sequence[Series],
    *,
    title: str,
    subtitle: str = "",
    ylabel: str = "GFLOP/s",
    xlabel: str = "shape",
    not_measured_token: str = "n/a",
    options: Optional[PlotOptions] = None,
) -> "Figure":
    """Grouped bars.  A missing value draws no bar and is labelled instead."""
    _require_matplotlib()
    options = options or PlotOptions()
    figure, axes = _new_figure(options)
    count = max(len(series), 1)
    # A 2px surface gap between adjacent bars at the default DPI.
    slot = 0.82 / count
    positions = list(range(len(categories)))
    for item in series:
        style = item.index % len(PALETTE)
        offset = (item.index - (count - 1) / 2.0) * slot
        drawn_x = [position + offset for position, value in zip(positions, item.y) if not math.isnan(value)]
        drawn_y = [value for value in item.y if not math.isnan(value)]
        axes.bar(
            drawn_x,
            drawn_y,
            width=slot * 0.88,
            label=item.legend_label(),
            color=PALETTE[style],
            edgecolor="white",
            linewidth=1.0,
            hatch=HATCHES[style % len(HATCHES)] or None,
            zorder=3,
        )
        for position, value in zip(positions, item.y):
            if math.isnan(value):
                axes.text(
                    position + offset,
                    0.0,
                    not_measured_token,
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color=INK_MUTED,
                    zorder=4,
                )
    axes.set_xticks(positions)
    axes.set_xticklabels(categories, rotation=45, ha="right", fontsize=8)
    axes.grid(True, axis="y", color=GRID, linewidth=0.6)
    axes.grid(False, axis="x")
    finite = [value for item in series for value in item.y if not math.isnan(value)]
    if finite:
        axes.set_ylim(0.0, max(finite) * 1.15)
    _annotate_if_empty(axes, series)
    return _finish(figure, axes, title, subtitle, xlabel, ylabel, options, legend=True)


def build_roofline_figure(
    series: Sequence[Series],
    *,
    title: str,
    subtitle: str = "",
    options: Optional[PlotOptions] = None,
) -> "Figure":
    """Rate versus size against the best rate actually observed.

    A textbook roofline needs the machine's peak flop rate and peak bandwidth.
    The harness does not measure either, and inventing them would publish a
    made-up efficiency figure, so the ceiling drawn here is the highest rate any
    arm reached in this dataset and is labelled as exactly that.
    """
    _require_matplotlib()
    options = options or PlotOptions()
    figure = build_rate_figure(series, title=title, subtitle=subtitle, options=options)
    axes = figure.axes[0]
    finite = [value for item in series for value in item.y if not math.isnan(value)]
    if finite:
        ceiling = max(finite)
        axes.axhline(ceiling, color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
        axes.text(
            0.99,
            ceiling,
            f"best observed {ceiling:.1f} GFLOP/s",
            transform=axes.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=8,
            color=INK_MUTED,
        )
    return figure


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def figure_title(merged: Mapping[str, Any], op: str, scalar: str) -> str:
    meta = (merged.get("ops") or {}).get(op) or {}
    return f"{meta.get('display_name', op)}, {scalar}"


def figure_subtitle(merged: Mapping[str, Any], config_id: str) -> str:
    """The line under a figure's title; the same fields render.py captions with.

    Shared so a plot cannot describe a configuration differently from the table
    of the same cell -- notably, both now carry the dirty-worktree caveat.
    """
    return " | ".join(config_parts(merged, config_id))


def build_figure(
    merged: Mapping[str, Any],
    config_id: str,
    op: str,
    scalar: str,
    kind: str,
    options: PlotOptions,
) -> "Figure":
    """Build one figure.  Pure: nothing here touches the filesystem."""
    _require_matplotlib()
    if kind not in KINDS:
        raise PlotError(f"unknown plot kind {kind!r}", EXIT_USAGE)
    scoped = PlotOptions(
        configs=[config_id],
        ops=[op],
        scalars=[scalar],
        baseline=options.baseline,
        width=options.width,
        height=options.height,
        dpi=options.dpi,
        log_x=options.log_x,
    )
    cells = select_cells(merged, scoped)
    if not cells:
        raise PlotError(f"no cell for {config_id}/{op}/{scalar}")
    baseline = resolve_baseline(merged, options.baseline)
    title = figure_title(merged, op, scalar)
    subtitle = figure_subtitle(merged, config_id)
    # A size axis is only meaningful while size_key separates the shapes. Where it
    # does not, the shapes themselves become the categories: bars keyed on
    # compact_shape say which shape each number belongs to, where a line through
    # two shapes sharing an abscissa says something untrue about both.
    collisions = size_key_collisions(cells)
    if collisions:
        shared = ", ".join(f"{key:g} ({'/'.join(labels)})" for key, labels in collisions[:3])
        subtitle = _join_subtitle(
            subtitle,
            "shown per shape: distinct shapes share a geometric-mean size here (" + shared + ")",
        )
    if kind == "ratio-vs-size":
        label = arm_display(merged, baseline) if baseline else "reference"
        ylabel = f"Eigen / {label} (>1 means Eigen is faster)"
        if collisions:
            return build_bar_figure(
                [compact_shape(cell) for cell in cells],
                ratio_series(merged, cells, baseline),
                title=title,
                subtitle=subtitle,
                ylabel=ylabel,
                options=options,
            )
        return build_ratio_figure(
            ratio_series(merged, cells, baseline),
            title=title,
            subtitle=subtitle,
            ylabel=ylabel,
            options=options,
        )
    if kind == "bar" or collisions:
        return build_bar_figure(
            [compact_shape(cell) for cell in cells],
            rate_series(merged, cells, baseline),
            title=title,
            subtitle=subtitle,
            options=options,
        )
    if kind == "rate-vs-size":
        return build_rate_figure(rate_series(merged, cells, baseline), title=title, subtitle=subtitle, options=options)
    return build_roofline_figure(rate_series(merged, cells, baseline), title=title, subtitle=subtitle, options=options)


def _join_subtitle(subtitle: str, extra: str) -> str:
    return f"{subtitle} | {extra}" if subtitle else extra


def output_filename(config_id: str, op: str, scalar: str, kind: str, extension: str) -> str:
    return f"{config_id}__{op}__{scalar}__{kind}.{extension}"


def save_figure(figure: "Figure", path: str, dpi: int) -> None:
    """Write one figure, stripping the metadata that would defeat determinism."""
    _require_matplotlib()
    extension = os.path.splitext(path)[1].lower()
    metadata = {"Date": None} if extension == ".svg" else {"Software": None}
    figure.savefig(path, dpi=dpi, facecolor=figure.get_facecolor(), metadata=metadata)


def plan(merged: Mapping[str, Any], options: PlotOptions, kinds: Sequence[str]) -> List[Tuple[str, str, str, str]]:
    """Deterministic list of (config_id, op, scalar, kind) to render."""
    cells = select_cells(merged, options)
    combos = sorted({(cell["config_id"], cell["op"], cell["scalar"]) for cell in cells})
    return [(config_id, op, scalar, kind) for config_id, op, scalar in combos for kind in kinds]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = UsageErrorArgumentParser(prog="plots.py", description="Plot the merged Eigen benchmark comparison data.")
    parser.add_argument("merged", nargs="?", default="-", metavar="MERGED", help="merged file; '-' is stdin")
    parser.add_argument("--out-dir", required=True, metavar="DIR")
    parser.add_argument("--format", choices=["svg", "png", "both"], default="svg")
    parser.add_argument("--kind", action="append", default=[], metavar="LIST",
                        help="rate-vs-size|ratio-vs-size|bar|roofline (repeatable, comma-separated)")
    parser.add_argument("--config", action="append", default=[], metavar="ID")
    parser.add_argument("--op", action="append", default=[], metavar="OP")
    parser.add_argument("--scalar", action="append", default=[], metavar="TAG")
    parser.add_argument("--baseline", default=None, metavar="ARM")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--width", type=float, default=7.0, metavar="IN")
    parser.add_argument("--height", type=float, default=4.5, metavar="IN")
    parser.add_argument("--log-x", dest="log_x", action="store_true", default=True)
    parser.add_argument("--no-log-x", dest="log_x", action="store_false")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    def warn(message: str) -> None:
        print(f"plots.py: {message}", file=sys.stderr)

    try:
        _require_matplotlib()
    except PlotError as exc:
        warn("matplotlib is unavailable; no plot was written")
        return exc.code

    kinds = split_list(args.kind) or ["rate-vs-size"]
    unknown = [kind for kind in kinds if kind not in KINDS]
    if unknown:
        warn(f"unknown --kind {unknown[0]!r}")
        return EXIT_USAGE

    # Pinned so SVG element ids do not vary between runs.
    matplotlib.rcParams["svg.hashsalt"] = "eigen-benchmark-comparison"
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["path.simplify"] = False

    options = PlotOptions(
        configs=split_list(args.config) or None,
        ops=split_list(args.op) or None,
        scalars=split_list(args.scalar) or None,
        baseline=args.baseline,
        width=args.width,
        height=args.height,
        dpi=args.dpi,
        log_x=args.log_x,
    )
    extensions = ["svg", "png"] if args.format == "both" else [args.format]

    try:
        merged = load_merged(args.merged)
        os.makedirs(args.out_dir, exist_ok=True)
        for config_id, op, scalar, kind in plan(merged, options, kinds):
            figure = build_figure(merged, config_id, op, scalar, kind, options)
            try:
                for extension in extensions:
                    path = os.path.join(args.out_dir, output_filename(config_id, op, scalar, kind, extension))
                    save_figure(figure, path, args.dpi)
                    if args.verbose:
                        warn(f"wrote {path}")
            finally:
                plt.close(figure)
    except PipelineError as exc:
        warn(str(exc))
        return exc.code
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
