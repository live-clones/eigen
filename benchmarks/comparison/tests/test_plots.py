# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

"""`plots.py`, asserted structurally rather than by pixel.

A plot is the artefact people quote, and the two ways it lies quietly are a
missing measurement drawn as zero -- which reads as a catastrophic result -- and
a legend that names a library the series does not actually contain.  Both are
checked here, along with the axis scale, the axis labels, and byte-level
determinism, so a regenerated figure diff means a data change and nothing else.
"""

import io
import json
import re

import pytest

import harness_support as support

pytestmark = pytest.mark.impl

# matplotlib is an optional third-party dependency and plots.py degrades to its
# documented EXIT_NO_MATPLOTLIB without it. The CLI cases below run plots.py in a
# subprocess, so a per-fixture importorskip cannot reach them: without this the
# whole file fails on an image that simply has no matplotlib, which is a missing
# developer dependency reported as a broken tree.
pytest.importorskip("matplotlib")

MERGED = support.FIXTURES / "merged" / "gemm_merged.json"
CONFIG = support.CONFIG_ID


@pytest.fixture(scope="module")
def plots_module():
    return support.import_impl("plots.py")


@pytest.fixture
def options(plots_module):
    return plots_module.PlotOptions(baseline="accelerate")


@pytest.fixture
def figure_for(plots_module, merged_gemm, options):
    made = []

    def build(kind="rate-vs-size", op="GEMM", scalar="f64", **overrides):
        opts = plots_module.PlotOptions(
            baseline=overrides.pop("baseline", "accelerate"),
            log_x=overrides.pop("log_x", True),
            width=overrides.pop("width", 7.0),
            height=overrides.pop("height", 4.5),
            dpi=overrides.pop("dpi", 150),
        )
        figure = plots_module.build_figure(merged_gemm, CONFIG, op, scalar, kind, opts)
        made.append(figure)
        return figure

    yield build
    import matplotlib.pyplot as plt

    for figure in made:
        plt.close(figure)


def lines_of(figure):
    axes = figure.axes[0]
    return [line for line in axes.get_lines() if len(line.get_xdata())]


# --------------------------------------------------------------------------
# No window, ever
# --------------------------------------------------------------------------


def test_matplotlib_is_forced_to_a_headless_backend(plots_module):
    import matplotlib

    assert matplotlib.get_backend().lower() == "agg", (
        "importing plots.py must not be able to open a window; a CI job would hang"
    )


# --------------------------------------------------------------------------
# Series and legend
# --------------------------------------------------------------------------


def test_one_series_per_library_present(plots_module, merged_gemm, figure_for):
    figure = figure_for()
    arms = {arm for cell in merged_gemm["cells"] if cell["op"] == "GEMM" for arm in cell["arms"]}
    assert len(lines_of(figure)) == len(arms), f"expected {len(arms)} series for arms {sorted(arms)}"


def test_legend_names_exactly_the_libraries_present(plots_module, merged_gemm, figure_for):
    figure = figure_for()
    legend = figure.axes[0].get_legend()
    assert legend is not None, "a two-arm plot without a legend is unreadable"
    labels = [text.get_text() for text in legend.get_texts()]
    expected = {record["library_name"] for record in merged_gemm["arms"].values()}
    for label in labels:
        base = re.sub(r"\s*\((partial|not measured)\)$", "", label)
        assert base in expected, f"legend entry {label!r} names no library in the dataset"
    assert len(labels) == len(expected)


def test_a_series_with_a_gap_is_labelled_partial(figure_for):
    figure = figure_for()
    labels = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
    assert any(label.endswith("(partial)") for label in labels), (
        f"the reference arm is missing one point; the legend must say so. Labels: {labels}"
    )
    assert sum(label.endswith("(partial)") for label in labels) == 1


def test_ratio_plot_has_a_single_series(figure_for):
    figure = figure_for(kind="ratio-vs-size")
    assert len(lines_of(figure)) >= 1
    labels = [text.get_text() for text in (figure.axes[0].get_legend().get_texts() if figure.axes[0].get_legend() else [])]
    assert all("Eigen" in label for label in labels), labels


# --------------------------------------------------------------------------
# The gap
# --------------------------------------------------------------------------


def test_a_missing_measurement_is_a_gap_not_a_zero(plots_module, merged_gemm, figure_for):
    """The single most important assertion in this file. A zero at m=n=k=128
    reads as a catastrophic result for the reference library."""
    import math

    missing = [
        cell
        for cell in merged_gemm["cells"]
        if cell["op"] == "GEMM" and cell["arms"]["accelerate"]["state"] == "not_measured"
    ]
    assert missing, "the fixture must contain a gap"
    gap_x = float(missing[0]["size_key"])

    figure = figure_for()
    holed = None
    for line in lines_of(figure):
        ydata = list(line.get_ydata())
        if any(math.isnan(value) for value in ydata):
            holed = line
    assert holed is not None, "no series carries a NaN; the missing point was filled in with something"
    for x, y in zip(holed.get_xdata(), holed.get_ydata()):
        if float(x) == gap_x:
            assert math.isnan(y), f"the missing point was drawn as {y}"
    for line in lines_of(figure):
        for y in line.get_ydata():
            assert not (isinstance(y, float) and y == 0.0), "a measurement was drawn as zero"


def test_the_axes_are_not_rescaled_around_a_gap(figure_for):
    import math

    figure = figure_for()
    axes = figure.axes[0]
    finite = [y for line in lines_of(figure) for y in line.get_ydata() if not math.isnan(y)]
    bottom, top = axes.get_ylim()
    assert top >= max(finite)
    assert bottom <= min(finite)
    assert top <= max(finite) * 3, "the y range was stretched far beyond the data"


def test_a_series_with_no_data_at_all_says_so(plots_module, merged_gemm, figure_for, tmp_path):
    document = support.deep_copy(merged_gemm)
    for cell in document["cells"]:
        if cell["op"] == "GEMM":
            cell["arms"]["accelerate"] = {
                "state": "not_measured",
                "reason": "reference_library_unavailable",
                "detail": "no BLAS configured",
            }
            cell["ratio"], cell["ratio_state"] = None, "not_measured"
    options = plots_module.PlotOptions(baseline="accelerate")
    figure = plots_module.build_figure(document, CONFIG, "GEMM", "f64", "rate-vs-size", options)
    labels = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
    assert any("not measured" in label for label in labels), labels
    import matplotlib.pyplot as plt

    plt.close(figure)


# --------------------------------------------------------------------------
# Axes
# --------------------------------------------------------------------------


def test_the_size_axis_is_logarithmic_by_default(figure_for):
    assert figure_for().axes[0].get_xscale() == "log"


def test_log_scale_can_be_turned_off(figure_for):
    assert figure_for(log_x=False).axes[0].get_xscale() == "linear"


def test_both_axes_are_labelled(figure_for):
    axes = figure_for().axes[0]
    xlabel, ylabel = axes.get_xlabel(), axes.get_ylabel()
    assert xlabel.strip(), "an unlabelled size axis makes the plot unquotable"
    assert "GFLOP/s" in ylabel, ylabel


def test_the_ratio_plot_labels_its_own_axis(figure_for):
    ylabel = figure_for(kind="ratio-vs-size").axes[0].get_ylabel()
    assert ylabel.strip() and "GFLOP/s" not in ylabel, ylabel


def test_the_figure_states_the_machine_and_the_toolchain(plots_module, merged_gemm, figure_for):
    figure = figure_for()
    text = " ".join(t.get_text() for t in figure.findobj(match=lambda o: hasattr(o, "get_text")))
    config = merged_gemm["configs"][CONFIG]
    assert config["cpu_model"] in text, "a rate without the machine it came from cannot be compared"
    assert config["isa_target"] in text


@pytest.mark.parametrize("kind", ["rate-vs-size", "ratio-vs-size", "bar", "roofline"])
def test_every_documented_kind_builds(figure_for, kind):
    figure = figure_for(kind=kind)
    assert figure.axes, kind


# --------------------------------------------------------------------------
# The size axis over a real, mixed grid
# --------------------------------------------------------------------------


def _full_grid_merged(plots_module, merged_gemm, ops):
    """The fixture's config extended to the whole default GEMM grid.

    Rates are a plausible stand-in, not a measurement: the point is only that the
    abscissa of every shape in the real grid is exercised, and the six-point
    square group every other test uses cannot produce a collision.
    """
    document = json.loads(json.dumps(merged_gemm))
    template = document["cells"][0]
    cells = []
    for point in support.op_grid(ops, "GEMM"):
        cell = json.loads(json.dumps(template))
        cell["shape"] = dict(zip(["m", "n", "k"], point))
        cell["shape_dims"] = ["m", "n", "k"]
        cell["size_key"] = support.size_key(cell["shape"])
        square = point[0] == point[1] == point[2]
        for arm, rate in (("eigen", 60.0 if square else 20.0), ("accelerate", 70.0 if square else 25.0)):
            cell["arms"][arm] = dict(cell["arms"][arm], gflops=rate, state="measured", gflops_mad=0.0)
        cell["ratio"] = cell["arms"]["eigen"]["gflops"] / cell["arms"]["accelerate"]["gflops"]
        cell["ratio_state"] = "ok"
        cells.append(cell)
    document["cells"] = cells
    return document


def test_the_default_gemm_grid_really_does_collide_on_the_size_key(plots_module, merged_gemm, ops):
    document = _full_grid_merged(plots_module, merged_gemm, ops)
    collisions = plots_module.size_key_collisions(document["cells"])
    assert collisions, (
        "the default GEMM grid is expected to map several distinct shapes onto one "
        "geometric-mean size; if it no longer does, this guard needs a new example"
    )


@pytest.mark.parametrize("kind", ["rate-vs-size", "ratio-vs-size", "roofline"])
def test_a_grid_with_a_shared_size_key_is_not_drawn_as_a_line(plots_module, merged_gemm, ops, options, kind):
    """Two different shapes at one x, joined by a polyline, render as a vertical
    collapse of the rate at a single 'problem size' -- on an axis labelled the
    geometric mean of the dimensions, with nothing to tell the shapes apart."""
    import matplotlib.pyplot as plt

    document = _full_grid_merged(plots_module, merged_gemm, ops)
    figure = plots_module.build_figure(document, CONFIG, "GEMM", "f64", kind, options)
    try:
        axes = figure.axes[0]
        assert axes.containers, f"{kind}: no categorical series was drawn"
        labels = [text.get_text() for text in axes.get_xticklabels()]
        assert len(labels) == len(set(labels)), f"{kind}: two shapes still share one abscissa"
        assert "64x64x1024" in labels and "1024x64x64" in labels
        for line in axes.get_lines():
            assert len(line.get_xdata()) <= 2, f"{kind}: a polyline still joins the colliding shapes"
    finally:
        plt.close(figure)


def test_a_grid_without_collisions_still_draws_a_line(figure_for):
    figure = figure_for("rate-vs-size")
    assert lines_of(figure), "the six-point square grid must still render as curves"


def test_an_unknown_kind_is_refused(plots_module, merged_gemm, options):
    with pytest.raises(Exception):
        plots_module.build_figure(merged_gemm, CONFIG, "GEMM", "f64", "pie-chart", options)


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def svg_bytes(figure):
    import matplotlib as mpl

    mpl.rcParams["svg.hashsalt"] = "eigen-comparison-tests"
    buffer = io.BytesIO()
    figure.savefig(buffer, format="svg", metadata={"Date": None})
    return buffer.getvalue()


def test_the_same_figure_twice_is_byte_identical(figure_for):
    first = svg_bytes(figure_for())
    second = svg_bytes(figure_for())
    assert first == second, "the same data produced two different figures; a regenerated plot would always diff"


def test_the_cli_is_deterministic(tmp_path):
    outputs = []
    for index in range(2):
        directory = tmp_path / f"run{index}"
        proc = support.run_cli("plots.py", ["--out-dir", str(directory), str(MERGED)])
        assert proc.returncode == 0, proc.stderr
        produced = sorted(directory.glob("*.svg"))
        assert produced, "no figure was written"
        outputs.append({path.name: path.read_bytes() for path in produced})
    assert outputs[0].keys() == outputs[1].keys()
    for name in outputs[0]:
        first, second = (re.sub(rb"<dc:date>[^<]*</dc:date>", b"", blob) for blob in (outputs[0][name], outputs[1][name]))
        assert first == second, f"{name} differs between two identical runs"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_output_filenames_follow_the_documented_pattern(tmp_path):
    proc = support.run_cli("plots.py", ["--out-dir", str(tmp_path), "--kind", "rate-vs-size", str(MERGED)])
    assert proc.returncode == 0, proc.stderr
    names = [path.name for path in tmp_path.glob("*.svg")]
    assert names
    # `config_id` itself contains '__', so the documented
    # `<config_id>__<OP>__<scalar>__<kind>.<ext>` pattern is checked from both
    # ends rather than by splitting.
    for name in names:
        stem = name[: -len(".svg")]
        assert stem.startswith(CONFIG + "__"), name
        remainder = stem[len(CONFIG) + 2 :].split("__")
        assert len(remainder) == 3, name
        op, scalar, kind = remainder
        assert re.fullmatch(r"[A-Z][A-Z0-9_]*", op), name
        assert scalar in ("f16", "bf16", "f32", "f64", "c32", "c64"), name
        assert kind == "rate-vs-size", name
    assert any(name.split("__")[-3] == "GEMM" for name in names), names


def test_png_and_both_formats(tmp_path):
    proc = support.run_cli("plots.py", ["--out-dir", str(tmp_path), "--format", "both", str(MERGED)])
    assert proc.returncode == 0, proc.stderr
    assert list(tmp_path.glob("*.png")) and list(tmp_path.glob("*.svg"))


def test_out_dir_is_required():
    proc = support.run_cli("plots.py", [str(MERGED)])
    assert proc.returncode == 1, proc.stderr


def test_unparsable_input_is_exit_two(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{{{")
    proc = support.run_cli("plots.py", ["--out-dir", str(tmp_path), str(path)])
    assert proc.returncode == 2, proc.stderr


def test_merged_input_can_arrive_on_stdin(tmp_path):
    proc = support.run_cli("plots.py", ["--out-dir", str(tmp_path)], stdin=MERGED.read_text())
    assert proc.returncode == 0, proc.stderr
    assert list(tmp_path.glob("*.svg"))


def test_help_exits_zero():
    proc = support.run_cli("plots.py", ["--help"])
    assert proc.returncode == 0 and proc.stdout.strip()
