"""Tests for ``explore_persona_space.analysis.paper_plots``."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from explore_persona_space.analysis.paper_plots import (
    add_direction_arrow,
    paper_palette,
    paper_palette_blog,
    paper_palette_role,
    proportion_ci,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

# ---------------------------------------------------------------------------
# paper_palette
# ---------------------------------------------------------------------------


def test_paper_palette_lengths_and_uniqueness() -> None:
    for n in range(1, 9):
        colors = paper_palette(n)
        assert len(colors) == n
        assert len(set(colors)) == n, f"palette has duplicate colors at n={n}"
        for c in colors:
            assert isinstance(c, str) and c.startswith("#") and len(c) == 7


def test_paper_palette_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        paper_palette(0)
    with pytest.raises(ValueError):
        paper_palette(9)
    with pytest.raises(ValueError):
        paper_palette(-1)


def test_paper_palette_returns_copy() -> None:
    colors = paper_palette(3)
    colors[0] = "mutated"
    fresh = paper_palette(3)
    assert fresh[0] != "mutated"


# ---------------------------------------------------------------------------
# proportion_ci
# ---------------------------------------------------------------------------


def test_proportion_ci_midpoint() -> None:
    lo, hi = proportion_ci(0.5, 100)
    # Expected half-width ~ 1.96 * sqrt(0.25/100) = 0.098
    assert lo == pytest.approx(0.402, abs=1e-3)
    assert hi == pytest.approx(0.598, abs=1e-3)


def test_proportion_ci_extreme_clamped() -> None:
    lo, hi = proportion_ci(0.0, 10)
    assert 0.0 <= lo <= hi <= 1.0
    assert lo == 0.0
    lo2, hi2 = proportion_ci(1.0, 10)
    assert 0.0 <= lo2 <= hi2 <= 1.0
    assert hi2 == 1.0


def test_proportion_ci_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        proportion_ci(0.5, 0)
    with pytest.raises(ValueError):
        proportion_ci(-0.01, 100)
    with pytest.raises(ValueError):
        proportion_ci(1.01, 100)


# ---------------------------------------------------------------------------
# set_paper_style — idempotence + key rcParams
# ---------------------------------------------------------------------------


def test_set_paper_style_idempotent() -> None:
    set_paper_style("neurips")
    snapshot_1 = dict(matplotlib.rcParams)
    set_paper_style("neurips")
    snapshot_2 = dict(matplotlib.rcParams)
    # All relevant keys should match between snapshots
    for key in (
        "font.family",
        "font.size",
        "axes.labelsize",
        "figure.figsize",
        "axes.spines.top",
        "axes.spines.right",
        "pdf.fonttype",
        "ps.fonttype",
        "savefig.dpi",
    ):
        assert snapshot_1[key] == snapshot_2[key], key


def test_set_paper_style_neurips_vs_generic_figsize() -> None:
    set_paper_style("neurips")
    assert tuple(matplotlib.rcParams["figure.figsize"]) == (5.5, 3.4)
    set_paper_style("generic")
    assert tuple(matplotlib.rcParams["figure.figsize"]) == (6.0, 4.0)


def test_set_paper_style_type42_fonts() -> None:
    set_paper_style("neurips")
    assert matplotlib.rcParams["pdf.fonttype"] == 42
    assert matplotlib.rcParams["ps.fonttype"] == 42


def test_set_paper_style_rejects_bad_target() -> None:
    with pytest.raises(ValueError):
        set_paper_style("paper")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# add_direction_arrow
# ---------------------------------------------------------------------------


def test_add_direction_arrow_appends_symbol() -> None:
    fig, ax = plt.subplots()
    ax.set_ylabel("Accuracy")
    add_direction_arrow(ax, axis="y", direction="up")
    assert ax.get_ylabel() == "Accuracy ↑ better"
    plt.close(fig)


def test_add_direction_arrow_down_on_x() -> None:
    fig, ax = plt.subplots()
    ax.set_xlabel("Loss")
    add_direction_arrow(ax, axis="x", direction="down")
    assert ax.get_xlabel() == "Loss ↓ better"
    plt.close(fig)


def test_add_direction_arrow_verbatim_label() -> None:
    fig, ax = plt.subplots()
    ax.set_ylabel("Accuracy")
    add_direction_arrow(ax, axis="y", label="Custom label")
    assert ax.get_ylabel() == "Custom label"
    plt.close(fig)


def test_add_direction_arrow_rejects_empty_label() -> None:
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        add_direction_arrow(ax, axis="y", direction="up")
    plt.close(fig)


def test_add_direction_arrow_rejects_bad_args() -> None:
    fig, ax = plt.subplots()
    ax.set_ylabel("X")
    with pytest.raises(ValueError):
        add_direction_arrow(ax, axis="z", direction="up")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        add_direction_arrow(ax, axis="y", direction="sideways")  # type: ignore[arg-type]
    plt.close(fig)


# ---------------------------------------------------------------------------
# savefig_paper
# ---------------------------------------------------------------------------


def _make_simple_fig() -> plt.Figure:
    set_paper_style("neurips")
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 4])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return fig


def test_savefig_paper_writes_png_pdf_and_meta(tmp_path: Path) -> None:
    fig = _make_simple_fig()
    written = savefig_paper(fig, "subdir/test_plot", dir=tmp_path)
    plt.close(fig)

    assert "png" in written and written["png"].exists()
    assert "pdf" in written and written["pdf"].exists()
    assert "meta" in written and written["meta"].exists()

    meta = json.loads(written["meta"].read_text())
    # Provenance keys are always present; `points` (+ provenance) appear when the
    # figure's data is extractable (the simple line fig here IS extractable).
    assert {"commit", "created", "figsize"} <= set(meta.keys())
    assert isinstance(meta["commit"], str) and meta["commit"]
    assert isinstance(meta["created"], str) and meta["created"].endswith("Z")
    assert len(meta["figsize"]) == 2
    assert all(isinstance(x, float) for x in meta["figsize"])
    # The 3-vertex line is extracted into the sidecar's `points` payload.
    assert meta["points"] == [
        {"x": 0.0, "y": 0.0, "_kind": "line"},
        {"x": 1.0, "y": 1.0, "_kind": "line"},
        {"x": 2.0, "y": 4.0, "_kind": "line"},
    ]
    assert meta["total_points"] == 3

    # PNG should have non-trivial size; PDF too.
    assert written["png"].stat().st_size > 0
    assert written["pdf"].stat().st_size > 0


def test_savefig_paper_png_only(tmp_path: Path) -> None:
    fig = _make_simple_fig()
    written = savefig_paper(fig, "only_png", dir=tmp_path, formats=("png",))
    plt.close(fig)
    assert "png" in written
    assert "pdf" not in written
    assert "meta" in written


def test_savefig_paper_rejects_unknown_format(tmp_path: Path) -> None:
    fig = _make_simple_fig()
    with pytest.raises(ValueError):
        savefig_paper(fig, "bad", dir=tmp_path, formats=("svg",))
    plt.close(fig)


# ---------------------------------------------------------------------------
# blog style
# ---------------------------------------------------------------------------


def test_paper_palette_blog_returns_blog_colors() -> None:
    colors = paper_palette_blog(4)
    assert colors == ["#1F4E9F", "#E08220", "#3FA577", "#C0413B"]


def test_paper_palette_blog_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        paper_palette_blog(0)
    with pytest.raises(ValueError):
        paper_palette_blog(99)


def test_set_paper_style_blog_applies_distinct_rcparams() -> None:
    set_paper_style("blog")
    rc = matplotlib.rcParams
    assert rc["axes.titleweight"] == "semibold"
    assert rc["axes.titlelocation"] == "left"
    assert rc["axes.grid.axis"] == "y"
    assert rc["axes.axisbelow"] is True
    assert rc["legend.frameon"] is False
    assert rc["figure.facecolor"] == "#FAFAFA"
    assert rc["figure.constrained_layout.use"] is True
    assert tuple(rc["figure.figsize"]) == (6.5, 4.0)


def test_paper_palette_role_switches_with_active_style() -> None:
    set_paper_style("neurips")
    assert paper_palette_role("primary") == "#0072B2"  # Wong blue
    set_paper_style("blog")
    assert paper_palette_role("primary") == "#1F4E9F"  # blog deep blue
    assert paper_palette_role("control") == "#3FA577"
    set_paper_style("neurips")  # restore for downstream tests


def test_paper_palette_role_rejects_unknown_role() -> None:
    set_paper_style("blog")
    with pytest.raises(ValueError):
        paper_palette_role("not-a-role")
    set_paper_style("neurips")


def test_set_title_subtitle_replaces_existing_title() -> None:
    set_paper_style("blog")
    fig, ax = plt.subplots()
    ax.set_title("OLD")
    set_title_subtitle(ax, "NEW", subtitle="sub")
    assert ax.get_title(loc="left") == "NEW"
    plt.close(fig)
    set_paper_style("neurips")


def test_set_paper_style_default_is_blog() -> None:
    set_paper_style()  # no arg → blog
    assert tuple(matplotlib.rcParams["figure.figsize"]) == (6.5, 4.0)
    assert matplotlib.rcParams["axes.titlelocation"] == "left"
    set_paper_style("neurips")  # restore neutral state for downstream tests


# ---------------------------------------------------------------------------
# savefig_paper — per-point data extraction into the sidecar (dashboard viewer)
# ---------------------------------------------------------------------------


def test_sidecar_extracts_scatter_with_point_labels(tmp_path: Path) -> None:
    """A labeled scatter emits one `points` row per point, with the label column."""
    set_paper_style("blog")
    fig, ax = plt.subplots()
    personas = ["villain", "kind teacher", "engineer"]
    x = [0.8, 0.1, 0.4]
    y = [0.7, 0.05, 0.3]
    ax.scatter(x, y)
    ax.set_xlabel("Alignment (cosine)")
    ax.set_ylabel("Base rate")
    for xi, yi, lbl in zip(x, y, personas, strict=True):
        ax.text(xi, yi + 0.02, lbl)
    written = savefig_paper(fig, "scatter", dir=tmp_path)
    plt.close(fig)

    meta = json.loads(written["meta"].read_text())
    pts = meta["points"]
    assert len(pts) == 3
    assert meta["total_points"] == 3
    # Columns are the axis labels; the nearest text becomes the identifier.
    assert pts[0]["Alignment (cosine)"] == 0.8
    assert pts[0]["Base rate"] == 0.7
    assert {p["label"] for p in pts} == set(personas)
    assert all(p["_kind"] == "scatter" for p in pts)


def test_sidecar_extracts_bar_with_error(tmp_path: Path) -> None:
    """A bar chart with yerr emits category + height + error per bar."""
    set_paper_style("blog")
    fig, ax = plt.subplots()
    ax.bar(["contrastive", "positive-only"], [0.7, 0.85], yerr=[0.04, 0.05])
    ax.set_ylabel("Sycophancy rate")
    written = savefig_paper(fig, "bar", dir=tmp_path)
    plt.close(fig)

    meta = json.loads(written["meta"].read_text())
    pts = meta["points"]
    assert len(pts) == 2
    assert pts[0]["category"] == "contrastive"
    assert pts[0]["Sycophancy rate"] == 0.7
    assert pts[0]["error"] == pytest.approx(0.04, abs=1e-6)
    assert all(p["_kind"] == "bar" for p in pts)


def test_sidecar_multi_series_tags_group(tmp_path: Path) -> None:
    """Co-plotted scatter + line get distinct `_group` indices."""
    set_paper_style("blog")
    fig, ax = plt.subplots()
    ax.scatter([0.1, 0.2, 0.3], [1.0, 2.0, 3.0])
    ax.plot([0.1, 0.2, 0.3], [1.1, 2.1, 3.1], label="fit")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    written = savefig_paper(fig, "multi", dir=tmp_path)
    plt.close(fig)

    meta = json.loads(written["meta"].read_text())
    groups = {p["_group"] for p in meta["points"]}
    assert groups == {0, 1}
    assert meta["n_series"] == 2
    # The line series carries its legend label.
    line_rows = [p for p in meta["points"] if p["_kind"] == "line"]
    assert all(p["series"] == "fit" for p in line_rows)


def test_sidecar_embed_data_opt_out(tmp_path: Path) -> None:
    """embed_data=False writes a provenance-only sidecar (no `points`)."""
    set_paper_style("blog")
    fig, ax = plt.subplots()
    ax.scatter([1, 2, 3], [4, 5, 6])
    written = savefig_paper(fig, "noembed", dir=tmp_path, embed_data=False)
    plt.close(fig)

    meta = json.loads(written["meta"].read_text())
    assert set(meta.keys()) == {"commit", "created", "figsize"}
    assert "points" not in meta


def test_sidecar_imshow_falls_back_to_provenance_only(tmp_path: Path) -> None:
    """A figure with no extractable point data keeps the provenance-only sidecar."""
    import numpy as np

    set_paper_style("neurips")
    fig, ax = plt.subplots()
    ax.imshow(np.arange(9).reshape(3, 3))  # heatmap — no scatter/line/bar artists
    written = savefig_paper(fig, "heat", dir=tmp_path)
    plt.close(fig)

    meta = json.loads(written["meta"].read_text())
    assert "points" not in meta
    assert {"commit", "created", "figsize"} == set(meta.keys())


def _reject_js_invalid_constants(c: str) -> object:
    """parse_constant hook that mimics JS `JSON.parse`: reject NaN/Infinity."""
    raise ValueError(f"JS-invalid JSON constant: {c}")


def test_sidecar_nan_coordinate_serializes_as_json_null(tmp_path: Path) -> None:
    """A NaN/Inf coordinate must serialize as JSON `null`, never bare `NaN`.

    Python's json.dumps default (allow_nan=True) writes bare `NaN`/`Infinity`,
    which the dashboard's JS `JSON.parse` (dashboard/lib/task-data.ts) REJECTS —
    silently degrading the figure to provenance-only. The fix sanitizes non-
    finite cells to None (→ null). Verify: no `NaN`/`Infinity` literal in the
    text AND it round-trips through a JS-strict parse AND the bad cell is null
    with the row surviving.
    """
    import math

    set_paper_style("blog")
    fig, ax = plt.subplots()
    # One point has a NaN x (masked cell); another has an inf y.
    ax.scatter([0.1, math.nan, 0.5], [1.0, 2.0, math.inf])
    ax.set_xlabel("predictor")
    ax.set_ylabel("outcome")
    written = savefig_paper(fig, "nan_scatter", dir=tmp_path)
    plt.close(fig)

    text = written["meta"].read_text()
    # 1) No bare JS-invalid literals in the serialized sidecar.
    assert "NaN" not in text
    assert "Infinity" not in text
    # 2) Round-trips through a JS-strict parse (what the viewer's JSON.parse does).
    meta = json.loads(text, parse_constant=_reject_js_invalid_constants)
    # 3) The bad cells came through as null; all 3 rows survived.
    pts = meta["points"]
    assert len(pts) == 3
    assert pts[1]["predictor"] is None  # NaN x → null
    assert pts[2]["outcome"] is None  # inf y → null
    assert pts[0]["predictor"] == 0.1 and pts[0]["outcome"] == 1.0  # good row intact


def test_sidecar_bar_nan_height_serializes_as_json_null(tmp_path: Path) -> None:
    """A NaN bar height / error must serialize as JSON `null`, not bare `NaN`."""
    import math

    set_paper_style("blog")
    fig, ax = plt.subplots()
    ax.bar(["a", "b"], [0.5, math.nan], yerr=[0.05, math.nan])
    ax.set_ylabel("rate")
    written = savefig_paper(fig, "nan_bar", dir=tmp_path)
    plt.close(fig)

    text = written["meta"].read_text()
    assert "NaN" not in text and "Infinity" not in text
    meta = json.loads(text, parse_constant=_reject_js_invalid_constants)
    pts = meta["points"]
    assert pts[1]["rate"] is None
    assert pts[0]["rate"] == 0.5
