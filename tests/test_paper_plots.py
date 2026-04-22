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
    proportion_ci,
    savefig_paper,
    set_paper_style,
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
    assert set(meta.keys()) == {"commit", "created", "figsize"}
    assert isinstance(meta["commit"], str) and meta["commit"]
    assert isinstance(meta["created"], str) and meta["created"].endswith("Z")
    assert len(meta["figsize"]) == 2
    assert all(isinstance(x, float) for x in meta["figsize"])

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
