"""Rendering checks for baseline alignment, off-scale scores, and interval fidelity."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.text import Annotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import make_paper_figure2 as plotter  # noqa: E402


def _data():
    """Load the same banked inputs as the production entry point."""
    extension = plotter._load_extension_data(plotter.DEFAULT_EXTENSION_SOURCE)
    baselines = plotter._load_baselines_data(plotter.DEFAULT_BASELINES_SOURCE, extension)
    layer = plotter._load_layer_data(plotter.DEFAULT_LAYER_SOURCE)
    scaling = plotter._load_scaling_data(plotter.DEFAULT_SCALING_SOURCE, extension)
    boundary = plotter._load_boundary_data(plotter.DEFAULT_BOUNDARY_SOURCE)
    return layer, scaling, boundary, extension, baselines


def test_baseline_rows_and_offscale_values(tmp_path):
    data = _data()
    baselines = data[-1]
    fig, _ = plotter.make_figure(*data)
    try:
        axes = {ax.get_label(): ax for ax in fig.axes}
        labels, retrieval, r2 = [
            axes[k] for k in ("baseline-labels", "baseline-top1", "baseline-r2")
        ]
        fig.savefig(tmp_path / "baselines.pdf")
        assert r2.get_xlim() == (-0.1, 1.0)
        assert all(arm["key"] != "floor_e5" for arm in baselines["arms"])
        for i, arm in enumerate(baselines["arms"]):
            text = next(t for t in labels.texts if t.get_text() == arm["label"])
            row_y = text.get_transform().transform(text.get_position())[1]
            assert row_y == pytest.approx(retrieval.transData.transform((0, i))[1])
            assert row_y == pytest.approx(r2.transData.transform((0, i))[1])
            interval = retrieval.collections[2 * i].get_segments()[0]
            np.testing.assert_allclose(interval[:, 0], arm["top1_ci95"])
        arrows = [t for t in r2.texts if isinstance(t, Annotation)]
        assert len(arrows) == 2
        assert all(a.xy[0] < a.xyann[0] for a in arrows)
        notes = [t.get_text() for t in r2.texts if "outside plotted range" in t.get_text()]
        assert notes == ["\u22120.92  outside plotted range", "\u22122.70  outside plotted range"]
        dots = [line for line in r2.lines if line.get_marker() != "None"]
        np.testing.assert_allclose(
            [line.get_xdata()[0] for line in dots],
            [arm["r2"] for arm in baselines["arms"] if arm["r2"] >= -0.1],
        )
        top_axes = [ax for ax in fig.axes if ax not in (labels, retrieval, r2)]
        assert all(ax.get_position().y0 > r2.get_position().y1 for ax in top_axes)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("offsets", [(-0.02, -0.01), (0.01, 0.02)])
def test_percentile_interval_excluding_estimate_keeps_endpoints(tmp_path, offsets):
    data = _data()
    arm = data[-1]["arms"][0]
    bounds = [arm["top1"] + offset for offset in offsets]
    arm["top1_ci95"] = bounds
    fig, _ = plotter.make_figure(*data)
    try:
        fig.savefig(tmp_path / "nonenclosing-interval.pdf")
        retrieval = next(ax for ax in fig.axes if ax.get_label() == "baseline-top1")
        segment = retrieval.collections[0].get_segments()[0]
        np.testing.assert_allclose(segment[:, 0], bounds)
    finally:
        plt.close(fig)
