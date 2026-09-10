"""Rendering checks for baseline bars, axis cuts, and interval fidelity."""

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


def test_baseline_bars_align_and_axis_cuts_preserve_endpoints(tmp_path):
    data = _data()
    baselines = data[-1]
    fig, _ = plotter.make_figure(*data)
    try:
        axes = {ax.get_label(): ax for ax in fig.axes}
        labels, retrieval = [axes[k] for k in ("baseline-labels", "baseline-top1")]
        r2_axes = [axes[f"baseline-r2-{j}"] for j in range(len(plotter.BASELINE_R2_SEGMENTS))]
        fig.savefig(tmp_path / "baselines.pdf")
        keys = {arm["key"] for arm in baselines["arms"]}
        assert not keys.intersection({"floor_e5", "enc_e5", "identity_copy"})
        assert "identity_bias" in keys
        for i, arm in enumerate(baselines["arms"]):
            text = next(t for t in labels.texts if t.get_text() == arm["label"])
            row_y = text.get_transform().transform(text.get_position())[1]
            assert row_y == pytest.approx(retrieval.transData.transform((0, i))[1])
            for ax in r2_axes:
                assert row_y == pytest.approx(ax.transData.transform((0, i))[1])
                bar = ax.patches[i]
                assert bar.get_x() == 0.0
                assert bar.get_width() == arm["r2"]
                assert bar.get_y() + bar.get_height() / 2 == pytest.approx(i)
                assert bar.get_clip_on()
            bar = retrieval.patches[i]
            assert bar.get_x() == 0.0
            assert bar.get_width() == arm["top1"]
            assert bar.get_y() + bar.get_height() / 2 == pytest.approx(i)
            # No endpoint may be lost inside an omitted interval or outside the figure.
            assert sum(lo <= arm["r2"] <= hi for lo, hi in plotter.BASELINE_R2_SEGMENTS) == 1
            interval = retrieval.collections[2 * i].get_segments()[0]
            np.testing.assert_allclose(interval[:, 0], arm["top1_ci95"])
        scales = []
        for ax, limits in zip(r2_axes, plotter.BASELINE_R2_SEGMENTS, strict=True):
            assert ax.get_xlim() == limits
            assert not any(isinstance(t, Annotation) for t in ax.texts)
            assert not any("outside plotted range" in t.get_text() for t in ax.texts)
            scales.append(ax.transData.transform((1, 0))[0] - ax.transData.transform((0, 0))[0])
        np.testing.assert_allclose(scales, scales[0])
        top_axes = [ax for ax in fig.axes if ax not in (labels, retrieval, *r2_axes)]
        assert all(ax.get_position().y0 > r2_axes[0].get_position().y1 for ax in top_axes)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("score", [-3.0, -2.0, -0.4, 1.1])
def test_axis_cuts_reject_hidden_endpoints(score):
    data = _data()
    data[-1]["arms"][0]["r2"] = score
    try:
        with pytest.raises(ValueError, match="outside visible segments"):
            plotter.make_figure(*data)
    finally:
        plt.close("all")


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
