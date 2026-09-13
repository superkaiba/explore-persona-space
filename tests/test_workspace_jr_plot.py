"""Scientific interval endpoints and undefined estimates must remain faithful."""

import runpy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/workspace_jr_plot.py"))


def record(point, interval):
    return {"estimate": point, "interval": interval, "confidence": 0.95, "n_draws": 2000}


def test_percentile_interval_need_not_contain_point():
    fig, ax = plt.subplots()
    MODULE["interval_point"](ax, 0, record(0.8, [0.2, 0.6]), MODULE["STYLES"]["ridge"])
    np.testing.assert_array_equal(ax.collections[0].get_segments()[0], [[0.2, 0], [0.6, 0]])
    assert ax.lines[0].get_xdata()[0] == 0.8
    plt.close(fig)


def test_undefined_point_is_visible_without_a_zero_marker():
    fig, ax = plt.subplots()
    MODULE["interval_point"](ax, 0, record(None, None), MODULE["STYLES"]["ridge"])
    assert len(ax.lines) == 0
    assert [text.get_text() for text in ax.texts] == ["undefined"]
    plt.close(fig)


@pytest.mark.parametrize("interval", [[0.7, 0.2], [0.2, float("inf")]])
def test_invalid_interval_fails(interval):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="Invalid interval"):
        MODULE["interval_point"](ax, 0, record(0.4, interval), MODULE["STYLES"]["ridge"])
    plt.close(fig)
