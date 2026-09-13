"""Learning-curve intervals preserve endpoints and disclose undefined uncertainty."""

import runpy
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest


@pytest.fixture
def renderer(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    sys.modules.pop("workspace_jr_plot", None)
    return runpy.run_path(str(scripts / "workspace_jr_plot_supplement.py"))


def test_vertical_interval_preserves_percentile_endpoints(renderer):
    fig, ax = plt.subplots()
    renderer["vertical_interval"](
        ax,
        256,
        {"estimate": 0.8, "interval": [0.2, 0.6], "confidence": 0.95, "n_draws": 2000},
        renderer["STYLES"]["ridge"],
    )
    np.testing.assert_array_equal(ax.collections[0].get_segments()[0], [[256, 0.2], [256, 0.6]])
    assert ax.lines[0].get_ydata()[0] == 0.8
    plt.close(fig)


@pytest.mark.parametrize("point", [None, 0.4])
def test_missing_uncertainty_is_visible(renderer, point):
    fig, ax = plt.subplots()
    renderer["vertical_interval"](
        ax,
        256,
        {"estimate": point, "interval": None, "confidence": 0.95, "n_draws": 2000},
        renderer["STYLES"]["ridge"],
    )
    assert len(ax.lines) == int(point is not None)
    assert any("undefined" in label.get_text() for label in ax.texts)
    plt.close(fig)
