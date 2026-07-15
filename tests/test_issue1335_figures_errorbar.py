"""Pinned regression test for the #1335 round-4 crash-fix (matplotlib xerr offsets).

GCP attempt att-20260715-122509 died at ``[phase=p4_figures_smoke]`` with
``ValueError: 'xerr' must not contain negative values``: ``fig_waterfall``
passed signed CI-bound differences as ``xerr``. matplotlib takes NON-NEGATIVE
per-point offsets; at tiny smoke n the bootstrap CI can invert around the
point estimate, and production has first-class negative deltas (plan §3).

Pins (fail pre-fix, pass post-fix):
  (a) ``_ci_offsets`` clamps element-wise to non-negative offsets;
  (b) ``fig_waterfall`` renders (savefig) a summary containing a negative
      delta AND inverted CIs without ``ValueError``;
  (c) an empty deltas dict still renders (empty-cell rung edge).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

f1335 = pytest.importorskip("issue1335_figures")


def _summary(deltas: dict) -> dict:
    return {
        "per_model": {
            "base": {
                "deltas": deltas,
                "gap": {"G": {"value": -0.1, "ci_lo": -0.3, "ci_hi": 0.2}},
            }
        }
    }


def test_ci_offsets_clamps_inverted_bounds_elementwise():
    vals = [-0.5, 1.0, 0.0]
    lo = [-0.3, 0.5, -0.1]  # first entry INVERTED: lo > value
    hi = [-0.7, 1.5, 0.1]  # first entry INVERTED: hi < value
    lo_off, hi_off = f1335._ci_offsets(vals, lo, hi)
    assert np.all(lo_off >= 0.0) and np.all(hi_off >= 0.0)
    assert lo_off[0] == 0.0 and hi_off[0] == 0.0  # inverted CI clamps to 0
    assert lo_off[1] == pytest.approx(0.5)
    assert hi_off[1] == pytest.approx(0.5)
    assert lo_off[2] == pytest.approx(0.1)
    assert hi_off[2] == pytest.approx(0.1)


def test_ci_offsets_propagates_nan_without_negatives():
    lo_off, hi_off = f1335._ci_offsets([0.5], [np.nan], [np.nan])
    # NaN bounds propagate (matplotlib renders no bar for NaN, no crash);
    # they must never turn into negative offsets.
    assert np.isnan(lo_off[0]) and np.isnan(hi_off[0])


def test_fig_waterfall_renders_negative_delta_and_inverted_ci(tmp_path):
    deltas = {
        # negative delta, well-formed CI (production shape — plan §3 raw negative deltas)
        "label": {"value": -0.25, "ci_lo": -0.40, "ci_hi": -0.10},
        # inverted CI on BOTH sides (tiny-smoke-n bootstrap): lo > value, hi < value
        "header": {"value": 0.05, "ci_lo": 0.10, "ci_hi": -0.02},
        # one-sided inversion only (hi < value)
        "foils": {"value": 1.0, "ci_lo": 0.5, "ci_hi": 0.9},
    }
    (tmp_path / "ladder_summary.json").write_text(json.dumps(_summary(deltas)))
    args = SimpleNamespace(out_dir=tmp_path, fig_dir=tmp_path)
    # Pre-fix: ValueError("'xerr' must not contain negative values") here.
    f1335.fig_waterfall(args, ["base"])
    assert (tmp_path / "delta_waterfall.png").exists()


def test_fig_waterfall_renders_empty_deltas(tmp_path):
    (tmp_path / "ladder_summary.json").write_text(json.dumps(_summary({})))
    args = SimpleNamespace(out_dir=tmp_path, fig_dir=tmp_path)
    f1335.fig_waterfall(args, ["base"])
    assert (tmp_path / "delta_waterfall.png").exists()
