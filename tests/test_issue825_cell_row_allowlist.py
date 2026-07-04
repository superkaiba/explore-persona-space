"""Regression guard for issue825_fit_cells --cell-row-allowlist (onpolicy-user-turn).

Pins the plan-MF-A contract: flag absent => byte-identical legacy behavior
(the no-allowlist path returns the SAME xy object untouched, and a
full-coverage allowlist reproduces the no-flag fit numbers exactly); the
allowlist subsets rows BEFORE fold assignment and fails loud on ids missing
from the bundle.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue825_fit_cells as fc  # noqa: E402


def _fake_xy(n: int = 10, layers: int = 3, dim: int = 4) -> dict:
    rng = np.random.default_rng(0)
    return {
        "X": rng.normal(size=(n, layers, dim)).astype(np.float32),
        "Y": rng.normal(size=(n, layers, dim)).astype(np.float32),
        "conv_ids": np.asarray([str(i) for i in range(n)]),
        "nll": rng.uniform(1, 3, size=n).astype(np.float32),
    }


def test_no_flag_returns_same_object_untouched():
    """allowlist=None must be the identity — the legacy path is not even copied."""
    xy = _fake_xy()
    out = fc._apply_row_allowlist(xy, None, "cell")
    assert out is xy


def test_allowlist_subsets_rows_and_matches_ids():
    xy = _fake_xy(n=10)
    out = fc._apply_row_allowlist(xy, ["1", "3", "7"], "cell")
    assert list(out["conv_ids"]) == ["1", "3", "7"]
    assert out["X"].shape[0] == 3 and out["Y"].shape[0] == 3 and out["nll"].shape[0] == 3
    np.testing.assert_array_equal(out["X"], xy["X"][[1, 3, 7]])
    # int-typed allowlist ids match str-typed bundle ids (JSON vs sidecar drift)
    out2 = fc._apply_row_allowlist(xy, [1, 3, 7], "cell")
    np.testing.assert_array_equal(out2["X"], out["X"])


def test_allowlist_missing_id_fails_loud():
    xy = _fake_xy(n=5)
    with pytest.raises(AssertionError, match="allowlist/bundle drift"):
        fc._apply_row_allowlist(xy, ["3", "99"], "cell")


def test_full_coverage_allowlist_reproduces_no_flag_fit(tmp_path):
    """run_cell with an all-ids allowlist == run_cell with no allowlist, numerically.

    Proves the flag introduces no numeric perturbation: fold assignment is a
    deterministic permutation of the identical conv_id set, so a full-coverage
    allowlist and the legacy no-flag path must emit identical r2 tables.
    """
    ts = tmp_path / "ts"
    fc._fabricate_smoke_turnstore(ts, n=24, dim=8)
    cell = {
        "cell_id": "M_instruct_assistant_chat",
        "model": "instruct",
        "role": "assistant",
        "format": "chat",
    }
    out_a = tmp_path / "out_noflag"
    out_b = tmp_path / "out_full_allowlist"
    kw = {"n_folds": 3, "seed": 0, "null_draws": 2, "n_boot": 20}
    fc.run_cell(dict(cell), ts, out_a, **kw)
    all_ids = [f"smoke_{i:03d}" for i in range(24)]
    fc.run_cell(dict(cell), ts, out_b, allowlist=all_ids, **kw)
    a = json.loads((out_a / "cells_M_instruct_assistant_chat.json").read_text())
    b = json.loads((out_b / "cells_M_instruct_assistant_chat.json").read_text())
    assert a["r2_per_layer_obs"] == b["r2_per_layer_obs"]
    assert a["selection_symmetric"] == b["selection_symmetric"]
    assert a["cosine_frozen_layers"] == b["cosine_frozen_layers"]
    assert a["y_trace_cov_frozen"] == b["y_trace_cov_frozen"]
    assert a["row_allowlist_applied"] is False and b["row_allowlist_applied"] is True
