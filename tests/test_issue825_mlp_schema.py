"""#1320 schema pins for the batched ``run_mlp_secondary`` cells JSON block.

Byte-compat contract (plan #1320 acceptance criterion 3): after a tiny
``run_mlp_secondary`` call, ``payload["mlp"]`` holds exactly
``{str(layer): {r2_obs: float, r2_null: list, r2_obs_folds: list,
budget_hit_folds: list}}`` with ``len(r2_null) == n_null``, plus top-level
``mlp_budget_exhausted: bool`` — the three named downstream readers
(issue825_onpolicy_summarize.py, the mlp-followup gate, the onpolicy dispatch
presence check) parse unchanged.

Budget-exhaustion branch (D1.4 refinement): a monkeypatched
``MLP_TIME_BUDGET_S = 0`` yields NaN pooled reads for EVERY layer, full-length
NaN-padded ``r2_null``, populated ``budget_hit_folds`` (the remaining fold
ids, identical across layers), and ``mlp_budget_exhausted=True`` — and the
mlp-followup gate's extraction path (``_mlp_extract``/``max(obs)`` shape,
scripts/issue825_mlp_followup_dispatch.sh) is NaN-safe against that shape.

CPU-only, tiny shapes; writes only to tmp_path.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import issue825_fit_cells as fc

N_FOLDS = 3
N_NULL = 2


def _tiny_res(n: int = 24, n_layers: int = 15, d: int = 16, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_layers, d)).astype(np.float32)
    Y = (X * 0.5 + 0.1 * rng.standard_normal((n, n_layers, d))).astype(np.float32)
    conv_ids = np.array([f"c{i:03d}" for i in range(n)])
    return {"xy": {"X": X, "Y": Y, "conv_ids": conv_ids}}


def _run(tmp_path: Path) -> dict:
    fc.run_mlp_secondary(
        _tiny_res(), tmp_path, cell_id="schema", n_folds=N_FOLDS, seed=0, n_null=N_NULL
    )
    return json.loads((tmp_path / "cells_schema.json").read_text())


def _followup_gate_mlp_extract(mlp):
    """Verbatim replica of the mlp-followup gate's ``_mlp_extract`` walk
    (scripts/issue825_mlp_followup_dispatch.sh heredoc — not importable)."""
    obs: list[float] = []
    nulls: list[float] = []

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "r2_obs" and isinstance(v, (int, float)):
                    obs.append(float(v))
                elif isinstance(v, list) and "null" in str(k).lower():
                    nulls.extend(float(x) for x in v if isinstance(x, (int, float)))
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(mlp)
    return obs, nulls


def test_mlp_block_schema_keys_and_types(tmp_path):
    payload = _run(tmp_path)
    assert isinstance(payload["mlp_budget_exhausted"], bool)
    assert payload["mlp_budget_exhausted"] is False
    mlp = payload["mlp"]
    expected_layers = {str(li) for li in fc.FROZEN_LAYERS if li < 15}
    assert set(mlp) == expected_layers and mlp, (set(mlp), expected_layers)
    for li, blk in mlp.items():
        assert set(blk) == {"r2_obs", "r2_null", "r2_obs_folds", "budget_hit_folds"}, (li, blk)
        assert isinstance(blk["r2_obs"], float), (li, blk)
        assert isinstance(blk["r2_null"], list) and len(blk["r2_null"]) == N_NULL, (li, blk)
        assert all(isinstance(v, float) for v in blk["r2_null"]), (li, blk)
        assert isinstance(blk["r2_obs_folds"], list), (li, blk)
        assert len(blk["r2_obs_folds"]) == N_FOLDS, (li, blk)  # no skipped folds here
        assert all(isinstance(v, float) for v in blk["r2_obs_folds"]), (li, blk)
        assert blk["budget_hit_folds"] == [], (li, blk)


def test_mlp_block_merges_into_existing_cells_json(tmp_path):
    """run_mlp_secondary folds into an EXISTING cells JSON (the production shape)."""
    pre = {"metadata": {"n": 24}, "r2": {"obs": [0.1]}}
    (tmp_path / "cells_schema.json").write_text(json.dumps(pre))
    payload = _run(tmp_path)
    assert payload["metadata"] == {"n": 24}
    assert payload["r2"] == {"obs": [0.1]}
    assert "mlp" in payload and "mlp_budget_exhausted" in payload


def test_budget_exhaustion_branch_and_followup_gate_nan_safety(tmp_path, monkeypatch):
    monkeypatch.setattr(fc, "MLP_TIME_BUDGET_S", 0)
    payload = _run(tmp_path)
    assert payload["mlp_budget_exhausted"] is True
    mlp = payload["mlp"]
    expected_layers = {str(li) for li in fc.FROZEN_LAYERS if li < 15}
    assert set(mlp) == expected_layers  # NaN-keyed blocks for EVERY layer
    for li, blk in mlp.items():
        assert math.isnan(blk["r2_obs"]), (li, blk)
        assert len(blk["r2_null"]) == N_NULL and all(math.isnan(v) for v in blk["r2_null"]), (
            li,
            blk,
        )
        assert blk["r2_obs_folds"] == [], (li, blk)  # no fold completed at budget 0
        assert blk["budget_hit_folds"] == list(range(N_FOLDS)), (li, blk)
    # The mlp-followup gate's extraction path is NaN-safe against this shape:
    # the gate's `if not mlp` coverage check passes (blocks present), obs is
    # non-empty (NaN floats count), and max(obs) does not raise.
    obs, nulls = _followup_gate_mlp_extract(mlp)
    assert obs and len(obs) == len(expected_layers), obs
    assert nulls and len(nulls) == N_NULL * len(expected_layers), nulls
    max_r2 = max(obs)  # must not raise (the gate computes exactly this)
    assert isinstance(max_r2, float)
