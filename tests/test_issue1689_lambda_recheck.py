"""wider-lambda-ceilings follow-up (#1689 R17) — lambdas-passthrough contract.

Pins the round's source-module change on scripts/issue825_fit_cells.py (the
inner-group-cv path previously hard-asserted ``lambdas is None``):

  1. Default ``lambdas=None`` is BYTE-IDENTICAL to the pre-change behavior
     (and to an explicit 13-grid pass) under inner-group-cv selection.
  2. A custom wide grid (logspace(-2,7,19), a strict superset) is accepted on
     BOTH the serial observed path and the batched null path.
  3. Malformed grids (descending / negative / 2-D) are rejected loudly.
  4. The <2-usable-inner-folds GCV fallback still works under a custom grid.
  5. fit_ladder checkpoint regime meta pins lambda_grid with legacy
     back-compat (missing key reads as ladder13; wide19 never resumes it).
  6. Driver checkpoint-regime + realized-enumeration helpers.

Synthetic tiny tensors only — no HF / network access.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.issue825_fit_cells import (  # noqa: E402
    LAMBDAS,
    _validate_lambda_grid,
    heldout_r2_sweep,
)
from scripts.issue1689_common import LAMBDA_GRIDS, resolve_lambda_grid  # noqa: E402


def _toy(n=60, layers=1, dim=10, n_groups=12, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), n // n_groups)[:n].astype(str)
    X = rng.standard_normal((n, layers, dim)).astype(np.float32)
    W = (rng.standard_normal((layers, dim, dim)) * 0.4).astype(np.float32)
    Y = np.einsum("nld,lde->nle", X, W).astype(np.float32)
    Y += (rng.standard_normal((n, layers, dim)) * 0.2).astype(np.float32)
    return X, Y, groups


_KW = dict(
    n_folds=3, seed=42, null_draws=0, collect_lambdas=True, lambda_selection="inner-group-cv"
)


def test_default_none_and_explicit_13_grid_identical():
    """Contract 1: default (None) == explicit 13-grid, bit-identical (r2 AND
    selected lambdas) — the published percell path is untouched."""
    X, Y, g = _toy()
    s_def = heldout_r2_sweep(X, Y, g, **_KW)
    s_13 = heldout_r2_sweep(X, Y, g, lambdas=LAMBDAS.copy(), **_KW)
    assert np.array_equal(s_def["r2_obs"], s_13["r2_obs"])
    assert np.array_equal(s_def["gcv_lambda"], s_13["gcv_lambda"])
    assert np.isfinite(s_def["r2_obs"]).all()


def test_wide19_superset_accepted_serial_and_batched():
    """Contract 2: the wide grid runs on the observed (serial) AND null
    (batched) inner-group-cv paths; the grid is a strict superset of the
    13-grid, and selections stay within the grid."""
    g13 = resolve_lambda_grid("ladder13")
    g19 = resolve_lambda_grid("wide19")
    assert np.array_equal(g13, LAMBDAS)
    assert np.intersect1d(g13, g19).size == 13
    assert np.array_equal(g19[: len(g13)], g13)
    assert set(LAMBDA_GRIDS) == {"ladder13", "wide19"}

    X, Y, g = _toy()
    kw = dict(_KW)
    kw["null_draws"] = 3  # exercises _ridge_predict_cached_batched's inner branch
    s_19 = heldout_r2_sweep(X, Y, g, lambdas=g19, **kw)
    assert np.isfinite(s_19["r2_obs"]).all()
    assert s_19["r2_null"].shape == (3, 1)
    assert np.isfinite(s_19["r2_null"]).all()
    sel = [v for v in np.asarray(s_19["gcv_lambda"]).ravel() if np.isfinite(v)]
    assert sel and all(any(np.isclose(v, gv) for gv in g19) for v in sel)


@pytest.mark.parametrize(
    "bad",
    [
        resolve_lambda_grid("wide19")[::-1],  # descending
        -resolve_lambda_grid("ladder13"),  # negative
        np.ones((2, 3)),  # 2-D
        np.array([1.0, 1.0, 2.0]),  # non-strictly-ascending
    ],
)
def test_malformed_grids_rejected(bad):
    """Contract 3: malformed custom grids raise ValueError before any fit."""
    X, Y, g = _toy(n=30, n_groups=6)
    with pytest.raises(ValueError):
        heldout_r2_sweep(X, Y, g, lambdas=bad, **_KW)
    with pytest.raises(ValueError):
        _validate_lambda_grid(np.asarray(bad))


def test_gcv_fallback_and_fold_skip_under_custom_grid():
    """Contract 4: with too few groups for usable inner folds the per-fold
    GCV fallback fires and the custom grid still threads through it; empty
    outer folds (n_folds > n_groups) exercise the skip branch, r2 finite."""
    kw = dict(
        null_draws=0,
        collect_lambdas=True,
        lambda_selection="inner-group-cv",
        lambdas=resolve_lambda_grid("wide19"),
    )
    # 2 conv groups, 2 outer folds => each outer-train holds ONE group =>
    # <2 usable inner folds => the WARN + GCV fallback branch runs, custom grid.
    X, Y, g = _toy(n=20, n_groups=2, seed=1)
    s = heldout_r2_sweep(X, Y, g, n_folds=2, seed=42, **kw)
    assert np.isfinite(s["r2_obs"]).all()
    sels = {v for row in s["lambda_selector"] for v in row if v is not None}
    assert "gcv-fallback" in sels  # <2 usable inner group folds => fallback ran

    # 3 conv groups, 5 outer folds => folds 3/4 empty => the fold-skip branch
    # runs and r2 stays finite (skipped folds contribute nothing).
    X, Y, g = _toy(n=30, n_groups=3, seed=1)
    s = heldout_r2_sweep(X, Y, g, n_folds=5, seed=42, **kw)
    assert np.isfinite(s["r2_obs"]).all()
    skipped = [v for row in s["lambda_selector"] for v in row if v is None]
    assert skipped  # >=1 empty outer fold was skipped


def test_fit_ladder_ckpt_meta_backcompat():
    """Contract 5: lambda_grid joined the ladder checkpoint regime key;
    legacy (pre-key) checkpoints satisfy ladder13 requests, never wide19."""
    from scripts.issue1689_fit_ladder import _ckpt_meta_satisfies, _pair_ckpt_meta

    want13 = _pair_ckpt_meta("m", 19, 0, 40)
    assert want13["lambda_grid"] == "ladder13"
    legacy = {k: v for k, v in want13.items() if k != "lambda_grid"}
    assert _ckpt_meta_satisfies(legacy, want13)
    want19 = _pair_ckpt_meta("m", 19, 0, 40, "wide19")
    assert not _ckpt_meta_satisfies(legacy, want19)
    prior19 = dict(want19)
    assert _ckpt_meta_satisfies(prior19, want19)
    assert not _ckpt_meta_satisfies(prior19, want13)


def test_driver_ckpt_regime_and_enumeration(tmp_path):
    """Contract 6: driver resume predicate pins every regime key (refit13 is
    superset-satisfying); realized enumeration fails loud on unknown slugs
    and never cross-matches base/instruct filenames."""
    from scripts.issue1689_lambda_recheck import (
        KNOWN_MODEL_SLUGS,
        _ckpt_satisfies,
        _meta,
        realized_conditions,
    )

    want = _meta(19, "wide19", refit13=False)
    assert not _ckpt_satisfies(None, want)
    assert not _ckpt_satisfies({"meta": _meta(19, "ladder13", False)}, want)
    assert not _ckpt_satisfies({"meta": _meta(14, "wide19", False)}, want)
    assert _ckpt_satisfies({"meta": _meta(19, "wide19", True)}, want)  # refit13 superset
    assert not _ckpt_satisfies({"meta": _meta(19, "wide19", False)}, _meta(19, "wide19", True))

    base, instruct = KNOWN_MODEL_SLUGS
    (tmp_path / f"heldout_r2_{base}_assistant_chat.json").write_text("{}")
    (tmp_path / f"heldout_r2_{instruct}_dana_story.json").write_text("{}")
    assert realized_conditions(tmp_path, base) == ["assistant_chat"]
    assert realized_conditions(tmp_path, instruct) == ["dana_story"]
    (tmp_path / f"heldout_r2_{base}_not_a_condition.json").write_text("{}")
    with pytest.raises(ValueError):
        realized_conditions(tmp_path, base)


def test_driver_merge_verdict_arithmetic(tmp_path):
    """Merge-phase gates on a scratch universe: missing cell-arm fails loud;
    a complete universe produces the §3 verdict + affected-pairs shape."""
    import argparse

    from scripts.issue1689_lambda_recheck import KNOWN_MODEL_SLUGS, cmd_merge

    base = KNOWN_MODEL_SLUGS[0]
    pub_dir = tmp_path / "pub"
    out_dir = tmp_path / "out"
    pub_dir.mkdir()
    out_dir.mkdir()
    (pub_dir / f"heldout_r2_{base}_assistant_chat.json").write_text("{}")

    def rec(arm, delta):
        return {
            "meta": {"lambda_grid": "wide19"},
            "cell": "assistant_chat",
            "model": base,
            "arm": arm,
            "published_r2": 0.3,
            "ceiling_r2_19": 0.3 + delta,
            "delta_r2": delta,
            "lambda_star_13": [10000.0] * 5,
            "lambdas_selected_19": [1e7] * 5,
            "edge_hits_19": 5,
        }

    args = argparse.Namespace(
        lambda_grid="wide19",
        layer=19,
        published=pub_dir,
        out=out_dir,
        summary_dir=None,
    )
    (out_dir / f"{base}__assistant_chat__prefix.json").write_text(json.dumps(rec("prefix", 0.05)))
    with pytest.raises(RuntimeError, match="merge incomplete"):
        cmd_merge(args)  # context arm missing => fail loud
    (out_dir / f"{base}__assistant_chat__context.json").write_text(json.dumps(rec("context", 0.0)))
    assert cmd_merge(args) == 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["n_cell_arms"] == 2 and summary["n_moved"] == 1
    assert summary["verdict"] == "grid-limited"  # 1/2 = 0.5 >= 0.05
    assert summary["n_foldfits_at_new_ceiling"] == 10
    affected = json.loads((tmp_path / "affected_pairs.json").read_text())
    moved_pairs = affected[base]["prefix"]
    assert moved_pairs and all("assistant_chat" in p for p in moved_pairs)
    assert affected[base]["context"] == []
