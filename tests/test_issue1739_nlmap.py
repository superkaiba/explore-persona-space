"""Nonlinear context->answer map round (#1739): MapFit dispatch + CLI parity.

Covers the seams the nonlinear-map round adds on top of the reviewed #1739
pipeline:

- ``MapFit`` kind validation (a nonlinear kind without payloads, and a linear
  kind without ``w``, both fail LOUD rather than producing a silent wrong map).
- ``apply_map`` dispatches on ``kind``; the linear path is untouched, and the
  shuffled-weight override is REFUSED on a nonlinear map instead of silently
  ignored (arm 13 has no nonlinear analogue this round).
- ``fit_nonlinear_map`` executes the REAL #779 N1M fitter bodies for BOTH
  kinds on a tiny real-shape pool (no seam stubs), and its frozen payload
  round-trips through the same ``apply_map`` predict path the arms use.
- The diagnostics holdout is the IDENTICAL split ``fit_linear_map`` draws, so
  linear-vs-nonlinear R2 is a cell-for-cell comparison.
- The CLI ``--map-kind`` choices stay in parity with
  ``fits.NONLINEAR_MAP_KINDS`` (the choices tuple is a literal because every
  ``fits`` import in the script is deferred).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from explore_persona_space.experiments.issue_1739 import fits

REPO_ROOT = Path(__file__).resolve().parents[1]


def _pool(n=40, n_layers=2, d=6, seed=0):
    """Tiny real-shape (Ly, n, d) U pool with genuine x->y structure."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_layers, n, d))
    # a nonlinear-but-learnable target so a fitted map beats a constant
    y = np.tanh(x * 1.5) + 0.05 * rng.normal(size=x.shape)
    return x, y


# --------------------------------------------------------------------------
# MapFit validation + apply_map dispatch
# --------------------------------------------------------------------------


def test_mapfit_linear_requires_w():
    with pytest.raises(ValueError, match="requires w"):
        fits.MapFit(w=None, x_mu=None, x_sd=None, y_mu=None, diagnostics={})


def test_mapfit_nonlinear_requires_payloads():
    for kind in fits.NONLINEAR_MAP_KINDS:
        with pytest.raises(ValueError, match="requires nl_payloads"):
            fits.MapFit(w=None, x_mu=None, x_sd=None, y_mu=None, diagnostics={}, kind=kind)


def test_mapfit_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown MapFit kind"):
        fits.MapFit(
            w=None,
            x_mu=None,
            x_sd=None,
            y_mu=None,
            diagnostics={},
            kind="quadratic",
            nl_payloads=({},),
        )


def test_apply_map_linear_path_unchanged():
    """The pre-existing linear contract is byte-identical after the dispatch."""
    x, y = _pool()
    m = fits.fit_linear_map(x, y, seed=3)
    assert m.kind == "linear"
    manual = ((x - m.x_mu) / m.x_sd) @ m.w + m.y_mu
    assert np.allclose(fits.apply_map(x, m), manual, atol=1e-10)


def test_apply_map_refuses_shuffled_weights_on_nonlinear():
    x, _ = _pool(n=24, n_layers=1, d=4)
    m = fits.fit_nonlinear_map(*_pool(n=24, n_layers=1, d=4), kind="kernel")
    with pytest.raises(ValueError, match="linear-only"):
        fits.apply_map(x, m, w=np.zeros((1, 4, 4)))


# --------------------------------------------------------------------------
# real-body fits for BOTH kinds (no seam stubs)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", list(fits.NONLINEAR_MAP_KINDS))
def test_fit_nonlinear_map_real_body_and_payload_roundtrip(kind):
    """Executes the REAL N1M fitter body, then applies the frozen payload.

    This is the production-body test for the fitters the round reuses: no
    monkeypatched seams, real torch, real payload -> real ``apply_map``.
    """
    x, y = _pool(n=48, n_layers=2, d=6, seed=1)
    m = fits.fit_nonlinear_map(x, y, kind=kind, seed=0)

    assert m.kind == kind
    assert m.w is None and m.x_mu is None  # nonlinear carries no weight tensor
    assert len(m.nl_payloads) == x.shape[0]
    for p in m.nl_payloads:
        assert p, "fitter returned an empty capture payload"
    # the payload kind is the N1M tag the shared apply_map dispatches on
    expected_tag = {"mlp": "mlp", "kernel": "krr_nystrom"}[kind]
    assert all(p["kind"] == expected_tag for p in m.nl_payloads)

    # diagnostics carry the standing mapping-baselines pair per layer
    per_layer = m.diagnostics["per_layer"]
    assert len(per_layer) == x.shape[0]
    for row in per_layer:
        assert "r2_map" in row and "r2_identity_bias" in row
        assert set(row["knn"]) == {"euclidean", "cosine"}
    assert m.diagnostics["map_kind"] == kind
    assert m.diagnostics["w_refit_on_full_u"] is True
    assert m.diagnostics["w_fit_rows"] == x.shape[1]

    # frozen payload applies through the arms' own path, right shape, finite
    pred = fits.apply_map(x, m)
    assert pred.shape == x.shape
    assert np.isfinite(pred).all()


def test_nonlinear_holdout_matches_linear_holdout():
    """Comparability invariant: same held-out rows as the linear map."""
    n, seed = 50, 7
    hold, tr = fits._nl_split(n, 0.2, seed)
    # reproduce fit_linear_map's own split arithmetic
    rng = np.random.default_rng([1739, 4, seed])
    perm = rng.permutation(n)
    n_hold = max(2, round(0.2 * n))
    assert np.array_equal(hold, perm[:n_hold])
    assert np.array_equal(tr, perm[n_hold:])
    assert len(set(hold) & set(tr)) == 0


def test_refit_full_false_uses_split_fit_rows():
    x, y = _pool(n=40, n_layers=1, d=5, seed=2)
    m = fits.fit_nonlinear_map(x, y, kind="mlp", refit_full=False)
    assert m.diagnostics["w_refit_on_full_u"] is False
    assert m.diagnostics["w_fit_rows"] == m.diagnostics["n_train"] < x.shape[1]


def test_apply_nl_map_rejects_layer_count_mismatch():
    x, y = _pool(n=24, n_layers=2, d=4, seed=4)
    m = fits.fit_nonlinear_map(x, y, kind="mlp")
    with pytest.raises(ValueError, match="!= n_layers"):
        fits.apply_map(x[:1], m)  # 1 layer of x vs 2 payloads


# --------------------------------------------------------------------------
# CLI parity + the arms seam
# --------------------------------------------------------------------------


def test_cli_map_kind_choices_match_fits():
    """The literal argparse choices tuple must track fits.NONLINEAR_MAP_KINDS."""
    src = (REPO_ROOT / "scripts" / "issue1739_fits.py").read_text(encoding="utf-8")
    block = src.split('"--map-kind"', 1)[1].split(")", 1)[0]
    for kind in fits.NONLINEAR_MAP_KINDS:
        assert f'"{kind}"' in block, f"--map-kind choices missing {kind!r}"
    assert '"linear"' in block


def test_synthetic_smoke_runs_for_every_map_kind(tmp_path):
    """The script's own synthetic e2e (arms 6/7/8 included) under each kind."""
    for kind in ("linear", *fits.NONLINEAR_MAP_KINDS):
        out = tmp_path / kind
        proc = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "issue1739_fits.py"),
                "--synthetic",
                "60",
                "--synthetic-dim",
                "6",
                "--synthetic-layers",
                "2",
                "--map-kind",
                kind,
                "--out-root",
                str(out),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert proc.returncode == 0, f"{kind} synthetic run failed:\n{proc.stderr[-3000:]}"
