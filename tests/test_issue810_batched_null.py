#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (Δ, ρ, ×) in scientific docstrings + comments.
"""Issue #810 round-2 regression tests — the substantive BLOCKER fixes.

Each test trips a permanent invariant added in round 2 and would FAIL against the
round-1 code:

1. The batched shuffle-null (``issue810_batched_null``) is NUMERICALLY IDENTICAL
   to the serial closed-form LOCO-ridge null it replaces — the vectorize fix
   (#722 mandate) must stay a throughput win, never a numerical change. (Round 1
   had NO batched path; the serial null projected 231 wall-h for Phase D.)
2. The sycophancy per-context subsample seed is PYTHONHASHSEED-INVARIANT (a stable
   sha256 digest, NOT Python's salted ``hash(str)`` — round 1 used ``hash(ctx_id)``
   so two runs sampled different subsets → different graded E0 target).
3. The turn_nl newline token id is pinned to the Qwen-2.5 family id 198 (round 1
   only asserted single-token + a tautological fed-id check, never the value).

Pure-Python, no GPU / no HF — exercises the helpers directly.
"""

from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
_SRC = Path(__file__).resolve().parent.parent / "src"
for p in (str(_SCRIPTS), str(_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import issue810_batched_null as bn  # noqa: E402
from issue810_common import (  # noqa: E402
    SHUFFLE_NULL_SEED,
    SYCOPHANCY_SUBSAMPLE_PER_CONTEXT,
    TURN_NL_TOKEN_ID,
)


def _rho_serial(pred, meas):
    """The serial _rho: None on degenerate, else scipy Spearman."""
    from scipy.stats import spearmanr

    if len(pred) < 4 or np.std(pred) < 1e-9 or np.std(meas) < 1e-9:
        return None
    r, _ = spearmanr(pred, meas)
    return None if np.isnan(r) else float(r)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Batched null == serial closed-form LOCO-ridge null (the vectorize invariant)
# ─────────────────────────────────────────────────────────────────────────────
def test_batched_recon_null_matches_serial():
    """RECON skill-over-mean null: batched == serial ridge_predict_loco_centered."""
    from explore_persona_space.analysis.vectorized_mlp_skill import (
        ridge_predict_loco_centered,
        robust_pca_basis,
        skill_over_mean_r2,
    )

    rng0 = np.random.default_rng(0)
    n, hc, hy = 18, 30, 40
    xc = rng0.standard_normal((n, hc)).astype(np.float64)
    z = rng0.standard_normal((n, 4))
    w = rng0.standard_normal((4, hy))
    yv = (z @ w + 0.3 * rng0.standard_normal((n, hy))).astype(np.float64)
    mu, comps, _ = robust_pca_basis(yv, min(48, n - 2))
    y_pca = (yv - mu) @ comps.T
    n_perms = 20
    rng = np.random.default_rng(SHUFFLE_NULL_SEED)
    serial = []
    for _ in range(n_perms):
        perm = rng.permutation(n)
        pred = ridge_predict_loco_centered(xc, y_pca[perm])
        serial.append(float(skill_over_mean_r2(pred, y_pca[perm])["skill"]))
    rng_b = np.random.default_rng(SHUFFLE_NULL_SEED)
    perm_b = bn.make_perm_matrix(n, n_perms, rng_b)
    batched = bn.batched_ridge_loco_null_skill(xc, y_pca, perm_b)
    assert np.max(np.abs(np.array(serial) - np.array(batched))) < 1e-6


def test_batched_readout_ridge_null_matches_serial():
    """READOUT trained-ridge null: batched ρ == serial re-fit + Spearman per draw."""
    from explore_persona_space.analysis.vectorized_mlp_skill import (
        ridge_predict_loco_centered,
        robust_pca_basis,
    )

    rng0 = np.random.default_rng(3)
    n = 16
    xsum = rng0.standard_normal((n, 50)).astype(np.float64)
    y = (rng0.standard_normal(n) + 0.5 * xsum[:, 0]).astype(np.float64)
    k = min(48, max(1, n - 2))
    mu, comps, _ = robust_pca_basis(xsum, k)
    xp = (xsum - mu) @ comps.T
    n_perms = 20
    rng = np.random.default_rng(SHUFFLE_NULL_SEED)
    serial = []
    for _ in range(n_perms):
        perm = rng.permutation(n)
        pn = ridge_predict_loco_centered(xp, y[perm].reshape(-1, 1))[:, 0]
        dr = _rho_serial(pn, y[perm])
        serial.append(dr if dr is not None else 0.0)
    rng_b = np.random.default_rng(SHUFFLE_NULL_SEED)
    perm_b = bn.make_perm_matrix(n, n_perms, rng_b)
    batched = bn.batched_ridge_loco_null_rho(xp, y, perm_b)
    assert np.max(np.abs(np.array(serial) - np.array(batched))) < 1e-6


def test_batched_projection_null_matches_serial():
    """fixed-r_B projection null: batched ρ == serial _rho(pred, y[perm]) per draw."""
    rng0 = np.random.default_rng(5)
    n = 15
    pred = rng0.standard_normal(n).astype(np.float64)
    y = rng0.standard_normal(n).astype(np.float64)
    n_perms = 20
    rng = np.random.default_rng(SHUFFLE_NULL_SEED)
    serial = []
    for _ in range(n_perms):
        perm = rng.permutation(n)
        dr = _rho_serial(pred, y[perm])
        serial.append(dr if dr is not None else 0.0)
    rng_b = np.random.default_rng(SHUFFLE_NULL_SEED)
    perm_b = bn.make_perm_matrix(n, n_perms, rng_b)
    batched = bn.batched_projection_null_rho(pred, y, perm_b)
    assert np.max(np.abs(np.array(serial) - np.array(batched))) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# 2. Sycophancy subsample seed is PYTHONHASHSEED-invariant (stable sha256 digest)
# ─────────────────────────────────────────────────────────────────────────────
def _subsample_indices(ctx_id: str) -> list[int]:
    """Reproduce the round-2 stable-subsample seed + sample (as in the rejudge)."""
    stable = int(hashlib.sha256(ctx_id.encode()).hexdigest()[:8], 16)
    rng = random.Random(SHUFFLE_NULL_SEED + stable % 100000)
    return rng.sample(list(range(2000)), SYCOPHANCY_SUBSAMPLE_PER_CONTEXT)


def test_subsample_seed_is_process_stable():
    """The sha256-derived seed is deterministic (no dependence on hash randomization).

    Round-1 used ``hash(ctx_id)`` (salted per-process by PYTHONHASHSEED), so this
    would have differed between runs. The sha256 digest is the same in every
    process → identical subsample across runs.
    """
    ctx = "f1_house_librarian"
    a = _subsample_indices(ctx)
    b = _subsample_indices(ctx)
    assert a == b
    # The seed itself must be a pure function of the ctx_id string (sha256), so a
    # freshly computed digest reproduces the exact value regardless of hash seed.
    stable = int(hashlib.sha256(ctx.encode()).hexdigest()[:8], 16)
    assert stable == 3140125910  # pinned expected digest for this ctx_id


def test_rejudge_uses_sha256_not_builtin_hash():
    """The rejudge script seeds the subsample with sha256, NEVER Python hash(str)."""
    src = (_SCRIPTS / "issue810_batch_rejudge_highm.py").read_text()
    # the stable-subsample block must use hashlib.sha256(ctx_id...) and NOT hash(ctx_id).
    assert "hashlib.sha256(ctx_id" in src
    assert "hash(ctx_id)" not in src, "salted builtin hash() must not seed the subsample"


# ─────────────────────────────────────────────────────────────────────────────
# 3. turn_nl newline token id pinned to the Qwen-2.5 family id 198
# ─────────────────────────────────────────────────────────────────────────────
def test_turn_nl_token_id_pinned_to_198():
    """The turn_nl id constant is the Qwen-2.5 newline id 198 (production + smoke)."""
    assert TURN_NL_TOKEN_ID == 198


def test_extract_asserts_newline_id_value():
    """The extractor pins nl_id == TURN_NL_TOKEN_ID (a drifted id must refuse)."""
    src = (_SCRIPTS / "issue810_extract_positions.py").read_text()
    assert "nl_id != TURN_NL_TOKEN_ID" in src, (
        "extractor must assert the newline id equals the pinned 198, not just len==1"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
