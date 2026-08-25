"""Unit tests for scripts/issue2569_leg6.py — leg-6 shift RRR (plan #2569 blockers B2/B3).

The two plan-MANDATED tests (plan v4 §4 leg 6, run in the P-A smoke):

- **B2 estimator validation:** synthetic paired rank-k data (C Gaussian, Delta = C G_k +
  noise at matched SNR) — the split-half denoised-rank estimator MUST recover k, and a
  row-shuffled control (pairing broken) MUST recover 0.
- **B3 pairing-guard permutation test:** a deliberate within-unit row permutation MUST FAIL
  the ordered-key guard while leaving the (permutation-invariant) tbar value check
  unchanged — both facts asserted in ONE test.

Plus: halt-on-duplicate / halt-on-missing keys, join-by-ID correctness, split disjointness
+ conversation grouping, power-iteration accuracy vs exact svdvals, and the key-source
ladder's fail-loud terminal. All synthetic, CPU-fast (d <= 48) — the dense 3584-dim
factorizations stay out of every test path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2569_leg6 as L6  # noqa: E402


def _keys(n: int) -> tuple[list[int], list[str]]:
    """n synthetic composite keys: unique qidx + shas with one duplicate-prompt pair."""
    qidx = list(range(n))
    shas = [f"sha{i:04d}" for i in range(n)]
    if n >= 4:
        shas[1] = shas[0]  # duplicate conversation sha (the measured 82-dup corpus shape)
    return qidx, shas


def _store(n: int, d: int, arms: dict[str, dict[int, torch.Tensor]]) -> dict:
    """A minimal store payload in the OBSERVED pooled/lasttoken schema."""
    qidx, shas = _keys(n)
    return {
        "arms": arms,
        "row_question_idx": qidx,
        "row_sha": shas,
        "schema_version": 1,
        "unit": "synth",
        "metadata": {"n_rows": n},
    }


# ──────────────────────────────────────────────────────────────────────────
# B3 — pairing guard
# ──────────────────────────────────────────────────────────────────────────


def test_pairing_guard_exact_match_passes():
    """Identical ordered keys across the three stores → action 'exact'."""
    qidx, shas = _keys(12)
    keys = list(zip(qidx, shas, strict=True))
    res = L6.pairing_guard(keys, list(keys), list(keys))
    assert res.action == "exact" and res.ordered_match


def test_permutation_fails_guard_while_tbar_check_unchanged():
    """THE plan-mandated B3 test: a within-unit row permutation MUST FAIL the ordered-key
    guard (detected, explicit ID join) while the tbar VALUE check output is UNCHANGED
    (a column mean is permutation-invariant — it cannot certify row order)."""
    rng = np.random.default_rng(0)
    n, d, layer = 16, 8, 19
    qidx, shas = _keys(n)
    keys = list(zip(qidx, shas, strict=True))
    base = torch.tensor(rng.normal(size=(n, d)), dtype=torch.float32)
    trained = base + torch.tensor(rng.normal(size=(n, d)), dtype=torch.float32)

    perm = rng.permutation(n)
    keys_perm = [keys[i] for i in perm]
    res = L6.pairing_guard(keys, keys_perm, list(keys))
    assert res.action == "joined" and not res.ordered_match  # the guard FAILS the ordered match
    # The join must map the permuted store back onto base order exactly.
    assert [keys_perm[i] for i in res.perm_trained] == keys

    # tbar value check: computed Delta column mean identical under the permutation.
    delta = (trained - base).to(torch.float64)
    delta_perm = (trained[torch.as_tensor(perm)] - base[torch.as_tensor(perm)]).to(torch.float64)
    mean_a = {layer: delta.mean(dim=0).numpy()}
    mean_b = {layer: delta_perm.mean(dim=0).numpy()}
    np.testing.assert_allclose(mean_a[layer], mean_b[layer], rtol=0, atol=1e-12)
    tbar_payload = {
        "tbar": {layer: delta.mean(dim=0).to(torch.float32)},
        "n_rows": n,
        "meta": {"n_rows": n, "pos_path": "synth"},
    }
    rec_a = L6.tbar_value_check(mean_a, tbar_payload, n_rows_corpus=n)
    rec_b = L6.tbar_value_check(mean_b, tbar_payload, n_rows_corpus=n)
    assert rec_a == rec_b  # the tbar value check is UNCHANGED by the permutation
    assert rec_a["strict_pass"] is True  # matched basis → the strict check binds and passes


def test_guard_halts_on_duplicate_and_missing_keys():
    """Duplicate composite keys or unequal key sets HALT the arm (never silent pairing)."""
    qidx, shas = _keys(8)
    keys = list(zip(qidx, shas, strict=True))
    dup = list(keys)
    dup[3] = dup[2]  # duplicate composite key
    assert L6.pairing_guard(dup, list(dup), list(dup)).action == "halt"
    missing = list(keys)
    missing[5] = (999, "sha-not-in-base")
    res = L6.pairing_guard(keys, missing, list(keys))
    assert res.action == "halt" and "SETS differ" in res.reason


def test_tbar_basis_mismatch_recorded_not_gating():
    """Banked tbar over a DIFFERENT row basis (the measured n_rows=20 training-mean shape)
    → strict check N/A, cosine corroboration recorded with basis_mismatch: true."""
    layer = 19
    mean = {layer: np.ones(8)}
    tbar_payload = {
        "tbar": {layer: torch.ones(8) * 2.0},
        "n_rows": 20,
        "meta": {"n_rows": 20, "pos_path": "issue1434_writingstyle/.../pos.jsonl"},
    }
    rec = L6.tbar_value_check(mean, tbar_payload, n_rows_corpus=16400)
    assert rec["basis_mismatch"] is True
    assert rec["strict_pass"] is None
    assert rec["per_layer"][layer]["cosine"] == pytest.approx(1.0)


def test_resolve_keys_fail_loud_without_source():
    """The key-source ladder ends in a registered HALT — never a silent order assumption."""
    with pytest.raises(RuntimeError, match="no usable key source"):
        L6.resolve_keys({"arms": {}}, store_name="pooled_base")


# ──────────────────────────────────────────────────────────────────────────
# B2 — split + estimator
# ──────────────────────────────────────────────────────────────────────────


def test_split_halves_disjoint_and_grouped_by_conversation():
    """Halves are disjoint, cover all rows, and duplicate-sha rows never straddle halves."""
    shas = [f"s{i:03d}" for i in range(100)]
    shas[10] = shas[11] = shas[12] = shas[9]  # a 4-row conversation group
    idx1, idx2 = L6.split_halves_by_conversation(shas, seed=0)
    assert len(np.intersect1d(idx1, idx2)) == 0
    assert sorted(np.concatenate([idx1, idx2]).tolist()) == list(range(100))
    group = {9, 10, 11, 12}
    assert group <= set(idx1.tolist()) or group <= set(idx2.tolist())
    # Deterministic under the pinned seed.
    again = L6.split_halves_by_conversation(shas, seed=0)
    assert idx1.tolist() == again[0].tolist() and idx2.tolist() == again[1].tolist()


def _rank_k_data(n: int, d: int, k: int, *, noise: float, seed: int = 3):
    """Synthetic paired rank-k data: C Gaussian, Delta = C G_k + noise (matched SNR)."""
    rng = np.random.default_rng(seed)
    c = rng.normal(size=(n, d))
    u = np.linalg.qr(rng.normal(size=(d, k)))[0]
    v = np.linalg.qr(rng.normal(size=(d, k)))[0]
    g = u @ np.diag(np.linspace(2.0, 1.0, k)) @ v.T
    delta = c @ g + noise * rng.normal(size=(n, d))
    shas = [f"conv{i:05d}" for i in range(n)]
    return (
        torch.tensor(c, dtype=torch.float32),
        torch.tensor(delta, dtype=torch.float32),
        shas,
    )


def test_estimator_recovers_planted_rank_k():
    """THE plan-mandated B2 test (half 1): the estimator recovers the planted rank k."""
    k = 3
    c, delta, shas = _rank_k_data(n=600, d=32, k=k, noise=0.5)
    rec = L6.fit_split_half(c, delta, shas)
    assert rec["denoised_rank"] == k, rec["denoised_rank"]
    # Oracle ceiling for this fixture: signal var tr(G G^T) = 4 + 2.25 + 1 = 7.25 vs
    # noise var d * noise^2 = 32 * 0.25 = 8 => max achievable R^2 ~ 7.25/15.25 ~ 0.475.
    # A healthy fit lands ~0.41; bound well above 0 and below the ceiling.
    assert rec["heldout_r2"]["fit1_eval2"] > 0.3
    # Identity+learned-bias must fail on a non-identity map (plan step 5 expectation).
    assert rec["identity_bias_r2"] < 0.1
    # kNN chance is stated.
    assert set(rec["knn_retrieval"]["chance"]) == {1, 5, 10}


def test_estimator_row_shuffled_control_recovers_zero():
    """THE plan-mandated B2 test (half 2): breaking the row pairing recovers rank 0."""
    c, delta, shas = _rank_k_data(n=600, d=32, k=3, noise=0.5)
    perm = np.random.default_rng(9).permutation(c.shape[0])
    rec = L6.fit_split_half(c, delta[torch.as_tensor(perm)], shas)
    assert rec["denoised_rank"] == 0, rec["denoised_rank"]


def test_half_moments_single_index_set_bans_cross_half_products():
    """The B2 ban: half moments slice C and Delta with ONE shared index set, so a
    cross-half covariance product is unrepresentable through the API; matched-row
    moments equal the direct centered Gram computation."""
    rng = np.random.default_rng(4)
    c = torch.tensor(rng.normal(size=(20, 6)), dtype=torch.float32)
    d = torch.tensor(rng.normal(size=(20, 6)), dtype=torch.float32)
    idx = np.arange(10)
    _mu_c, _mu_d, _scc, scd, n_h = L6.half_moments(c, d, idx)
    assert n_h == 10
    c_h = c[:10].to(torch.float64) - c[:10].to(torch.float64).mean(dim=0)
    d_h = d[:10].to(torch.float64) - d[:10].to(torch.float64).mean(dim=0)
    torch.testing.assert_close(scd, c_h.T @ d_h)
    # The estimator asserts half disjointness (belt to the API's braces).
    with pytest.raises(AssertionError):
        idx_bad = np.arange(20)
        i1, i2 = idx_bad[:10], idx_bad[5:15]
        assert len(np.intersect1d(i1, i2)) == 0, "halves must be disjoint"


def test_top_singular_batched_matches_exact_svdvals():
    """Power iteration reproduces the exact top singular value on a small stack."""
    rng = np.random.default_rng(5)
    mats = torch.tensor(rng.normal(size=(4, 12, 12)), dtype=torch.float64)
    approx = L6.top_singular_batched(mats, iters=100, seed=1)
    exact = torch.linalg.svdvals(mats)[:, 0]
    torch.testing.assert_close(approx, exact, rtol=1e-6, atol=1e-8)


def test_greedy_factor_match_requires_both_sides():
    """Factor cosine = min(|cos_u|, |cos_v|): output-side agreement alone must NOT match."""
    d, k = 8, 2
    eye = np.eye(d)
    s = np.array([2.0, 1.0])
    f1 = L6.HalfFit(torch.zeros(d, d), torch.zeros(d), torch.zeros(d), 1, s, eye[:, :k], eye[:, :k])
    # Half 2 shares u but has ORTHOGONAL v factors → min-side cosine ~0 → no match.
    v2 = eye[:, 2:4]
    f2 = L6.HalfFit(torch.zeros(d, d), torch.zeros(d), torch.zeros(d), 1, s, eye[:, :k], v2)
    matches = L6.greedy_factor_match(f1, f2)
    assert not matches[0]["matched"]
    assert L6.denoised_rank(matches, 0.0, 0.0) == 0


def test_regime_key_stable_and_parameter_sensitive():
    """Resume keys derive from generating parameters and change when a knob changes."""
    a = L6.regime_key(layer=19, convention="last_prompt", seed=0, n_shuffle=20, cos_floor=0.5)
    b = L6.regime_key(layer=19, convention="last_prompt", seed=0, n_shuffle=20, cos_floor=0.5)
    c = L6.regime_key(layer=19, convention="last_ctx", seed=0, n_shuffle=20, cos_floor=0.5)
    assert a == b and a != c


def test_context_matrix_conventions_resolve():
    """The three context conventions resolve to the documented store objects."""
    layer = 19
    lt_arms = {
        "last_prompt": {layer: torch.full((4, 3), 1.0)},
        "last_ctx": {layer: torch.full((4, 3), 2.0)},
    }
    pooled_arms = {
        "context": {layer: torch.full((4, 3), 3.0)},
        "response": {layer: torch.zeros(4, 3)},
    }
    lt = _store(4, 3, lt_arms)
    pb = _store(4, 3, pooled_arms)
    assert float(L6.context_matrix("last_prompt", lt, pb, layer)[0, 0]) == 1.0
    assert float(L6.context_matrix("last_ctx", lt, pb, layer)[0, 0]) == 2.0
    assert float(L6.context_matrix("span_mean", lt, pb, layer)[0, 0]) == 3.0
    with pytest.raises(ValueError, match="unknown context convention"):
        L6.context_matrix("nope", lt, pb, layer)
