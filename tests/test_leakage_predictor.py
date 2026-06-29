# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (Σ, ρ, λ, η, δ, ×, ≤, Δ, ⁻¹, ᵀ, ∝, ⊥) in scientific docstrings + asserts.
"""L̂ assembly, the (C7) apples-to-apples cosine toggles, η recovery, the base
prior, and the shuffle controls for the issue-666 leakage predictor.

Plan §4d (L̂ assembly), §4e (apples-to-apples cosine — three independent
boolean toggles δ→r_B / drop-norms / Σc⁻¹→I), §4f (base-behavior prior E0),
§4h (η on-source recovery), §5 (shuffled-key / shuffled-query controls).
Theory: `docs/leakage_theory_paper.tex` boxed predictor (L260-268) +
§"Relation to cosine similarity" (L287-313).

The boxed predictor (ignoring η, which drops out of ranking/correlation tests):

    L̂_{C,B→C',B'} ∝ (r_{B'}ᵀ δ_{C,B}) · g_C(C')      with  δ = t − v0(C)

The cosine special-case (the THREE allowed toggles, C7):

    L̂^cos ∝ cos(r_{B'}, r_B) · cos(c_C, c_{C'})

  toggle 1 (δ→r_B): the behavior term uses r_B in place of δ;
  toggle 2 (drop-norms): drop the source/target norm handling (asymmetric→symmetric);
  toggle 3 (Σc⁻¹→I): the gate whitening is removed (composes with the (B3) test).

CPU-only; hand-computed synthetic inputs; no store, no network, no GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


class _LazyModule:
    """Proxy that imports the target on first attribute access (TDD).

    The net-new module does NOT exist this round, so the first ``pred.<fn>``
    access inside each test raises ImportError → the test FAILS (not skips).
    A module-level ``importorskip`` was rejected because it skips COLLECTION,
    so the proposed-test count could not be verified by approve-tests.
    """

    def __init__(self, dotted: str):
        self._dotted = dotted

    def __getattr__(self, name):
        import importlib

        return getattr(importlib.import_module(self._dotted), name)


pred = _LazyModule("explore_persona_space.analysis.leakage_predictor")

RTOL = 1e-6


def _rand_unit(rng, d):
    v = rng.standard_normal(d)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# L̂ assembly against hand-computed value.
# ---------------------------------------------------------------------------
def test_lhat_matches_hand_computed_value():
    # Small d, fully hand-computable.
    eta = 2.5
    r_Bp = np.array([1.0, 0.0, -2.0, 0.5])  # read-out for evaluated behavior B'
    delta = np.array([0.3, 1.2, 0.1, -0.4])  # δ_{C,B} = t − v0(C)
    c_C = np.array([2.0, 1.0, 0.0, 1.0])
    c_Cp = np.array([1.0, -1.0, 3.0, 0.5])
    Sigma_inv = np.diag([1.0, 0.5, 2.0, 4.0])

    g = (c_C @ Sigma_inv @ c_Cp) / (c_C @ Sigma_inv @ c_C)
    expected = eta * (r_Bp @ delta) * g

    got = pred.lhat(eta=eta, r_Bp=r_Bp, delta=delta, c_C=c_C, c_Cp=c_Cp, Sigma_inv=Sigma_inv)
    assert got == pytest.approx(expected, rel=RTOL, abs=RTOL)


def test_lhat_relative_invariant_to_eta_scale():
    """At a fixed source, η is a positive scalar multiplier → ranking-invariant."""
    rng = np.random.default_rng(10)
    d = 6
    r_Bp = rng.standard_normal(d)
    delta = rng.standard_normal(d)
    c_C = rng.standard_normal(d)
    Sigma_inv = np.eye(d)
    targets = [rng.standard_normal(d) for _ in range(8)]
    eta_a, eta_b = 1.0, 7.3
    vals_a = [
        pred.lhat(eta=eta_a, r_Bp=r_Bp, delta=delta, c_C=c_C, c_Cp=t, Sigma_inv=Sigma_inv)
        for t in targets
    ]
    vals_b = [
        pred.lhat(eta=eta_b, r_Bp=r_Bp, delta=delta, c_C=c_C, c_Cp=t, Sigma_inv=Sigma_inv)
        for t in targets
    ]
    # ranking identical
    assert np.argsort(vals_a).tolist() == np.argsort(vals_b).tolist()
    # exact positive-scalar relationship
    assert np.allclose(np.array(vals_b), (eta_b / eta_a) * np.array(vals_a), rtol=RTOL)


# ---------------------------------------------------------------------------
# C7 toggle 1 — δ → r_B in the behavior term.
# ---------------------------------------------------------------------------
def test_c7_toggle_delta_to_rB_zeroes_when_rB_orthogonal_to_delta():
    rng = np.random.default_rng(11)
    d = 16
    # r_B ⊥ δ: behavior term r_Bᵀδ → 0 under the toggle, but the full L̂ keeps r_{B'}ᵀδ.
    r_B = _rand_unit(rng, d)
    # Build δ orthogonal to r_B.
    raw = rng.standard_normal(d)
    delta = raw - (raw @ r_B) * r_B
    assert abs(delta @ r_B) < 1e-10
    r_Bp = _rand_unit(rng, d)  # evaluated behavior B' (generic, not ⊥ to δ)
    c_C = _rand_unit(rng, d)
    c_Cp = _rand_unit(rng, d)
    Sigma_inv = np.eye(d)

    full = pred.lhat(eta=1.0, r_Bp=r_Bp, delta=delta, c_C=c_C, c_Cp=c_Cp, Sigma_inv=Sigma_inv)
    toggled = pred.lhat_variant(
        eta=1.0,
        r_Bp=r_Bp,
        r_B=r_B,
        delta=delta,
        c_C=c_C,
        c_Cp=c_Cp,
        Sigma_inv=Sigma_inv,
        toggle_delta_to_rB=True,
    )
    # Toggled behavior term r_{B'}ᵀ r_B (B'==B case driving it to 0): construct
    # the toggle so the behavior term uses r_B → here we set r_Bp == r_B so the
    # toggle's behavior term becomes r_Bᵀ(r_B)?  No — toggle replaces δ with r_B:
    # toggled behavior term = r_{B'}ᵀ r_B. To zero it, make r_{B'} ⊥ r_B.
    r_Bp_perp = raw / np.linalg.norm(raw)  # raw ⊥ r_B by construction above
    r_Bp_perp = r_Bp_perp - (r_Bp_perp @ r_B) * r_B
    r_Bp_perp = r_Bp_perp / np.linalg.norm(r_Bp_perp)
    toggled_zero = pred.lhat_variant(
        eta=1.0,
        r_Bp=r_Bp_perp,
        r_B=r_B,
        delta=delta,
        c_C=c_C,
        c_Cp=c_Cp,
        Sigma_inv=Sigma_inv,
        toggle_delta_to_rB=True,
    )
    assert abs(toggled_zero) < 1e-9, "δ→r_B toggle must zero when r_{B'} ⊥ r_B"
    # The full predictor (uses δ, not r_B) is NOT zero for the same r_{B'}_perp,
    # because r_{B'}ᵀδ need not vanish.
    full_perp = pred.lhat(
        eta=1.0, r_Bp=r_Bp_perp, delta=delta, c_C=c_C, c_Cp=c_Cp, Sigma_inv=Sigma_inv
    )
    assert abs(full_perp) > 1e-6, "full L̂ uses δ and should not vanish here"
    # full / toggled are different objects in general:
    assert not np.isclose(full, toggled, rtol=1e-3)


# ---------------------------------------------------------------------------
# C7 toggle 2 — drop the source/target norm handling (asymmetric → symmetric).
# ---------------------------------------------------------------------------
def test_c7_toggle_drop_norms_is_noop_on_unit_c_but_changes_nonunit():
    rng = np.random.default_rng(12)
    d = 10
    r_Bp = rng.standard_normal(d)
    delta = rng.standard_normal(d)
    r_B = rng.standard_normal(d)
    Sigma_inv = np.eye(d)

    # unit-norm c → drop-norms is a no-op (asymmetric == symmetric form there).
    c_C_u = _rand_unit(rng, d)
    c_Cp_u = _rand_unit(rng, d)
    keep = pred.lhat_variant(
        eta=1.0,
        r_Bp=r_Bp,
        r_B=r_B,
        delta=delta,
        c_C=c_C_u,
        c_Cp=c_Cp_u,
        Sigma_inv=Sigma_inv,
        drop_norms=False,
    )
    drop = pred.lhat_variant(
        eta=1.0,
        r_Bp=r_Bp,
        r_B=r_B,
        delta=delta,
        c_C=c_C_u,
        c_Cp=c_Cp_u,
        Sigma_inv=Sigma_inv,
        drop_norms=True,
    )
    assert keep == pytest.approx(drop, rel=RTOL, abs=RTOL)

    # non-unit c → drop-norms changes the gate term.
    c_C_n = _rand_unit(rng, d) * 3.0
    c_Cp_n = _rand_unit(rng, d) * 0.4
    keep_n = pred.lhat_variant(
        eta=1.0,
        r_Bp=r_Bp,
        r_B=r_B,
        delta=delta,
        c_C=c_C_n,
        c_Cp=c_Cp_n,
        Sigma_inv=Sigma_inv,
        drop_norms=False,
    )
    drop_n = pred.lhat_variant(
        eta=1.0,
        r_Bp=r_Bp,
        r_B=r_B,
        delta=delta,
        c_C=c_C_n,
        c_Cp=c_Cp_n,
        Sigma_inv=Sigma_inv,
        drop_norms=True,
    )
    assert not np.isclose(keep_n, drop_n, rtol=1e-3), (
        "drop-norms must change the gate for non-unit-norm c"
    )


# ---------------------------------------------------------------------------
# C7 toggle 3 — Σc⁻¹ → I reduces the gate to (c_Cᵀ c_{C'}) (composes with (B3)).
# ---------------------------------------------------------------------------
def test_c7_toggle_sigma_to_identity_reduces_gate():
    rng = np.random.default_rng(13)
    d = 14
    r_Bp = rng.standard_normal(d)
    delta = rng.standard_normal(d)
    r_B = rng.standard_normal(d)
    c_C = _rand_unit(rng, d)
    c_Cp = _rand_unit(rng, d)
    # Anisotropic real Σc⁻¹.
    A = rng.standard_normal((d, d))
    Sigma = A @ A.T + np.eye(d)
    Sigma_inv = np.linalg.inv(Sigma)

    toggled = pred.lhat_variant(
        eta=1.0,
        r_Bp=r_Bp,
        r_B=r_B,
        delta=delta,
        c_C=c_C,
        c_Cp=c_Cp,
        Sigma_inv=Sigma_inv,
        toggle_sigma_to_identity=True,
    )
    # With Σc⁻¹→I the gate is c_Cᵀc_{C'}/c_Cᵀc_C; build the reference directly.
    gate_I = (c_C @ c_Cp) / (c_C @ c_C)
    ref = (r_Bp @ delta) * gate_I  # behavior term still uses δ (only the gate toggled)
    assert toggled == pytest.approx(ref, rel=RTOL, abs=RTOL)
    # And it differs from the whitened full predictor.
    full = pred.lhat(eta=1.0, r_Bp=r_Bp, delta=delta, c_C=c_C, c_Cp=c_Cp, Sigma_inv=Sigma_inv)
    assert not np.isclose(toggled, full, rtol=1e-3)


def test_all_three_toggles_recover_cosine_special_case():
    """All three toggles together == the apples-to-apples cosine predictor."""
    rng = np.random.default_rng(14)
    d = 12
    r_Bp = rng.standard_normal(d)
    r_B = rng.standard_normal(d)
    delta = rng.standard_normal(d)
    c_C = _rand_unit(rng, d) * 2.2
    c_Cp = _rand_unit(rng, d) * 0.6
    A = rng.standard_normal((d, d))
    Sigma_inv = np.linalg.inv(A @ A.T + np.eye(d))

    cos_all = pred.lhat_variant(
        eta=1.0,
        r_Bp=r_Bp,
        r_B=r_B,
        delta=delta,
        c_C=c_C,
        c_Cp=c_Cp,
        Sigma_inv=Sigma_inv,
        toggle_delta_to_rB=True,
        drop_norms=True,
        toggle_sigma_to_identity=True,
    )
    expected = ((r_Bp @ r_B) / (np.linalg.norm(r_Bp) * np.linalg.norm(r_B))) * (
        (c_C @ c_Cp) / (np.linalg.norm(c_C) * np.linalg.norm(c_Cp))
    )
    assert cos_all == pytest.approx(expected, rel=1e-5, abs=1e-5)


# ---------------------------------------------------------------------------
# η on-source recovery (plan §4h): η = Δs_on-source / (r_Bᵀδ), gate==1 at (C,C).
# ---------------------------------------------------------------------------
def test_eta_on_source_recovery():
    rng = np.random.default_rng(15)
    d = 20
    r_B = rng.standard_normal(d)
    delta = rng.standard_normal(d)
    # On-source latent shift Δs = r_Bᵀ(v⁺(C)−v0(C)) = r_Bᵀ r_plus.
    r_plus = rng.standard_normal(d)
    ds_on_source = float(r_B @ r_plus)
    eta = pred.recover_eta(ds_on_source=ds_on_source, r_B=r_B, delta=delta)
    assert eta == pytest.approx(ds_on_source / (r_B @ delta), rel=RTOL, abs=RTOL)
    # And the recovered η reproduces the on-source L̂ (gate==1):
    c_C = rng.standard_normal(d)
    Sigma_inv = np.eye(d)
    lhat_self = pred.lhat(eta=eta, r_Bp=r_B, delta=delta, c_C=c_C, c_Cp=c_C, Sigma_inv=Sigma_inv)
    assert lhat_self == pytest.approx(ds_on_source, rel=RTOL, abs=RTOL)


# ---------------------------------------------------------------------------
# Base-prior baseline E0(C',B') = r_{B'}ᵀ v0(C').
# ---------------------------------------------------------------------------
def test_base_prior_is_direct_dot_product():
    rng = np.random.default_rng(16)
    d = 18
    r_Bp = rng.standard_normal(d)
    v0_Cp = rng.standard_normal(d)
    got = pred.base_prior(r_Bp=r_Bp, v0_Cp=v0_Cp)
    assert got == pytest.approx(r_Bp @ v0_Cp, rel=RTOL, abs=RTOL)


# ---------------------------------------------------------------------------
# Shuffle controls (plan §5): permutation invariance / equivariance.
# ---------------------------------------------------------------------------
def test_shuffled_key_is_permutation_of_predictor():
    """Shuffling the gate keys (c_C across contexts) yields a relabeled predictor.

    Across many targets, the shuffled-key predictor should correlate near zero
    with the unshuffled predictor (the gate's context-specificity destroyed).
    """
    from scipy.stats import spearmanr

    rng = np.random.default_rng(17)
    d = 8
    n_ctx = 60
    r_Bp = rng.standard_normal(d)
    delta = rng.standard_normal(d)
    c_C = rng.standard_normal(d)  # the true source key
    Sigma_inv = np.eye(d)
    c_Cps = [rng.standard_normal(d) for _ in range(n_ctx)]

    real = np.array(
        [
            pred.lhat(eta=1.0, r_Bp=r_Bp, delta=delta, c_C=c_C, c_Cp=t, Sigma_inv=Sigma_inv)
            for t in c_Cps
        ]
    )
    # Shuffle the KEY across contexts (per-target a random key, breaking the
    # shared-source-key structure) via the module's documented control hook.
    shuffled = pred.shuffle_key_predictor(
        r_Bp=r_Bp, delta=delta, c_Cps=c_Cps, Sigma_inv=Sigma_inv, seed=123
    )
    assert len(shuffled) == n_ctx
    rho, _ = spearmanr(real, shuffled)
    assert abs(rho) < 0.5, f"shuffled-key ρ should be near zero, got {rho:.3f}"


def test_shuffled_query_destroys_target_specificity():
    from scipy.stats import spearmanr

    rng = np.random.default_rng(18)
    d = 8
    n_ctx = 60
    r_Bp = rng.standard_normal(d)
    delta = rng.standard_normal(d)
    c_C = rng.standard_normal(d)
    Sigma_inv = np.eye(d)
    c_Cps = [rng.standard_normal(d) for _ in range(n_ctx)]
    real = np.array(
        [
            pred.lhat(eta=1.0, r_Bp=r_Bp, delta=delta, c_C=c_C, c_Cp=t, Sigma_inv=Sigma_inv)
            for t in c_Cps
        ]
    )
    shuffled = pred.shuffle_query_predictor(
        r_Bp=r_Bp, delta=delta, c_C=c_C, c_Cps=c_Cps, Sigma_inv=Sigma_inv, seed=321
    )
    assert len(shuffled) == n_ctx
    # The same set of values, permuted → identical multiset, near-zero rank corr.
    assert sorted(np.round(shuffled, 9)) == sorted(np.round(real, 9))
    rho, _ = spearmanr(real, shuffled)
    assert abs(rho) < 0.5, f"shuffled-query ρ should be near zero, got {rho:.3f}"
