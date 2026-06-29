# ruff: noqa: RUF003
# Intentional Unicode (Σ, ρ, λ, ×, ≤, Δ, ⁻¹, ᵀ, ∝) in scientific docstrings + assert messages.
"""(B3) — THE LOAD-BEARING reduction unit test for the issue-666 whitened context gate.

Plan §4c + theory paper (`docs/leakage_theory_paper.tex`) §"Relation to cosine
similarity" (L287-313). The boxed gate factor (A7, L240) is

    g_C(C') = c_Cᵀ Σc⁻¹ c_{C'} / c_Cᵀ Σc⁻¹ c_C

and the paper states (L304-313) the cosine special-case obtains in the limit
``Σc ∝ I`` (with ‖c_{C'}‖ treated as constant + δ ∥ r_B for the FULL predictor).
In that ``Σc = I`` limit the gate becomes

    g_C(C') = c_Cᵀ c_{C'} / c_Cᵀ c_C            (asymmetric source normalization)

which, for a UNIT-NORM source ``c_C`` (c_Cᵀc_C = 1), reduces exactly to
``c_Cᵀ c_{C'}`` = ``cos(c_C, c_{C'})`` when ``c_{C'}`` is also unit-norm. The
paper is explicit (L312-313) that cosine "discards ... ASYMMETRIC source
normalization" — so for NON-unit-norm c_C the gate is ``c_Cᵀ c_{C'} / ‖c_C‖²``,
NOT the symmetric cosine. Both forms are asserted below.

**NO Phase-4 number is computed or reported until this test PASSes** (plan §4c,
§7, §3 H2 falsification). The test is pre-registered with ``rtol=1e-6`` on random
unit-norm ``c`` vectors with ``Σc = I``.

WRONG-IMPLEMENTATION CATCH (sketched, do NOT delete): a ``g_C`` that uses ``Σc``
DIRECTLY where it should use ``Σc⁻¹`` (a sign/inverse bug) gives, for an
anisotropic diagonal ``Σc = diag(d)``,  ``c_Cᵀ Σc c_{C'} / c_Cᵀ Σc c_C`` — which
does NOT equal the whitened value ``c_Cᵀ Σc⁻¹ c_{C'} / c_Cᵀ Σc⁻¹ c_C`` for
d != 1. ``test_anisotropic_diag_regression_catches_missing_inverse`` is
constructed so the inverse-omitting value and the correct value disagree by
>> rtol, so the mis-implementation FAILS. The ``Σc=I`` reduction test alone does
NOT catch it (at Σc=I, Σc == Σc⁻¹), which is exactly why the anisotropic
regression case is mandatory.

CPU-only; no store, no network, no GPU. Seeds pinned for determinism.
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
    """Proxy that imports the target module on first attribute access.

    TDD this round: ``src/explore_persona_space/analysis/leakage_predictor.py``
    does NOT exist yet, so the first ``gate_mod.g_C`` access inside each test
    raises ImportError → the test FAILS (the TDD point — the tests COLLECT and
    FAIL, they do not skip the module). The implementation round (after
    ``epm:approve-tests``) makes them pass. A module-level ``importorskip`` was
    rejected: it skips COLLECTION, so the proposed-test COUNT could not be
    verified by the orchestrator's approve-tests gate.
    """

    def __init__(self, dotted: str):
        self._dotted = dotted

    def __getattr__(self, name):
        import importlib

        return getattr(importlib.import_module(self._dotted), name)


gate_mod = _LazyModule("explore_persona_space.analysis.leakage_predictor")

RTOL = 1e-6  # pre-registered (plan §4c)


def _rand_unit(rng: np.random.Generator, d: int) -> np.ndarray:
    v = rng.standard_normal(d)
    return v / np.linalg.norm(v)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---------------------------------------------------------------------------
# Scenario 1 — Σc = I, unit-norm c → g_C reduces to cos(c_C, c_{C'}).
# ---------------------------------------------------------------------------
def test_identity_sigma_unit_norm_reduces_to_cosine():
    rng = np.random.default_rng(0)
    d = 32
    Sigma_inv = np.eye(d)
    for _ in range(50):
        c_C = _rand_unit(rng, d)
        c_Cp = _rand_unit(rng, d)
        g = gate_mod.g_C(c_C, c_Cp, Sigma_inv)
        # c_Cᵀc_C == 1 for unit c_C, so g == c_Cᵀ c_{C'} == cos(c_C, c_{C'}).
        assert g == pytest.approx(c_C @ c_Cp, rel=RTOL, abs=RTOL)
        assert g == pytest.approx(_cos(c_C, c_Cp), rel=RTOL, abs=RTOL)


def test_identity_sigma_self_gate_is_one():
    """g_C(C) == 1 by construction (the on-source gate, plan §4h η recovery)."""
    rng = np.random.default_rng(1)
    d = 16
    Sigma_inv = np.eye(d)
    for _ in range(20):
        c_C = _rand_unit(rng, d) * rng.uniform(0.3, 4.0)  # any norm
        g = gate_mod.g_C(c_C, c_C, Sigma_inv)
        assert g == pytest.approx(1.0, rel=RTOL, abs=RTOL)


# ---------------------------------------------------------------------------
# Scenario 2 — Σc = I, NON-unit-norm c → ASYMMETRIC source normalization,
# NOT the symmetric cosine (paper L312-313).
# ---------------------------------------------------------------------------
def test_identity_sigma_nonunit_norm_is_asymmetric_not_cosine():
    rng = np.random.default_rng(2)
    d = 24
    Sigma_inv = np.eye(d)
    for _ in range(50):
        c_C = _rand_unit(rng, d) * rng.uniform(2.0, 5.0)  # scale up source
        c_Cp = _rand_unit(rng, d) * rng.uniform(0.2, 0.8)  # scale down target
        g = gate_mod.g_C(c_C, c_Cp, Sigma_inv)
        # Asymmetric form: c_Cᵀ c_{C'} / c_Cᵀ c_C = c_Cᵀ c_{C'} / ‖c_C‖².
        expected_asym = (c_C @ c_Cp) / (c_C @ c_C)
        assert g == pytest.approx(expected_asym, rel=RTOL, abs=RTOL)
        # And it must DIFFER from the symmetric cosine (the thing cosine discards).
        sym_cos = _cos(c_C, c_Cp)
        assert abs(g - sym_cos) > 1e-3, (
            "non-unit-norm gate must NOT equal symmetric cosine — the paper says "
            "cosine discards asymmetric source normalization"
        )


# ---------------------------------------------------------------------------
# Scenario 3 — Σc = diag(d), c_C an eigenvector of Σc → correct scaling.
# ---------------------------------------------------------------------------
def test_diagonal_sigma_eigenvector_source_scaling():
    rng = np.random.default_rng(3)
    d = 8
    eig = rng.uniform(0.5, 10.0, size=d)  # eigenvalues of Σc
    Sigma = np.diag(eig)
    Sigma_inv = np.diag(1.0 / eig)
    # c_C aligned with axis k (an eigenvector of Σc).
    for k in range(d):
        c_C = np.zeros(d)
        c_C[k] = rng.uniform(1.0, 3.0)
        c_Cp = rng.standard_normal(d)
        g = gate_mod.g_C(c_C, c_Cp, Sigma_inv)
        # c_Cᵀ Σc⁻¹ c_{C'} = c_C[k] * (1/eig[k]) * c_Cp[k]
        # c_Cᵀ Σc⁻¹ c_C    = c_C[k]² * (1/eig[k])
        # → g = c_Cp[k] / c_C[k]   (eigenvalue cancels for an eigenvector source)
        expected = c_Cp[k] / c_C[k]
        assert g == pytest.approx(expected, rel=RTOL, abs=RTOL)
        # Sanity: building Σ⁻¹ from Σ and re-inverting agrees.
        assert np.allclose(np.linalg.inv(Sigma), Sigma_inv, rtol=RTOL)


# ---------------------------------------------------------------------------
# Scenario 4 — anisotropic random PSD Σc, d=4, analytic regression.
# ALSO the wrong-implementation catch (inverse omitted).
# ---------------------------------------------------------------------------
def test_anisotropic_psd_d4_analytic_regression():
    # Fixed inputs so this is a numerical regression (no RNG drift). d=4.
    A = np.array(
        [
            [2.0, 0.3, -0.1, 0.0],
            [0.3, 1.5, 0.2, 0.1],
            [-0.1, 0.2, 3.0, -0.4],
            [0.0, 0.1, -0.4, 1.2],
        ]
    )
    Sigma = A @ A.T  # SPD by construction
    Sigma_inv = np.linalg.inv(Sigma)
    c_C = np.array([1.0, -2.0, 0.5, 3.0])
    c_Cp = np.array([0.7, 1.1, -1.3, 0.4])
    g = gate_mod.g_C(c_C, c_Cp, Sigma_inv)
    num = c_C @ Sigma_inv @ c_Cp
    den = c_C @ Sigma_inv @ c_C
    expected = num / den
    assert g == pytest.approx(expected, rel=RTOL, abs=RTOL)


def test_anisotropic_diag_regression_catches_missing_inverse():
    """A g_C that uses Σc directly instead of Σc⁻¹ must FAIL this case.

    With an anisotropic diagonal Σc the whitened value and the inverse-omitting
    value diverge by >> rtol, so a correct implementation matches the whitened
    value and a mis-implemented one does not.
    """
    eig = np.array([0.1, 10.0, 0.5, 4.0])  # strongly anisotropic
    Sigma = np.diag(eig)
    Sigma_inv = np.diag(1.0 / eig)
    c_C = np.array([1.0, 1.0, 1.0, 1.0])
    c_Cp = np.array([1.0, -1.0, 2.0, -0.5])
    g = gate_mod.g_C(c_C, c_Cp, Sigma_inv)

    whitened = (c_C @ Sigma_inv @ c_Cp) / (c_C @ Sigma_inv @ c_C)
    inverse_omitting = (c_C @ Sigma @ c_Cp) / (c_C @ Sigma @ c_C)
    # The two forms are numerically far apart on this anisotropic Σc — proving
    # the test discriminates a missing-inverse bug.
    assert abs(whitened - inverse_omitting) > 0.1, (
        "test mis-constructed: whitened and inverse-omitting forms must differ"
    )
    assert g == pytest.approx(whitened, rel=RTOL, abs=RTOL)
    assert abs(g - inverse_omitting) > 0.1, (
        "g_C returned the inverse-omitting value — Σc⁻¹ is mis-implemented"
    )


# ---------------------------------------------------------------------------
# Scenario 5 — conditioning: Σc = ridge λI for small λ stays finite + uses the
# REGULARIZED inverse (Σc + λI)⁻¹.
# ---------------------------------------------------------------------------
def test_ridge_regularized_inverse_finite_and_well_defined():
    rng = np.random.default_rng(5)
    d = 12
    # Rank-deficient raw Σc (rank 3 < d) — the inverse only exists after ridge.
    base = rng.standard_normal((d, 3))
    Sigma = base @ base.T  # rank 3, singular
    assert np.linalg.matrix_rank(Sigma) <= 3
    for lam in (1e-6, 1e-3, 1.0):
        Sigma_inv = np.linalg.inv(Sigma + lam * np.eye(d))
        c_C = _rand_unit(rng, d)
        c_Cp = _rand_unit(rng, d)
        g = gate_mod.g_C(c_C, c_Cp, Sigma_inv)
        assert np.isfinite(g), f"gate not finite at λ={lam}"
    # As λ → ∞ the gate → the Σc=I (identity) limit (whitening washed out).
    big = 1e9
    Sigma_inv_big = np.linalg.inv(Sigma + big * np.eye(d))
    c_C = _rand_unit(rng, d)
    c_Cp = _rand_unit(rng, d)
    g_big = gate_mod.g_C(c_C, c_Cp, Sigma_inv_big)
    assert g_big == pytest.approx(c_C @ c_Cp, abs=1e-3), (
        "large-λ gate should approach the Σc=I cosine limit"
    )
