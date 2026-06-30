# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (ρ, Δ, ×, ≤, ᵀ) in scientific docstrings + asserts.
"""Noise-floor (test-retest reliability) estimator for issue-666 (plan §4i, §11).

The test-retest reliability is the CEILING on any predictor's achievable ρ. Two
independent estimates of the latent ``Δs`` from independent probe-split halves of
``v_plus_probe``/``v0_probe`` (reusing the ``issue664_aggregate_gate.probe_split_floor``
half-split logic), recompute ``Δs`` on each half, and the test-retest ρ between
the two halves is the floor.

Pre-registered MC (plan §11): **200 probe-split resamples × 3 RNG seeds = 600
total**. These tests pin: (1) the estimator splits the n_probe axis into two
independent halves per resample; (2) the registered MC count is what the
estimator runs; (3) a self-consistent predictor floors at ρ ≈ 1.0 and a noisy
one returns ρ < 1.

CPU-only; small synthetic probe tensors; no store, no network, no GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import torch  # noqa: E402  (installed; not a TDD-deferred dep)


class _LazyModule:
    """Proxy that imports a per-issue script on first attribute access (TDD).

    The net-new script does NOT exist this round, so the first ``nf.<fn>``
    access inside each test raises ImportError → the test FAILS (not skips).
    A module-level ``importorskip`` was rejected because it skips COLLECTION,
    so the proposed-test count could not be verified by approve-tests.
    """

    def __init__(self, dotted: str):
        object.__setattr__(self, "_dotted", dotted)

    def __getattr__(self, name):
        import importlib

        return getattr(importlib.import_module(self._dotted), name)

    def __setattr__(self, name, value):
        # Forward attribute SETS to the real module so monkeypatch.setattr(proxy,
        # ...) patches the module function the implementation actually calls (and
        # monkeypatch's teardown restore forwards back the same way). Without this,
        # a set landed on the proxy instance and the real module stayed unpatched.
        import importlib

        setattr(importlib.import_module(self._dotted), name, value)


nf = _LazyModule("issue666_noise_floor")

# Pre-registered MC config (plan §11).
MC_RESAMPLES = 200
MC_SEEDS = (0, 1, 2)
EXPECTED_TOTAL = MC_RESAMPLES * len(MC_SEEDS)  # 600


def test_registered_mc_counts():
    assert nf.MC_RESAMPLES == MC_RESAMPLES
    assert tuple(nf.MC_SEEDS) == MC_SEEDS
    assert nf.MC_RESAMPLES * len(nf.MC_SEEDS) == EXPECTED_TOTAL == 600


# ---------------------------------------------------------------------------
# Probe-split structure: two disjoint halves of the n_probe axis per resample.
# ---------------------------------------------------------------------------
def test_probe_split_uses_two_disjoint_halves():
    rng = np.random.default_rng(0)
    n_probe = 48
    h1, h2 = nf.probe_half_indices(n_probe, rng)
    s1, s2 = set(h1.tolist()), set(h2.tolist())
    assert s1.isdisjoint(s2), "probe halves must be disjoint"
    assert len(s1) == n_probe // 2
    assert s1 | s2 <= set(range(n_probe))
    # Two halves cover (close to) the full axis (n_probe even → exact).
    assert len(s1) + len(s2) <= n_probe


def test_noise_floor_runs_the_registered_count(monkeypatch):
    """The estimator performs exactly MC_RESAMPLES × len(MC_SEEDS) splits."""
    calls = {"n": 0}
    real = nf.probe_half_indices

    def _counting(n_probe, rng):
        calls["n"] += 1
        return real(n_probe, rng)

    monkeypatch.setattr(nf, "probe_half_indices", _counting)

    # Tiny synthetic probe tensors (n_ctx, n_probe, n_layer, d).
    rng = np.random.default_rng(1)
    n_ctx, n_probe, nl, d = 6, 48, 2, 16
    v_plus_probe = torch.from_numpy(rng.standard_normal((n_ctx, n_probe, nl, d)).astype("float32"))
    v0_probe = torch.from_numpy(rng.standard_normal((n_ctx, n_probe, nl, d)).astype("float32"))
    r_B = rng.standard_normal(d).astype("float32")

    nf.estimate_noise_floor(v_plus_probe, v0_probe, r_B=r_B, layer=0, source_idx=0)
    assert calls["n"] == EXPECTED_TOTAL, (
        f"expected {EXPECTED_TOTAL} probe splits (200×3), got {calls['n']}"
    )


# ---------------------------------------------------------------------------
# Sanity: self-consistent → ρ ≈ 1; noisy → ρ < 1.
# ---------------------------------------------------------------------------
def test_self_consistent_predictor_floors_near_one():
    """When the two halves see the SAME deterministic Δs, the floor ρ ≈ 1.0."""
    rng = np.random.default_rng(2)
    n_ctx, n_probe, nl, d = 30, 48, 2, 16
    # A per-context deterministic signal replicated across ALL probes (zero
    # within-context measurement noise) → the two halves agree → ρ ≈ 1.
    base = rng.standard_normal((n_ctx, nl, d)).astype("float32")
    shift = rng.standard_normal((n_ctx, nl, d)).astype("float32")
    v0 = np.broadcast_to(base[:, None], (n_ctx, n_probe, nl, d)).copy()
    v_plus = np.broadcast_to((base + shift)[:, None], (n_ctx, n_probe, nl, d)).copy()
    r_B = rng.standard_normal(d).astype("float32")
    floor = nf.estimate_noise_floor(
        torch.from_numpy(v_plus), torch.from_numpy(v0), r_B=r_B, layer=0, source_idx=0
    )
    assert floor.rho_mean == pytest.approx(1.0, abs=1e-3), (
        f"noise-free → test-retest ρ should be ≈ 1.0, got {floor.rho_mean:.4f}"
    )


def test_noisy_predictor_returns_rho_below_one():
    """With finite within-context measurement noise, the floor ρ < 1."""
    rng = np.random.default_rng(3)
    n_ctx, n_probe, nl, d = 40, 48, 2, 16
    base = rng.standard_normal((n_ctx, nl, d)).astype("float32")
    shift = rng.standard_normal((n_ctx, nl, d)).astype("float32")
    # Strong per-probe noise so the two halves disagree substantially.
    noise_v0 = rng.standard_normal((n_ctx, n_probe, nl, d)).astype("float32") * 2.0
    noise_vp = rng.standard_normal((n_ctx, n_probe, nl, d)).astype("float32") * 2.0
    v0 = base[:, None] + noise_v0
    v_plus = (base + shift)[:, None] + noise_vp
    r_B = rng.standard_normal(d).astype("float32")
    floor = nf.estimate_noise_floor(
        torch.from_numpy(v0.astype("float32") * 0 + v_plus.astype("float32")),
        torch.from_numpy(v0.astype("float32")),
        r_B=r_B,
        layer=0,
        source_idx=0,
    )
    assert floor.rho_mean < 0.999, (
        f"noisy halves should degrade the test-retest ρ below 1, got {floor.rho_mean:.4f}"
    )
    # The estimator reports a spread across the 600 resamples.
    assert floor.rho_std >= 0.0
    assert hasattr(floor, "rho_mean") and hasattr(floor, "rho_std")
