"""Unit tests for linear CKA (issue #654, plan §3(b)).

Pins the Kornblith et al. 2019 (arXiv 1905.00414) linear-CKA properties:
self-similarity 1, orthogonal invariance, isotropic-scale invariance,
symmetry, and an independent-Gaussian low-CKA floor. CPU-only, fixed seed,
pure linear algebra — runs in well under a second.
"""

from __future__ import annotations

import torch

from explore_persona_space.analysis.representation_shift import cka_per_layer, linear_cka

SEED = 654
TOL = 1e-5


def _gaussian(n: int, d: int, *, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g, dtype=torch.float64)


def test_self_similarity_is_one():
    X = _gaussian(120, 48, seed=SEED)
    assert abs(linear_cka(X, X) - 1.0) < TOL


def test_orthogonal_invariance():
    """CKA(X, X @ Q) == 1 for a random orthogonal Q (column rotation)."""
    X = _gaussian(120, 64, seed=SEED)
    g = torch.Generator().manual_seed(SEED + 1)
    a = torch.randn(64, 64, generator=g, dtype=torch.float64)
    Q, _ = torch.linalg.qr(a)  # Q is orthogonal
    # Sanity: Q is genuinely orthogonal.
    assert torch.allclose(Q.T @ Q, torch.eye(64, dtype=torch.float64), atol=1e-8)
    assert abs(linear_cka(X, X @ Q) - 1.0) < TOL


def test_isotropic_scale_invariance():
    X = _gaussian(120, 48, seed=SEED)
    assert abs(linear_cka(X, 3.7 * X) - 1.0) < TOL
    # Also invariant to a negative isotropic scale.
    assert abs(linear_cka(X, -2.0 * X) - 1.0) < TOL


def test_symmetry():
    X = _gaussian(120, 48, seed=SEED)
    Y = _gaussian(120, 32, seed=SEED + 2)
    assert abs(linear_cka(X, Y) - linear_cka(Y, X)) < TOL


def test_independent_gaussian_floor():
    """Independent Gaussians give a CKA far below self-similar 1.0.

    Linear CKA between two INDEPENDENT Gaussians carries a finite-sample
    (biased-HSIC) floor ≈ d/n (Song et al. 2012); at the plan's named n=200,
    d=64 (d/n=0.32) the empirical floor sits ~0.24 — the plan's "<=~0.15"
    estimate is unreachable at those dims, so the n=200 check uses a 0.35
    bound that still cleanly separates independence from the self-similar 1.0
    (test_self_similarity_is_one), and the stricter sub-0.1 check below holds
    the same construct at n=1000 where the d/n bias shrinks into the plan's
    intended low-CKA region.
    """
    X = _gaussian(200, 64, seed=SEED + 10)
    Y = _gaussian(200, 64, seed=SEED + 20)
    cka = linear_cka(X, Y)
    assert 0.0 <= cka <= 0.35, f"independent-Gaussian CKA={cka} (n=200,d=64) should be < ~0.35"
    # n >> d shrinks the finite-sample bias into the plan's intended <~0.1 floor.
    Xb = _gaussian(1000, 64, seed=SEED + 11)
    Yb = _gaussian(1000, 64, seed=SEED + 21)
    cka_big = linear_cka(Xb, Yb)
    assert 0.0 <= cka_big <= 0.12, (
        f"independent-Gaussian CKA={cka_big} (n=1000,d=64) should be <~0.1"
    )


def test_range_bounds():
    """CKA always lands in [0, 1] for arbitrary inputs."""
    X = _gaussian(50, 30, seed=SEED + 30)
    Y = _gaussian(50, 30, seed=SEED + 40)
    cka = linear_cka(X, Y)
    assert 0.0 <= cka <= 1.0


def test_n_below_two_raises():
    X = _gaussian(1, 16, seed=SEED)
    Y = _gaussian(1, 16, seed=SEED + 1)
    try:
        linear_cka(X, Y)
    except AssertionError:
        pass
    else:
        raise AssertionError("linear_cka must reject n<2")


def test_mismatched_n_raises():
    X = _gaussian(10, 16, seed=SEED)
    Y = _gaussian(12, 16, seed=SEED + 1)
    try:
        linear_cka(X, Y)
    except AssertionError:
        pass
    else:
        raise AssertionError("linear_cka must reject mismatched n")


def test_cka_per_layer_shape_and_values():
    """cka_per_layer returns one CKA per layer, all in [0, 1]."""
    n, n_layers, hidden = 40, 5, 24
    g = torch.Generator().manual_seed(SEED + 50)
    bank_a = torch.randn(n, n_layers, hidden, generator=g, dtype=torch.float64)
    bank_b = bank_a.clone()
    per_layer = cka_per_layer(bank_a, bank_b)
    assert len(per_layer) == n_layers
    # bank_b == bank_a -> every layer CKA == 1.
    for v in per_layer:
        assert abs(v - 1.0) < TOL

    # A genuinely different second bank -> values still in [0, 1].
    bank_c = torch.randn(n, n_layers, hidden, generator=g, dtype=torch.float64)
    per_layer2 = cka_per_layer(bank_a, bank_c)
    assert len(per_layer2) == n_layers
    assert all(0.0 <= v <= 1.0 for v in per_layer2)
