from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from scripts.issue2643_gradient_pursuit import (
    apply_behavior_pursuit,
    factorized_local_coefficients,
    fit_behavior_pursuit,
    signed_gradient_pursuit,
)
from scripts.issue2643_marker_panel import clustered_auc_delta


def test_signed_gradient_pursuit_recovers_signed_atoms() -> None:
    rng = np.random.default_rng(2643)
    z = rng.normal(size=(600, 10)).astype(np.float32)
    target = 2.25 * z[:, 2] - 1.5 * z[:, 8]
    fit = signed_gradient_pursuit(z, target, max_k=2, checkpoints=(1, 2), ridge_relative=0.0)
    assert set(fit[2].support.tolist()) == {2, 8}
    by_atom = dict(zip(fit[2].support, fit[2].coefficients, strict=True))
    np.testing.assert_allclose(by_atom[2], 2.25, atol=1e-5)
    np.testing.assert_allclose(by_atom[8], -1.5, atol=1e-5)


def test_factorized_local_coefficients_matches_explicit_linear_chain() -> None:
    context_w_dec = torch.tensor([[1.0, 2.0], [-1.0, 0.5], [0.25, -2.0]])
    answer_w_enc = torch.tensor([[2.0, -1.0], [0.5, 3.0]])
    ridge_w = torch.tensor([[1.5, -0.25], [0.75, 2.0]])
    ridge_xsd = torch.tensor([2.0, 0.5])
    scale = torch.tensor([3.0, 0.25])
    answer_weight = torch.tensor([0.4, -1.2])
    mapper = SimpleNamespace(
        context_sae=SimpleNamespace(w_dec=context_w_dec, dict_size=3),
        answer_sae=SimpleNamespace(w_enc=answer_w_enc, dict_size=2),
        ridge={"W": ridge_w, "xsd": ridge_xsd},
        scale=scale,
    )
    got = factorized_local_coefficients(mapper, answer_weight)
    expected = (
        context_w_dec @ ((ridge_w @ (answer_w_enc @ (answer_weight * scale))) / ridge_xsd)
    ).numpy()
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_behavior_pursuit_uses_fit_rows_only_and_improves_with_k() -> None:
    rng = np.random.default_rng(17)
    z = rng.normal(size=(500, 12)).astype(np.float32)
    mapped = 4.0 + 2.0 * z[:, 1] - 1.0 * z[:, 7] + 0.5 * z[:, 10]
    train = np.zeros(500, dtype=bool)
    train[:300] = True
    raw = np.zeros(12)
    raw[[1, 7, 10, 4]] = [2.0, -1.0, 0.5, 0.1]
    fit_a = fit_behavior_pursuit(
        z,
        mapped,
        train,
        raw,
        candidates=8,
        k_ladder=(1, 2, 3),
        ridge_relative=0.0,
    )
    changed_eval = mapped.copy()
    changed_eval[~train] += rng.normal(scale=100.0, size=(~train).sum())
    fit_b = fit_behavior_pursuit(
        z,
        changed_eval,
        train,
        raw,
        candidates=8,
        k_ladder=(1, 2, 3),
        ridge_relative=0.0,
    )
    for k in fit_a.k_ladder:
        np.testing.assert_array_equal(
            fit_a.methods["gradient_pursuit"][k].support,
            fit_b.methods["gradient_pursuit"][k].support,
        )
        np.testing.assert_allclose(
            fit_a.methods["gradient_pursuit"][k].coefficients,
            fit_b.methods["gradient_pursuit"][k].coefficients,
        )
    scores = apply_behavior_pursuit(fit_a, z)
    err1 = np.square(scores["gradient_pursuit_k1"].numpy()[~train] - mapped[~train]).mean()
    err3 = np.square(scores["gradient_pursuit_k3"].numpy()[~train] - mapped[~train]).mean()
    assert err3 < err1 * 1e-4
    assert set(fit_a.candidates[fit_a.methods["gradient_pursuit"][3].support]) == {1, 7, 10}


def test_clustered_auc_delta_is_paired() -> None:
    labels = [0, 1, 0, 1, 0, 1]
    scores = [0.1, 0.8, 0.2, 0.7, 0.3, 0.9]
    got = clustered_auc_delta(
        labels,
        scores,
        scores,
        ["a", "a", "b", "b", "c", "c"],
        draws=100,
        seed=3,
    )
    assert got["auroc_delta"] == 0.0
    assert got["cluster_bootstrap_95ci"] == [0.0, 0.0]
    assert got["n_boot_valid"] == 100
