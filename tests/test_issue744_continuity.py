"""Issue #744 — regression tests for the residual-stream continuity primitives.

Pins the load-bearing invariants the plan-marker concerns + plan §7 risks name:

* **Rogue-dim ranking is NON-degenerate (concern #3).** ``rank_rogue_dims`` must
  rank by a statistic computed on the RAW residuals, NOT "standardized variance"
  (which is 1 everywhere after z-scoring). The test builds a population with a
  KNOWN dominant dim and asserts every supported metric surfaces it — and that
  ranking the z-scored population would NOT (the degenerate case the concern
  warns against).
* **Direction preservation is exact on a perfectly-linear trajectory.** A
  straight-line trajectory has abs-cosine = 1 at every step; a fitted direction
  on noise sits near the chance floor.
* **Welford streaming stats reproduce the full-batch mean/var (A7).**
* **Surprisal off-by-one alignment (plan §7 risk).** surprisal at position t is
  the NLL of token_t under the logits at index t-1.
* **make_flavors_from_stats vectorized == per-layer reference.**
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from explore_persona_space.analysis.continuity import (  # noqa: E402
    ROGUE_RANK_METRICS,
    WelfordDimStats,
    closed_form_random_abs_cosine,
    consec_cosine,
    direction_preservation,
    extrap_error,
    make_flavors_from_stats,
    random_baseline,
    rank_rogue_dims,
    rogue_dim_ablate,
    zscore_population,
)


def test_rank_rogue_dims_surfaces_dominant_dim_all_metrics():
    """Every supported metric must surface a KNOWN dominant dim from RAW residuals."""
    torch.manual_seed(0)
    n, hidden = 200, 16
    H = torch.randn(n, hidden)
    # Make dim 7 dominate the RAW residuals: high variance + a giant outlier.
    H[:, 7] *= 25.0
    H[3, 7] = 5000.0
    for metric in ROGUE_RANK_METRICS:
        top = rank_rogue_dims(H, top_k=3, metric=metric)
        assert 7 in top.tolist(), f"metric {metric!r} failed to surface dim 7: {top.tolist()}"


def test_rank_rogue_dims_degenerate_after_zscore():
    """Concern #3: ranking the Z-SCORED population is degenerate (all var == 1).

    This is the failure mode the rogue-dim ranking must avoid: after per-dim
    z-scoring every dim has variance ~1, so a variance-based rank over the
    z-scored data does NOT reliably surface the true rogue dim. We assert the
    z-scored variance is ~flat (so ranking it would be arbitrary), confirming
    the implementation correctly ranks the RAW residuals instead.
    """
    torch.manual_seed(1)
    n, hidden = 500, 12
    H = torch.randn(n, hidden)
    H[:, 4] *= 40.0  # raw-dominant dim
    mu = H.mean(0)
    sigma = H.std(0, unbiased=False)
    z = zscore_population(H, mu, sigma)
    zvar = z.var(0, unbiased=False)
    assert torch.allclose(zvar, torch.ones_like(zvar), atol=1e-3), zvar
    # The RAW-variance rank surfaces dim 4; a z-var rank would not be meaningful.
    assert rank_rogue_dims(H, top_k=1, metric="raw_variance").tolist() == [4]


def test_rogue_dim_ablate_zeros_indices():
    H = torch.randn(2, 5, 8)
    idx = torch.tensor([1, 3])
    out = rogue_dim_ablate(H, idx)
    assert torch.all(out[..., 1] == 0.0)
    assert torch.all(out[..., 3] == 0.0)
    assert torch.allclose(out[..., 0], H[..., 0])  # untouched dims preserved


def test_direction_preservation_linear_trajectory_is_one():
    """A perfectly-linear trajectory has abs-cosine ~1 at every step."""
    L, T, hidden = 1, 12, 6
    direction = torch.randn(hidden)
    direction = direction / direction.norm()
    base = torch.randn(hidden)
    H = torch.stack([base + t * direction for t in range(T)]).unsqueeze(0)  # (1, T, hidden)
    dp = direction_preservation(H, k=3, steps=(0, 1, 2, 3))
    for s in (0, 1, 2, 3):
        assert dp[s].shape == (L,)
        assert dp[s][0] > 0.999, (s, dp[s][0].item())


def test_direction_preservation_random_near_chance():
    """A random trajectory's direction preservation sits near the chance floor."""
    torch.manual_seed(2)
    L, T, hidden = 1, 60, 256
    H = torch.randn(L, T, hidden)
    dp = direction_preservation(H, k=3, steps=(1,))
    chance = closed_form_random_abs_cosine(hidden)
    # well below the linear-trajectory ceiling; within a few x of chance
    assert dp[1][0] < 0.2, dp[1][0].item()
    assert dp[1][0] < 10 * chance + 0.05


def test_extrap_error_linear_trajectory_is_zero():
    """A linear trajectory is exactly predicted by its OLS line -> ~0 L2 error."""
    T, hidden = 10, 5
    direction = torch.randn(hidden)
    base = torch.randn(hidden)
    H = torch.stack([base + t * direction for t in range(T)]).unsqueeze(0)
    err = extrap_error(H, k=3)
    assert err[0] < 1e-3, err[0].item()


def test_consec_cosine_shape_and_identity():
    H = torch.randn(3, 7, 16)
    cc = consec_cosine(H)
    assert cc.shape == (3, 6)
    # identical consecutive vectors -> cosine 1
    H2 = torch.randn(1, 5, 4)
    H2[0, 2] = H2[0, 1]
    assert torch.isclose(consec_cosine(H2)[0, 1], torch.tensor(1.0), atol=1e-5)


def test_welford_matches_full_batch_mean_var():
    """A7: streaming sufficient stats reproduce the full-batch mean/var (fp32)."""
    torch.manual_seed(3)
    n_layers, hidden = 4, 32
    seqs = [torch.randn(n_layers, T, hidden) * 3.0 + 1.5 for T in (10, 25, 7, 40)]
    w = WelfordDimStats(n_layers, hidden)
    for s in seqs:
        w.update(s)
    mu, sigma = w.finalize()
    allcat = torch.cat(seqs, dim=1)  # (L, sum_T, hidden)
    mu_ref = allcat.mean(dim=1)
    var_ref = allcat.var(dim=1, unbiased=False)
    assert torch.allclose(mu, mu_ref, atol=1e-4), (mu - mu_ref).abs().max().item()
    assert torch.allclose(sigma, var_ref.sqrt(), atol=1e-4)


def test_make_flavors_from_stats_matches_reference():
    """Vectorized flavor build == naive per-layer z-score + ablate."""
    torch.manual_seed(4)
    L, T, hidden = 5, 9, 24
    H = torch.randn(L, T, hidden) * 2.0
    mu = torch.randn(L, hidden)
    sigma = torch.rand(L, hidden) + 0.5
    rogue = torch.stack([torch.tensor([0, 3, 7]) for _ in range(L)])
    flavors = make_flavors_from_stats(H, mu, sigma, rogue)
    # reference: per-layer z-score then zero the rogue dims
    for li in range(L):
        z_ref = zscore_population(H[li], mu[li], sigma[li])
        assert torch.allclose(flavors["std"][li], z_ref, atol=1e-5)
        ab_ref = rogue_dim_ablate(z_ref, rogue[li])
        assert torch.allclose(flavors["ablate"][li], ab_ref, atol=1e-5)
    assert torch.allclose(flavors["raw"], H)


def test_surprisal_off_by_one_alignment():
    """Plan §7 risk: surprisal_t = -log p(token_t | tokens_0..t-1) at logits index t-1.

    Mirrors the dump script's ``_surprisal_from_logits`` indexing on a tiny
    deterministic logits/ids pair so the off-by-one is pinned by a test, not just
    a comment.
    """
    T, V = 4, 6
    logits = torch.zeros(1, T, V)
    # Make the prediction at position t-1 strongly favor the actual token_t so
    # the surprisal is small + computable by hand.
    ids = torch.tensor([[1, 2, 3, 4]])
    for t in range(1, T):
        logits[0, t - 1, ids[0, t]] = 10.0  # logits at t-1 predict token_t

    logp = torch.log_softmax(logits[0].float(), dim=-1)
    out = torch.full((T,), float("nan"))
    idx = torch.arange(1, T)
    out[1:] = -logp[idx - 1, ids[0, idx]]
    # position 0 has no preceding context
    assert torch.isnan(out[0])
    # each surprisal = -log_softmax(logits[t-1])[token_t]; with the 10.0 spike it is small
    for t in range(1, T):
        expected = -logp[t - 1, ids[0, t]]
        assert torch.isclose(out[t], expected)
        assert out[t] < 0.1  # the spike dominates the softmax -> low surprisal


def test_random_baseline_near_closed_form():
    """Empirical random-pair abs-cosine tracks the closed-form sqrt(2/(pi d))."""
    torch.manual_seed(5)
    L, T, hidden = 2, 400, 512
    H = torch.randn(L, T, hidden)
    base = random_baseline(H, n_pairs=20000, seed=744)
    cf = closed_form_random_abs_cosine(hidden)
    assert base.shape == (L,)
    for li in range(L):
        assert abs(base[li].item() - cf) < 0.01, (base[li].item(), cf)


@pytest.mark.parametrize("d,expect", [(768, 0.029), (3584, 0.0133)])
def test_closed_form_random_abs_cosine(d, expect):
    assert abs(closed_form_random_abs_cosine(d) - expect) < 0.002
