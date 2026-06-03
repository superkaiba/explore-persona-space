# ruff: noqa: RUF002, RUF003  # em-dash + Qwen marker " ※" + Greek intentional
"""Task #477 v4 — marker-channel Bernoulli KL post-hoc transform tests.

Pins the v4 headline DV:
  * ``marker_channel_bernoulli_kl(p_trained, p_base)`` — pointwise 2-class KL.
  * ``aggregate_bystander_marker_channel_kl(checkpoint)`` — mean over held-out
    (persona × q) leaves.
  * ``aggregate_source_self_marker_channel_kl(checkpoint)`` — H1 conditioning
    covariate (the partial's continuous regressor).
  * ``attach_marker_channel_aggregates(traj)`` — idempotent per-checkpoint
    aggregation across an entire trajectory.json.

All tests are CPU-only, sub-second, no torch / no vLLM imports.
"""

from __future__ import annotations

import math

import pytest


def test_marker_channel_kl_zero_when_p_equals_q() -> None:
    """KL(Bernoulli(p) ‖ Bernoulli(p)) == 0."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472.analyze import (
        marker_channel_bernoulli_kl,
    )

    for p in (1e-6, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0 - 1e-6):
        assert marker_channel_bernoulli_kl(p, p) == pytest.approx(0.0, abs=1e-9)


def test_marker_channel_kl_known_analytic_value() -> None:
    """p=0.5, p_base=0.001: 0.5*log(500) + 0.5*log(0.5/0.999) ≈ 2.7612 nats."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472.analyze import (
        marker_channel_bernoulli_kl,
    )

    kl = marker_channel_bernoulli_kl(0.5, 0.001)
    expected = 0.5 * math.log(0.5 / 0.001) + 0.5 * math.log(0.5 / 0.999)
    assert kl == pytest.approx(expected, abs=1e-9)
    # Hand-computed: 0.5 * 6.2146 + 0.5 * (-0.6931) = 3.1073 - 0.3466 = 2.7607.
    assert kl == pytest.approx(2.7612, abs=1e-3)


def test_marker_channel_kl_nonnegative() -> None:
    """KL ≥ 0 for arbitrary (p, q) in (0, 1)^2."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472.analyze import (
        marker_channel_bernoulli_kl,
    )

    grid = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    for p in grid:
        for q in grid:
            assert marker_channel_bernoulli_kl(p, q) >= -1e-12


def test_marker_channel_kl_eps_clamp_keeps_finite_at_p_zero_and_one() -> None:
    """No NaN / -inf at p=0 or p=1 (the eps clamp absorbs log(0))."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472.analyze import (
        marker_channel_bernoulli_kl,
    )

    assert math.isfinite(marker_channel_bernoulli_kl(0.0, 0.001))
    assert math.isfinite(marker_channel_bernoulli_kl(1.0, 0.001))
    assert math.isfinite(marker_channel_bernoulli_kl(0.5, 0.0))
    assert math.isfinite(marker_channel_bernoulli_kl(0.5, 1.0))


def test_marker_channel_kl_rejects_out_of_range() -> None:
    from explore_persona_space.experiments.contrastive_neg_geometry_472.analyze import (
        marker_channel_bernoulli_kl,
    )

    with pytest.raises(ValueError, match=r"p_trained=1\.5"):
        marker_channel_bernoulli_kl(1.5, 0.5)
    with pytest.raises(ValueError, match=r"p_base=-0\.1"):
        marker_channel_bernoulli_kl(0.5, -0.1)


def test_marker_channel_kl_matches_scipy_kl_div() -> None:
    """Agreement with scipy ``rel_entr`` on the 2-class distribution."""
    from scipy.special import rel_entr

    from explore_persona_space.experiments.contrastive_neg_geometry_472.analyze import (
        marker_channel_bernoulli_kl,
    )

    for p, q in [(0.3, 0.05), (0.7, 0.4), (0.95, 0.5), (0.05, 0.95)]:
        ours = marker_channel_bernoulli_kl(p, q)
        scipy_kl = float(rel_entr([p, 1 - p], [q, 1 - q]).sum())
        assert ours == pytest.approx(scipy_kl, abs=1e-9)


def test_aggregate_bystander_emits_source_self_and_bystander_per_checkpoint() -> None:
    """``attach_marker_channel_aggregates`` adds 3 keys per checkpoint."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472.analyze import (
        attach_marker_channel_aggregates,
    )

    # Construct a 2-checkpoint trajectory with 2 personas × 2 questions per
    # held-out persona block + a source_self mean block per checkpoint.
    # log(0.5)=-0.6931; log(0.001)=-6.9078.
    leaf_trained = {"g_logp": math.log(0.5), "b_logp": math.log(0.001)}
    leaf_calm = {"g_logp": math.log(0.01), "b_logp": math.log(0.001)}
    traj = {
        "checkpoints": [
            {
                "frac": 0.5,
                "step": 50,
                "source_self": {
                    "g_logp_mean": math.log(0.6),
                    "b_logp_mean": math.log(0.001),
                },
                "held_out": {
                    "p1": {"q1": leaf_trained, "q2": leaf_calm},
                    "p2": {"q1": leaf_trained, "q2": leaf_calm},
                },
            },
            {
                "frac": 1.0,
                "step": 100,
                "source_self": {
                    "g_logp_mean": math.log(0.9),
                    "b_logp_mean": math.log(0.001),
                },
                "held_out": {
                    "p1": {"q1": leaf_trained, "q2": leaf_trained},
                    "p2": {"q1": leaf_trained, "q2": leaf_trained},
                },
            },
        ]
    }
    out = attach_marker_channel_aggregates(traj)
    for ck in out["checkpoints"]:
        assert "source_self_marker_channel_kl" in ck
        assert "mean_bystander_marker_channel_kl" in ck
        assert "mean_bystander_full_vocab_kl" in ck
        assert ck["source_self_marker_channel_kl"] > 0
        assert ck["mean_bystander_marker_channel_kl"] >= 0
        # No `kl` key on the leaves → full-vocab aggregate is None.
        assert ck["mean_bystander_full_vocab_kl"] is None

    # Second checkpoint is higher source emission → higher source-self KL.
    assert (
        out["checkpoints"][1]["source_self_marker_channel_kl"]
        > out["checkpoints"][0]["source_self_marker_channel_kl"]
    )
    # Second checkpoint has heavier bystander leakage (all leaves at p=0.5)
    # → higher mean bystander KL.
    assert (
        out["checkpoints"][1]["mean_bystander_marker_channel_kl"]
        > out["checkpoints"][0]["mean_bystander_marker_channel_kl"]
    )


def test_aggregate_bystander_includes_full_vocab_when_kl_present() -> None:
    """If leaves carry a ``kl`` value, the full-vocab aggregate is the mean."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472.analyze import (
        attach_marker_channel_aggregates,
    )

    leaf = {"g_logp": math.log(0.3), "b_logp": math.log(0.05), "kl": 0.5}
    traj = {
        "checkpoints": [
            {
                "frac": 1.0,
                "step": 100,
                "source_self": {
                    "g_logp_mean": math.log(0.8),
                    "b_logp_mean": math.log(0.001),
                },
                "held_out": {
                    "p1": {"q1": dict(leaf, kl=0.4), "q2": dict(leaf, kl=0.6)},
                },
            }
        ]
    }
    out = attach_marker_channel_aggregates(traj)
    assert out["checkpoints"][0]["mean_bystander_full_vocab_kl"] == pytest.approx(0.5)


def test_aggregate_bystander_fails_loud_on_empty_held_out() -> None:
    from explore_persona_space.experiments.contrastive_neg_geometry_472.analyze import (
        aggregate_bystander_marker_channel_kl,
    )

    with pytest.raises(RuntimeError, match="0 bystander leaves"):
        aggregate_bystander_marker_channel_kl({"frac": 1.0, "step": 100, "held_out": {}})


# ── Cross-DV agreement gate (plan v4 §6 discipline #9) ──────────────────────


def test_rank_agreement_marker_vs_full_vocab_agreement() -> None:
    """Strong positive Spearman across kept cells → 'agreement'."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472.analyze import (
        rank_agreement_marker_vs_full_vocab,
    )

    # 4 cells, both DVs rank in the same order.
    kept = [
        {
            "mean_bystander_marker_channel_kl_at_picked_step": 0.10,
            "mean_bystander_full_vocab_kl_at_picked_step": 0.05,
        },
        {
            "mean_bystander_marker_channel_kl_at_picked_step": 0.20,
            "mean_bystander_full_vocab_kl_at_picked_step": 0.12,
        },
        {
            "mean_bystander_marker_channel_kl_at_picked_step": 0.30,
            "mean_bystander_full_vocab_kl_at_picked_step": 0.20,
        },
        {
            "mean_bystander_marker_channel_kl_at_picked_step": 0.40,
            "mean_bystander_full_vocab_kl_at_picked_step": 0.35,
        },
    ]
    out = rank_agreement_marker_vs_full_vocab(kept)
    assert out["verdict"] == "agreement"
    assert out["cross_dv_rank_spearman"] >= 0.70


def test_rank_agreement_divergence_downgrades() -> None:
    """Marker-channel and full-vocab disagree → 'divergence — downgrade H1'."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472.analyze import (
        rank_agreement_marker_vs_full_vocab,
    )

    # Anti-correlated: marker goes up, full-vocab goes down.
    kept = [
        {
            "mean_bystander_marker_channel_kl_at_picked_step": 0.10,
            "mean_bystander_full_vocab_kl_at_picked_step": 0.40,
        },
        {
            "mean_bystander_marker_channel_kl_at_picked_step": 0.20,
            "mean_bystander_full_vocab_kl_at_picked_step": 0.30,
        },
        {
            "mean_bystander_marker_channel_kl_at_picked_step": 0.30,
            "mean_bystander_full_vocab_kl_at_picked_step": 0.20,
        },
        {
            "mean_bystander_marker_channel_kl_at_picked_step": 0.40,
            "mean_bystander_full_vocab_kl_at_picked_step": 0.10,
        },
    ]
    out = rank_agreement_marker_vs_full_vocab(kept)
    assert out["verdict"] == "divergence — downgrade H1"
    assert out["cross_dv_rank_spearman"] < 0.70


def test_rank_agreement_missing_kl_returns_kl_not_computed() -> None:
    """Any cell with None full-vocab KL → gate skipped, verdict explains."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472.analyze import (
        rank_agreement_marker_vs_full_vocab,
    )

    kept = [
        {
            "cell": "c1",
            "mean_bystander_marker_channel_kl_at_picked_step": 0.1,
            "mean_bystander_full_vocab_kl_at_picked_step": None,
        },
        {
            "cell": "c2",
            "mean_bystander_marker_channel_kl_at_picked_step": 0.2,
            "mean_bystander_full_vocab_kl_at_picked_step": 0.1,
        },
    ]
    out = rank_agreement_marker_vs_full_vocab(kept)
    assert out["verdict"] == "kl not computed"
    assert out["missing_cells"] == ["c1"]
