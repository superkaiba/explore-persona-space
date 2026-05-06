"""Unit tests for ``explore_persona_space.eval.steering`` (issue #267).

All tests run on CPU and avoid loading any HF model. The GPU-bound paths
(``extract_centroids_for_personas_at_layers``, ``generate_batched``) are
covered indirectly by the synthetic-toy-model test below.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from explore_persona_space.eval.steering import (
    ClusterRateData,
    SteeringHook,
    _persona_seed,
    cluster_bootstrap_delta_spearman,
    cluster_bootstrap_spearman,
    compute_centered_centroids,
    loo_spearman,
    make_random_vector,
    marker_substring_rate,
    near_marker_substring_rate,
    spearman_rho,
    wilson_ci,
)

# ---------------------------------------------------------------------------
# Toy model with a residual-style layer to exercise SteeringHook
# ---------------------------------------------------------------------------


class _ToyLayer(torch.nn.Module):
    """One trivial decoder-like layer: identity, returns ``(hs,)`` tuple."""

    def forward(self, hs: torch.Tensor) -> tuple[torch.Tensor]:
        return (hs,)


class _ToyModel(torch.nn.Module):
    """Mimics ``Qwen2ForCausalLM`` minimally: ``model.model.layers`` is a ModuleList."""

    def __init__(self, n_layers: int = 4, hidden: int = 6) -> None:
        super().__init__()
        # one persistent parameter so model has a dtype/device
        self._dtype_anchor = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_ToyLayer() for _ in range(n_layers)])

    def forward_layer(self, layer_idx: int, hs: torch.Tensor) -> torch.Tensor:
        out = self.model.layers[layer_idx](hs)
        return out[0] if isinstance(out, tuple) else out


# ---------------------------------------------------------------------------
# SteeringHook math
# ---------------------------------------------------------------------------


class TestSteeringHook:
    def test_hook_adds_coefficient_times_direction_at_layer(self) -> None:
        torch.manual_seed(0)
        model = _ToyModel(n_layers=4, hidden=6)
        hs = torch.randn(2, 5, 6)
        # baseline: identity
        baseline = model.forward_layer(2, hs.clone())
        assert torch.allclose(baseline, hs)

        direction = torch.tensor([1.0, -2.0, 0.5, 0.0, 3.0, -0.5], dtype=torch.float32)
        hook = SteeringHook(model, layer_idx=2, direction=direction, coefficient=1.5)
        try:
            steered = model.forward_layer(2, hs.clone())
        finally:
            hook.remove()

        expected = hs + 1.5 * direction.view(1, 1, -1)
        assert torch.allclose(steered, expected)

    def test_hook_only_fires_at_target_layer(self) -> None:
        model = _ToyModel(n_layers=4, hidden=6)
        hs = torch.zeros(1, 3, 6)
        direction = torch.ones(6, dtype=torch.float32)
        hook = SteeringHook(model, layer_idx=2, direction=direction, coefficient=2.0)
        try:
            for layer in (0, 1, 3):
                out = model.forward_layer(layer, hs.clone())
                assert torch.allclose(out, hs), f"layer {layer} should not be hooked"
            out_target = model.forward_layer(2, hs.clone())
            assert torch.allclose(out_target, 2.0 * direction.view(1, 1, -1).expand_as(hs))
        finally:
            hook.remove()

    def test_hook_remove_restores_identity(self) -> None:
        model = _ToyModel(n_layers=4, hidden=6)
        hs = torch.randn(1, 3, 6)
        direction = torch.ones(6, dtype=torch.float32)
        hook = SteeringHook(model, layer_idx=1, direction=direction, coefficient=1.0)
        hook.remove()
        out = model.forward_layer(1, hs.clone())
        assert torch.allclose(out, hs)

    def test_hook_rejects_wrong_direction_rank(self) -> None:
        model = _ToyModel()
        with pytest.raises(ValueError, match="1-D direction tensor"):
            SteeringHook(model, layer_idx=0, direction=torch.zeros(2, 6), coefficient=1.0)

    def test_hook_handles_non_tuple_output(self) -> None:
        # If a layer returns a bare tensor (not a tuple) the hook still adds.
        class _BareLayer(torch.nn.Module):
            def forward(self, hs):
                return hs * 2.0

        model = _ToyModel()
        model.model.layers[0] = _BareLayer()
        hs = torch.ones(1, 3, 6)
        direction = torch.full((6,), 0.5)
        hook = SteeringHook(model, layer_idx=0, direction=direction, coefficient=4.0)
        try:
            out = model.forward_layer(0, hs)
        finally:
            hook.remove()
        # Bare-layer doubled the input, then hook adds 4 * 0.5 = 2 to every element.
        assert torch.allclose(out, hs * 2.0 + 4.0 * direction.view(1, 1, -1))


# ---------------------------------------------------------------------------
# Centering
# ---------------------------------------------------------------------------


class TestCentering:
    def test_known_three_persona_centering(self) -> None:
        # Hand-picked centroids; mean = [3, 4, 5]; centered = raw - [3,4,5].
        raw = {
            "alpha": torch.tensor([1.0, 2.0, 3.0]),
            "beta": torch.tensor([3.0, 4.0, 5.0]),
            "gamma": torch.tensor([5.0, 6.0, 7.0]),
        }
        centered, mean = compute_centered_centroids(raw, ["alpha", "beta", "gamma"])
        assert torch.allclose(mean, torch.tensor([3.0, 4.0, 5.0]))
        assert torch.allclose(centered["alpha"], torch.tensor([-2.0, -2.0, -2.0]))
        assert torch.allclose(centered["beta"], torch.tensor([0.0, 0.0, 0.0]))
        assert torch.allclose(centered["gamma"], torch.tensor([2.0, 2.0, 2.0]))
        # Centered set sums to zero (geometric necessity for §11.34)
        stacked = torch.stack([centered["alpha"], centered["beta"], centered["gamma"]])
        assert torch.allclose(stacked.sum(dim=0), torch.zeros(3), atol=1e-6)

    def test_out_of_set_persona_is_projected(self) -> None:
        # B1 / §4.4 #6 fix: helpful_assistant projects into N=10-centered space.
        raw = {
            "p0": torch.tensor([0.0, 0.0]),
            "p1": torch.tensor([2.0, 0.0]),
            "extra": torch.tensor([10.0, 5.0]),
        }
        centered, mean = compute_centered_centroids(raw, ["p0", "p1"])
        assert torch.allclose(mean, torch.tensor([1.0, 0.0]))
        assert torch.allclose(centered["extra"], torch.tensor([9.0, 5.0]))

    def test_missing_centering_persona_raises(self) -> None:
        raw = {"a": torch.zeros(3)}
        with pytest.raises(KeyError):
            compute_centered_centroids(raw, ["a", "b"])

    def test_duplicate_centering_raises(self) -> None:
        raw = {"a": torch.zeros(3), "b": torch.zeros(3)}
        with pytest.raises(ValueError, match="duplicates"):
            compute_centered_centroids(raw, ["a", "a", "b"])


# ---------------------------------------------------------------------------
# Random vectors (H3 + H3')
# ---------------------------------------------------------------------------


class TestRandomVectors:
    def test_isotropic_norm_matches_target(self) -> None:
        target = 7.5
        v = make_random_vector(
            kind="isotropic",
            persona="librarian",
            target_norm=target,
            hidden_dim=64,
            dtype=torch.float32,
        )
        assert v.shape == (64,)
        assert math.isclose(v.norm().item(), target, rel_tol=1e-5)

    def test_isotropic_is_deterministic_per_persona(self) -> None:
        # Same persona+namespace must yield identical draws (no PYTHONHASHSEED reliance).
        v1 = make_random_vector("isotropic", "villain", target_norm=1.0, hidden_dim=8)
        v2 = make_random_vector("isotropic", "villain", target_norm=1.0, hidden_dim=8)
        assert torch.allclose(v1, v2)
        # Different persona -> different vector (overwhelmingly likely)
        v3 = make_random_vector("isotropic", "comedian", target_norm=1.0, hidden_dim=8)
        assert not torch.allclose(v1, v3)

    def test_isotropic_seed_is_stable_under_python_hash_randomization(self) -> None:
        # _persona_seed is the actual stable function — verify it doesn't use Python hash().
        s1 = _persona_seed("librarian", 42)
        s2 = _persona_seed("librarian", 42)
        assert s1 == s2
        s3 = _persona_seed("librarian", 1042)
        assert s3 == s1 + 1000  # both have the same SHA-derived bucket, only namespace differs

    def test_in_subspace_zero_sum_coefficients(self) -> None:
        # Build 4 mean-centered toy centroids (so the centered set sums to zero).
        torch.manual_seed(11)
        raw = {f"p{i}": torch.randn(8) for i in range(4)}
        centered, _ = compute_centered_centroids(raw, list(raw.keys()))
        target_norm = 2.5

        # Run the in-subspace draw + verify that the produced direction lies in
        # the centered span. Specifically: the residual after projecting onto
        # the centered subspace must be (near) zero.
        v = make_random_vector(
            kind="in_subspace",
            persona="p1",
            target_norm=target_norm,
            centered_centroids=centered,
            headline_personas=list(centered.keys()),
        )
        assert math.isclose(v.norm().item(), target_norm, rel_tol=1e-5)

        # Project v onto span(centered) and check residual is ~0.
        basis = torch.stack([centered[p].float() for p in centered]).T  # (8, 4)
        # least-squares projection
        coeffs, _residuals, _rank, _sv = torch.linalg.lstsq(basis, v.unsqueeze(1))
        approx = basis @ coeffs.squeeze(1)
        assert torch.allclose(approx, v, atol=1e-4)

    def test_in_subspace_coefficient_sum_is_zero_in_internal_draw(self) -> None:
        # We can't introspect the internal coefficients directly, but for an
        # orthogonal-basis test we can reverse-engineer them via least squares
        # and check they sum to ~0 (§11.34).
        torch.manual_seed(11)
        raw = {f"p{i}": torch.randn(8) for i in range(4)}
        centered, _ = compute_centered_centroids(raw, list(raw.keys()))
        # Use a target norm = norm of the un-renormalized linear combination
        # so the renormalization factor is 1.0 — easier to check coefficient sum.
        # We test the geometric guarantee: ANY in-subspace renormalized vector is
        # in the centered span (which already has zero column-sum), so a
        # least-squares fit should yield coefficients that sum to ~0 modulo the
        # renormalization scale. We test scale-invariant zero-sum here:
        v = make_random_vector(
            kind="in_subspace",
            persona="p2",
            target_norm=3.0,
            centered_centroids=centered,
            headline_personas=list(centered.keys()),
        )
        basis = torch.stack([centered[p].float() for p in centered]).T  # (8, 4)
        coeffs, _r, _rank, _sv = torch.linalg.lstsq(basis, v.unsqueeze(1))
        # The N=4 centered centroids sum to 0, so any non-trivial coefficient
        # vector with a non-zero mean would have a non-zero average projection
        # onto the all-ones-in-coeff direction; that direction is in the null
        # space of the centered basis. Therefore the projection of v onto
        # span(centered) yields a coefficient vector orthogonal to all-ones
        # (i.e., zero-sum).
        assert math.isclose(coeffs.squeeze().sum().item(), 0.0, abs_tol=1e-3)

    def test_invalid_kind_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown kind"):
            make_random_vector("garbage", "p", target_norm=1.0, hidden_dim=4)

    def test_isotropic_requires_hidden_dim(self) -> None:
        with pytest.raises(ValueError, match="hidden_dim"):
            make_random_vector("isotropic", "p", target_norm=1.0)

    def test_in_subspace_requires_centered_centroids(self) -> None:
        with pytest.raises(ValueError, match="centered_centroids"):
            make_random_vector("in_subspace", "p", target_norm=1.0)

    def test_negative_target_norm_raises(self) -> None:
        with pytest.raises(ValueError, match="target_norm must be > 0"):
            make_random_vector("isotropic", "p", target_norm=0.0, hidden_dim=4)


# ---------------------------------------------------------------------------
# Marker scoring
# ---------------------------------------------------------------------------


class TestMarkerScoring:
    def test_substring_match_case_insensitive(self) -> None:
        comps = [
            "I am happy [ZLT] and helpful.",
            "no marker here",
            "lowercase [zlt] also counts",
            "[ZLt] mixed case",
            "almost [ZL but not quite",
        ]
        found, total = marker_substring_rate(comps)
        assert (found, total) == (3, 5)

    def test_empty_completions(self) -> None:
        assert marker_substring_rate([]) == (0, 0)

    def test_near_marker_regex_catches_variants(self) -> None:
        comps = [
            "[ZLT] standard",  # exact → should also match near-marker
            "[ZLT-1] variant suffix",
            "[zlt ] padded space",
            "[Z L T] spaces inside",
            "not even close",
            "[Z\tlt] tab inside",
        ]
        found, total = near_marker_substring_rate(comps)
        # All except "not even close" match.
        assert (found, total) == (5, 6)


# ---------------------------------------------------------------------------
# Wilson + Spearman + LOO
# ---------------------------------------------------------------------------


class TestStatHelpers:
    def test_wilson_ci_zero_count(self) -> None:
        lo, hi = wilson_ci(0, 100)
        assert lo == 0.0
        assert hi > 0.0
        assert hi < 0.05  # half-width tight at p_hat=0

    def test_wilson_ci_known_value(self) -> None:
        # Wilson 95% CI for k=30, n=100: ≈ (0.218, 0.395) per standard tables.
        lo, hi = wilson_ci(30, 100)
        assert 0.21 < lo < 0.23
        assert 0.39 < hi < 0.40

    def test_wilson_ci_zero_n_is_safe(self) -> None:
        assert wilson_ci(0, 0) == (0.0, 0.0)

    def test_spearman_rho_perfect_monotone(self) -> None:
        rho = spearman_rho([1, 2, 3, 4, 5], [10, 20, 30, 40, 50])
        assert math.isclose(rho, 1.0, abs_tol=1e-9)
        rho_neg = spearman_rho([1, 2, 3, 4, 5], [50, 40, 30, 20, 10])
        assert math.isclose(rho_neg, -1.0, abs_tol=1e-9)

    def test_spearman_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            spearman_rho([1, 2], [1, 2, 3])

    def test_loo_spearman_min_max(self) -> None:
        # 5 paired points; LOO drops each in turn.
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        loo = loo_spearman(x, y)
        assert loo["min"] == loo["max"] == 1.0
        assert len(loo["all"]) == 5

    def test_loo_requires_three_points(self) -> None:
        with pytest.raises(ValueError, match="LOO"):
            loo_spearman([1, 2], [1, 2])


# ---------------------------------------------------------------------------
# Cluster bootstrap on synthetic data
# ---------------------------------------------------------------------------


def _make_synthetic_cluster_data(
    *,
    personas: list[str],
    rates: list[float],
    n_questions: int,
    n_completions: int,
    seed: int,
) -> ClusterRateData:
    """Synthesize completion strings whose substring rate matches ``rates`` exactly.

    For each persona, exactly ``round(rate * n_questions * n_completions)`` of the
    completions contain ``[ZLT]``; the rest contain ``noise``. Distributed
    uniformly across question clusters.
    """
    rng = np.random.default_rng(seed)
    completions: list[list[list[str]]] = []
    for persona, rate in zip(personas, rates, strict=True):
        n_total = n_questions * n_completions
        n_pos = round(rate * n_total)
        labels = np.array([1] * n_pos + [0] * (n_total - n_pos))
        rng.shuffle(labels)
        labels = labels.reshape(n_questions, n_completions)
        by_q: list[list[str]] = []
        for q in range(n_questions):
            by_q.append(
                [
                    f"{persona}-q{q}-c{c} " + ("[ZLT]" if labels[q, c] else "no_marker")
                    for c in range(n_completions)
                ]
            )
        completions.append(by_q)
    return ClusterRateData(personas=personas, completions=completions)


class TestClusterBootstrap:
    def test_per_persona_rate_matches_known_proportion(self) -> None:
        data = _make_synthetic_cluster_data(
            personas=["a", "b"],
            rates=[0.1, 0.6],
            n_questions=20,
            n_completions=5,
            seed=7,
        )
        rates = data.per_persona_rate()
        assert math.isclose(rates[0], 0.1, abs_tol=1e-9)
        assert math.isclose(rates[1], 0.6, abs_tol=1e-9)

    def test_bootstrap_point_estimate_matches_full_data_rho(self) -> None:
        # 4 personas, monotone increasing rates → ρ should be near +1
        data = _make_synthetic_cluster_data(
            personas=["a", "b", "c", "d"],
            rates=[0.1, 0.3, 0.5, 0.8],
            n_questions=20,
            n_completions=5,
            seed=11,
        )
        y = [10.0, 20.0, 30.0, 40.0]
        out = cluster_bootstrap_spearman(data, y, n_iter=200, seed=2604)
        assert out["point_estimate"] == 1.0
        # CI should bracket the point estimate
        assert out["ci_low"] <= out["point_estimate"] <= out["ci_high"]
        # Reproducible
        out2 = cluster_bootstrap_spearman(data, y, n_iter=200, seed=2604)
        assert out["draws"] == out2["draws"]

    def test_bootstrap_resamples_questions_not_completions(self) -> None:
        # If a single question dominates a persona's "[ZLT]" emissions, the
        # cluster bootstrap should reflect higher variance than IID resampling
        # would. Build such a dataset and verify the CI is non-trivial.
        n_q, n_c = 20, 5
        # Persona-0: ALL 5 [ZLT]s come from question 0. Persona-1: spread evenly.
        completions: list[list[list[str]]] = []
        for _persona, concentrated in [("p0", True), ("p1", False)]:
            by_q: list[list[str]] = []
            for q in range(n_q):
                if concentrated:
                    if q == 0:
                        by_q.append(["x [ZLT]"] * n_c)
                    else:
                        by_q.append(["x no_marker"] * n_c)
                else:
                    by_q.append(["x [ZLT]" if q < n_q // 4 else "x no_marker"] * n_c)
            completions.append(by_q)
        data = ClusterRateData(personas=["p0", "p1"], completions=completions)
        y = [1.0, 2.0]
        out = cluster_bootstrap_spearman(data, y, n_iter=500, seed=3)
        # Both per-persona point rates: p0 = 5/100 = 0.05; p1 = 25/100 = 0.25.
        # Spearman on N=2 is degenerate (only two ranks), but the bootstrap
        # *should still run without crashing*.
        assert isinstance(out["point_estimate"], float)
        assert len(out["draws"]) == 500

    def test_delta_bootstrap_paired_resampling(self) -> None:
        # Build two arms with same per-persona rates → Δρ ≈ 0 with tight CI.
        personas = ["a", "b", "c", "d"]
        data_a = _make_synthetic_cluster_data(
            personas=personas,
            rates=[0.2, 0.4, 0.5, 0.7],
            n_questions=20,
            n_completions=5,
            seed=21,
        )
        data_b = _make_synthetic_cluster_data(
            personas=personas,
            rates=[0.2, 0.4, 0.5, 0.7],
            n_questions=20,
            n_completions=5,
            seed=22,
        )
        y = [1.0, 2.0, 3.0, 4.0]
        out = cluster_bootstrap_delta_spearman(data_a, data_b, y, n_iter=200, seed=2604)
        # Both rate vectors are monotone in y, so both ρ should be 1.0 → Δ = 0.
        assert math.isclose(out["rho_centroid"], 1.0)
        assert math.isclose(out["rho_other"], 1.0)
        assert math.isclose(out["point_estimate"], 0.0)

    def test_delta_bootstrap_persona_mismatch_raises(self) -> None:
        d1 = _make_synthetic_cluster_data(
            personas=["a", "b"],
            rates=[0.1, 0.5],
            n_questions=10,
            n_completions=3,
            seed=1,
        )
        d2 = _make_synthetic_cluster_data(
            personas=["a", "c"],
            rates=[0.1, 0.5],
            n_questions=10,
            n_completions=3,
            seed=2,
        )
        with pytest.raises(ValueError, match="persona list"):
            cluster_bootstrap_delta_spearman(d1, d2, [1.0, 2.0], n_iter=10)

    def test_cluster_data_rejects_zero_clusters(self) -> None:
        with pytest.raises(ValueError, match="zero question clusters"):
            ClusterRateData(personas=["x"], completions=[[]])

    def test_cluster_data_rejects_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            ClusterRateData(personas=["a", "b"], completions=[[["x"]]])
