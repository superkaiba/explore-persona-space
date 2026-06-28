"""CPU unit tests for issue #685 Phase-B geometry metrics.

Covers the consistency-cosine math (a perfectly-aligned Delta set -> cos 1.0; an
orthogonal pair -> cos 0.0), the consistency null (near-orthogonal in high dim),
PC1 share (rank-1 Delta matrix -> share 1.0), relative magnitude, projection onto
a known direction, and the name-aligned ``behavior_shift_metrics`` end-to-end
shape over a tiny synthetic vector set.
"""

import math

import numpy as np
import torch

from explore_persona_space.analysis.issue685.metrics import (
    behavior_shift_metrics,
    consistency_null,
    mean_pairwise_cosine,
    pc1_variance_share,
    project_onto_direction,
    relative_magnitude,
)


def test_mean_pairwise_cosine_aligned_is_one():
    # All rows the same direction (different magnitudes) -> cos 1.0.
    d = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    assert math.isclose(mean_pairwise_cosine(d), 1.0, abs_tol=1e-6)


def test_mean_pairwise_cosine_orthogonal_is_zero():
    d = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert math.isclose(mean_pairwise_cosine(d), 0.0, abs_tol=1e-6)


def test_mean_pairwise_cosine_antiparallel_is_minus_one():
    d = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    assert math.isclose(mean_pairwise_cosine(d), -1.0, abs_tol=1e-6)


def test_mean_pairwise_cosine_single_row_is_nan():
    assert math.isnan(mean_pairwise_cosine(torch.tensor([[1.0, 2.0]])))


def test_pc1_variance_share_rank1_is_one():
    # Rank-1 Delta matrix: every row a scalar multiple of one direction.
    base = torch.tensor([1.0, 2.0, -1.0, 0.5])
    d = torch.stack([c * base for c in (1.0, 2.0, 3.0, -1.0)])
    assert math.isclose(pc1_variance_share(d), 1.0, abs_tol=1e-5)


def test_pc1_variance_share_isotropic_below_one():
    g = torch.Generator().manual_seed(0)
    d = torch.randn(8, 64, generator=g)
    share = pc1_variance_share(d)
    assert 0.0 < share < 0.6  # isotropic noise has no dominant direction


def test_consistency_null_near_zero_in_high_dim():
    null = consistency_null(hidden_dim=3584, n_context=10, n_perm=200, seed=42)
    assert abs(null["mean"]) < 0.02
    assert null["p95"] < 0.1
    assert math.isclose(null["expected_abs_scale"], 1.0 / math.sqrt(3584), rel_tol=1e-6)
    assert null["n_perm"] == 200


def test_consistency_null_deterministic_under_seed():
    a = consistency_null(512, 6, n_perm=50, seed=7)
    b = consistency_null(512, 6, n_perm=50, seed=7)
    assert a["mean"] == b["mean"] and a["p95"] == b["p95"]


def test_relative_magnitude_shapes_and_denominator():
    bank = torch.tensor([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])  # spreads {5,10,5}
    deltas = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]])  # norms {1,2,5}
    out = relative_magnitude(deltas, bank)
    assert math.isclose(out["median_spread"], 5.0, abs_tol=1e-6)
    assert len(out["per_context"]) == 3
    assert math.isclose(out["per_context"][2], 1.0, abs_tol=1e-6)  # 5/5
    assert math.isclose(out["max"], 1.0, abs_tol=1e-6)


def test_project_onto_direction_aligned_is_one_orthogonal_zero():
    u = torch.tensor([1.0, 0.0, 0.0])
    deltas = torch.tensor([[2.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.0]])
    out = project_onto_direction(deltas, u)
    assert math.isclose(out["per_context"][0], 1.0, abs_tol=1e-6)  # aligned
    assert math.isclose(out["per_context"][1], 0.0, abs_tol=1e-6)  # orthogonal
    assert out["per_context"][2] == 0.0  # zero Delta -> 0.0, never NaN


def _toy_vectors(hidden_dim=32, seed=0):
    """Build a tiny {context: {layer: vec}} bare + aug set with a planted direction.

    For behavior 'planted', every context's Delta points along the SAME unit
    direction (so consistency cos -> 1.0); for 'noise', each context's Delta is
    independent random (consistency near 0).
    """
    g = torch.Generator().manual_seed(seed)
    contexts = ["c0", "c1", "c2", "c3"]
    behaviors = ["planted", "noise"]
    layers = [0, 1]
    planted_dir = torch.nn.functional.normalize(torch.randn(hidden_dim, generator=g), dim=0)

    bare = {c: {L: torch.randn(hidden_dim, generator=g) for L in layers} for c in contexts}
    aug = {}
    for c in contexts:
        for b in behaviors:
            for L in layers:
                base_vec = bare[c][L]
                if b == "planted":
                    delta = 3.0 * planted_dir  # same direction across contexts
                else:
                    delta = torch.randn(hidden_dim, generator=g)  # independent
                aug[f"{c}__{b}"] = aug.get(f"{c}__{b}", {})
                aug[f"{c}__{b}"][L] = base_vec + delta
    return bare, aug, contexts, behaviors, layers


def test_behavior_shift_metrics_end_to_end_shape_and_planted_direction():
    bare, aug, contexts, behaviors, layers = _toy_vectors(hidden_dim=48, seed=3)
    out = behavior_shift_metrics(
        bare,
        aug,
        context_names=contexts,
        behaviors=behaviors,
        layers=layers,
        null_n_perm=20,
        null_seed=42,
    )
    # Structure.
    assert set(out["cells"].keys()) == set(behaviors)
    for b in behaviors:
        assert set(out["cells"][b].keys()) == {str(L) for L in layers}
        for L in layers:
            cell = out["cells"][b][str(L)]
            assert "relative_magnitude" in cell
            assert "consistency_cosine_raw" in cell
            assert "consistency_cosine_mean_subtracted" in cell
            assert "pc1_variance_share" in cell
            assert "proj_on_known_direction" not in cell  # no known_directions passed
    # Planted behavior: consistency cosine ~1 and PC1 share ~1 (same direction).
    for L in layers:
        planted = out["cells"]["planted"][str(L)]
        assert planted["consistency_cosine_raw"] > 0.99
        assert planted["pc1_variance_share"] > 0.99
        noise = out["cells"]["noise"][str(L)]
        assert noise["consistency_cosine_raw"] < 0.5  # independent dirs -> low
    # Behavior separability + null present per layer.
    for L in layers:
        sep = out["behavior_separability"][str(L)]
        assert sep["names"] == behaviors
        assert np.array(sep["matrix"]).shape == (len(behaviors), len(behaviors))
        assert out["consistency_null"][str(L)]["n_perm"] == 20
    assert out["meta"]["n_context"] == len(contexts)
    assert out["meta"]["hidden_dim"] == 48


def test_behavior_shift_metrics_with_known_direction_projection():
    bare, aug, contexts, behaviors, layers = _toy_vectors(hidden_dim=48, seed=5)
    # Known direction == the planted direction recovered from the planted cell's
    # mean Delta at layer 0; projection of the planted behavior should be ~1.
    deltas0 = torch.stack([aug[f"{c}__planted"][0] - bare[c][0] for c in contexts])
    u0 = deltas0.mean(dim=0)
    known = {("planted", L): u0 for L in layers}
    out = behavior_shift_metrics(
        bare,
        aug,
        context_names=contexts,
        behaviors=behaviors,
        layers=layers,
        known_directions=known,
        null_n_perm=10,
    )
    proj = out["cells"]["planted"]["0"]["proj_on_known_direction"]
    assert proj["mean"] > 0.99  # planted Delta lies along the planted direction
    assert "proj_on_known_direction" not in out["cells"]["noise"]["0"]  # no dir for 'noise'
