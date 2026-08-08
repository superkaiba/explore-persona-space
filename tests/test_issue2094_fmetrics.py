"""CPU unit tests for the issue #2094 fraction-of-swap metric library (unit B).

Synthetic tensors only — no model, network, or file dependencies. Covers:
F_act exactness on constructed s/t, the disjoint-halves estimator vs the
naive shared-baseline estimator (demonstrating the #1415 inflation on
correlated noise), F_beh normalization incl. degenerate near-zero
denominators (explicit NaN + flag, never coercion), the transport
orientation bind (both planted orientations, both criteria, the ambiguous
fail-fast), and the homogeneity math on constructed linear/nonlinear
responses.
"""

from __future__ import annotations

import math

import pytest
import torch

from explore_persona_space.experiments.issue2094 import fmetrics as fm

torch.manual_seed(0)


# ── shared numerics ───────────────────────────────────────────────────


def test_safe_cosine_exact_and_zero_norm():
    a = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    b = torch.tensor([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    out = fm.safe_cosine(a, b)
    assert out[0].item() == pytest.approx(0.0, abs=1e-7)
    assert out[1].item() == pytest.approx(1.0, abs=1e-7)
    assert torch.isnan(out[2])  # zero-norm side → NaN, never coerced


def test_signed_projection_exact():
    t = torch.tensor([2.0, 0.0, 0.0])
    s = torch.tensor([1.0, 5.0, -3.0])  # projection onto t: 1.0/2.0 of t
    assert fm.signed_projection(s, t).item() == pytest.approx(0.5, abs=1e-7)
    # zero-norm t → NaN
    assert torch.isnan(fm.signed_projection(s, torch.zeros(3)))


def test_spearman_brown():
    assert fm.spearman_brown(0.5) == pytest.approx(2 * 0.5 / 1.5)
    t = fm.spearman_brown(torch.tensor([0.5, -1.0]))
    assert t[0].item() == pytest.approx(2 * 0.5 / 1.5)
    assert torch.isnan(t[1])


# ── disjoint-half machinery + F_act ───────────────────────────────────


def test_half_split_indices_even_odd():
    even, odd = fm.half_split_indices(10)
    assert even == [0, 2, 4, 6, 8] and odd == [1, 3, 5, 7, 9]
    with pytest.raises(AssertionError):
        fm.half_split_indices(1)


def test_f_act_exact_on_constructed_axis():
    """Noise-free anchors: F is exactly the planted signed projection."""
    d = 4
    t = torch.zeros(d)
    t[0] = 2.0  # ceiling - floor axis
    floor_draws = torch.zeros(10, d)  # all-zero floor, no noise
    ceiling_draws = t.expand(10, d).clone()
    # patched = floor + 0.5 * t + orthogonal component (must not move F)
    v_patched = 0.5 * t + torch.tensor([0.0, 3.0, -1.0, 2.0])
    res = fm.f_act(v_patched, floor_draws, ceiling_draws)
    assert res.f_act.item() == pytest.approx(0.5, abs=1e-7)
    # both half-assignments agree exactly (no noise), shared companion too
    assert torch.allclose(res.f_act_assignments, torch.full((2,), 0.5), atol=1e-7)
    assert res.f_act_shared.item() == pytest.approx(0.5, abs=1e-7)
    assert res.t_norm.item() == pytest.approx(2.0, abs=1e-6)
    assert not bool(res.degenerate)


def test_f_act_negative_and_overshoot_values():
    d = 3
    t = torch.tensor([1.0, 0.0, 0.0])
    floor_draws = torch.zeros(8, d)
    ceiling_draws = t.expand(8, d).clone()
    res_neg = fm.f_act(-0.25 * t, floor_draws, ceiling_draws)
    assert res_neg.f_act.item() == pytest.approx(-0.25, abs=1e-7)
    res_over = fm.f_act(1.5 * t, floor_draws, ceiling_draws)
    assert res_over.f_act.item() == pytest.approx(1.5, abs=1e-7)


def test_f_act_degenerate_axis_is_nan_flagged():
    d = 3
    floor_draws = torch.zeros(6, d)
    ceiling_draws = torch.zeros(6, d)  # ceiling == floor → zero axis
    res = fm.f_act(torch.ones(d), floor_draws, ceiling_draws)
    assert bool(res.degenerate)
    assert torch.isnan(res.f_act)


def test_f_act_batched_matches_percell():
    torch.manual_seed(1)
    b, k, d = 5, 10, 16
    vp = torch.randn(b, d)
    fl = torch.randn(b, k, d)
    ce = torch.randn(b, k, d) + 3.0
    batched = fm.f_act(vp, fl, ce)
    for i in range(b):
        single = fm.f_act(vp[i], fl[i], ce[i])
        assert batched.f_act[i].item() == pytest.approx(single.f_act.item(), abs=1e-6)
        assert batched.f_act_shared[i].item() == pytest.approx(single.f_act_shared.item(), abs=1e-6)


def test_f_act_broadcasts_shared_anchor_draws():
    torch.manual_seed(2)
    b, k, d = 4, 10, 8
    vp = torch.randn(b, d)
    fl = torch.randn(k, d)  # unbatched anchors shared across cells
    ce = torch.randn(k, d) + 2.0
    batched = fm.f_act(vp, fl, ce)
    for i in range(b):
        single = fm.f_act(vp[i], fl, ce)
        assert batched.f_act[i].item() == pytest.approx(single.f_act.item(), abs=1e-6)


def test_disjoint_halves_remove_shared_baseline_inflation():
    """The #1415 fix: on correlated noise (shared floor mean in s AND t) the
    naive shared-baseline estimator is inflated positive even when the true
    projection is ZERO; the disjoint-halves estimator is unbiased.

    Design: true floor = 0, true ceiling = c (modest signal), floor draws are
    iid noise, v_patched is INDEPENDENT noise → true F = 0. Shared estimator
    picks up E[||floor_mean||²] = d*σ²/K in the numerator.
    """
    torch.manual_seed(3)
    trials, k, d = 4000, 10, 128
    sigma = 1.0
    c = torch.zeros(d)
    c[:20] = 1.0  # ||c||² = 20, comparable to d*σ²/K = 12.8
    floor_draws = sigma * torch.randn(trials, k, d)
    ceiling_draws = c + sigma * torch.randn(trials, k, d)
    v_patched = sigma * torch.randn(trials, d)  # independent of anchors → true F = 0
    res = fm.f_act(v_patched, floor_draws, ceiling_draws)
    mean_disjoint = float(res.f_act.mean())
    mean_shared = float(res.f_act_shared.mean())
    # shared-baseline read is inflated well above zero...
    assert mean_shared > 0.15, mean_shared
    # ...the disjoint read is centered at zero...
    assert abs(mean_disjoint) < 0.05, mean_disjoint
    # ...and the inflation gap is large.
    assert mean_shared - mean_disjoint > 0.10, (mean_shared, mean_disjoint)


def test_axis_split_half_reliability_bounds():
    torch.manual_seed(4)
    k, d = 10, 64
    c = torch.zeros(d)
    c[0] = 50.0  # huge signal → reliability ~1
    fl_clean = 0.01 * torch.randn(k, d)
    ce_clean = c + 0.01 * torch.randn(k, d)
    rel_clean = fm.axis_split_half_reliability(fl_clean, ce_clean, n_splits=8, seed=0)
    assert float(rel_clean.mean_cos) > 0.99
    assert float(rel_clean.spearman_brown) > 0.99
    # pure noise, no signal → reliability near 0
    rel_noise = fm.axis_split_half_reliability(
        torch.randn(k, d), torch.randn(k, d), n_splits=16, seed=0
    )
    assert abs(float(rel_noise.mean_cos)) < 0.5


def test_shift_split_half_reliability_bounds():
    torch.manual_seed(5)
    k, d = 10, 64
    vp = torch.zeros(d)
    vp[0] = 50.0
    rel_hi = fm.shift_split_half_reliability(vp, 0.01 * torch.randn(k, d), n_splits=8, seed=0)
    assert float(rel_hi) > 0.99
    rel_lo = fm.shift_split_half_reliability(torch.zeros(d), torch.randn(k, d), n_splits=16, seed=0)
    assert abs(float(rel_lo)) < 0.6


# ── F_beh ─────────────────────────────────────────────────────────────


def test_delta_contrast_exact_and_range_validation():
    jb = torch.tensor([90.0, 10.0])
    ja = torch.tensor([10.0, 90.0])
    out = fm.delta_contrast(jb, ja)
    assert torch.allclose(out, torch.tensor([0.8, -0.8]), atol=1e-7)
    with pytest.raises(AssertionError):
        fm.delta_contrast(torch.tensor([101.0]), torch.tensor([0.0]))
    with pytest.raises(AssertionError):
        fm.delta_contrast(torch.tensor([50.0]), torch.tensor([-1.0]))
    with pytest.raises(AssertionError):
        fm.delta_contrast(torch.tensor([float("nan")]), torch.tensor([0.0]))


def test_f_beh_exact():
    res = fm.f_beh(
        torch.tensor([0.6]),
        torch.tensor([0.1]),
        torch.tensor([0.9]),
    )
    assert res.f_beh[0].item() == pytest.approx((0.6 - 0.1) / (0.9 - 0.1), abs=1e-7)
    assert res.contrast[0].item() == pytest.approx(0.5, abs=1e-7)
    assert res.denominator[0].item() == pytest.approx(0.8, abs=1e-7)
    assert not bool(res.degenerate_denominator[0])
    assert not bool(res.negative_denominator[0])


def test_f_beh_degenerate_denominator_nan_flagged_never_coerced():
    res = fm.f_beh(
        torch.tensor([0.6, 0.6]),
        torch.tensor([0.5, 0.1]),
        torch.tensor([0.5 + 1e-12, 0.9]),  # cell 0: near-zero separation
    )
    assert bool(res.degenerate_denominator[0]) and torch.isnan(res.f_beh[0])
    # unnormalized contrast stays valid for the degenerate cell
    assert res.contrast[0].item() == pytest.approx(0.1, abs=1e-6)
    # healthy sibling cell unaffected
    assert not bool(res.degenerate_denominator[1])
    assert res.f_beh[1].item() == pytest.approx(0.625, abs=1e-6)


def test_f_beh_negative_denominator_flagged_but_computed():
    res = fm.f_beh(torch.tensor([0.2]), torch.tensor([0.5]), torch.tensor([0.1]))
    assert bool(res.negative_denominator[0])
    assert not bool(res.degenerate_denominator[0])
    assert res.f_beh[0].item() == pytest.approx((0.2 - 0.5) / (0.1 - 0.5), abs=1e-6)


# ── transport apply + orientation bind ────────────────────────────────


def _make_bundle(w: torch.Tensor, d_in: int, d_out: int, seed: int = 0) -> dict:
    gen = torch.Generator().manual_seed(seed)
    return {
        "kind": "ridge",
        "xmu": torch.randn(d_in, generator=gen),
        "xsd": torch.rand(d_in, generator=gen) + 0.5,
        "ymu": torch.randn(d_out, generator=gen),
        "W": w,
        "layer": 14,
    }


def test_validate_map_bundle_missing_key_and_bad_xsd():
    bundle = _make_bundle(torch.eye(4), 4, 4)
    fm.validate_map_bundle(bundle)  # passes
    bad = dict(bundle)
    del bad["ymu"]
    with pytest.raises(AssertionError):
        fm.validate_map_bundle(bad)
    bad2 = dict(bundle)
    bad2["xsd"] = torch.zeros(4)
    with pytest.raises(AssertionError):
        fm.validate_map_bundle(bad2)


def test_apply_ridge_map_exact():
    torch.manual_seed(6)
    d = 5
    w = torch.randn(d, d)
    bundle = _make_bundle(w, d, d, seed=1)
    x = torch.randn(3, d)
    z = (x.double() - bundle["xmu"].double()) / bundle["xsd"].double()
    expect = (bundle["ymu"].double() + z @ w.double()).float()
    got = fm.apply_ridge_map(bundle, x, orientation="zW")
    assert torch.allclose(got, expect, atol=1e-5)
    expect_t = (bundle["ymu"].double() + z @ w.double().T).float()
    got_t = fm.apply_ridge_map(bundle, x, orientation="Wz")
    assert torch.allclose(got_t, expect_t, atol=1e-5)


@pytest.mark.parametrize("planted", ["zW", "Wz"])
def test_orientation_bind_probe_residual_picks_planted(planted: str):
    """Bundle stores W in a planted orientation; probe (x, y) pairs generated
    under the TRUE map recover it, for both plantings."""
    torch.manual_seed(7)
    d, n = 6, 8
    w_true = torch.randn(d, d)  # acts as z @ w_true (the "zW" role)
    w_stored = w_true if planted == "zW" else w_true.T
    bundle = _make_bundle(w_stored, d, d, seed=2)
    x = torch.randn(n, d)
    z = (x.double() - bundle["xmu"].double()) / bundle["xsd"].double()
    y = (bundle["ymu"].double() + z @ w_true.double()).float()
    y = y + 1e-4 * torch.randn(n, d)  # tiny noise
    decision = fm.bind_map_orientation(bundle, x, probe_y=y)
    assert decision.orientation == planted
    assert decision.criterion == "probe-residual"
    assert decision.margin > 1.1
    # decision serializes for map_parity.json
    js = decision.as_dict()
    assert js["orientation"] == planted and "stats" in js


def test_orientation_bind_scale_match_picks_planted():
    """No probe_y: the ymu-residual reference scale disambiguates a W whose
    two orientations produce very different output scales."""
    torch.manual_seed(8)
    d, n = 6, 16
    # W with strongly asymmetric row/col scales: z @ W has RMS ~ column-scale.
    col_scale = torch.tensor([100.0, 100.0, 100.0, 0.01, 0.01, 0.01])
    w = torch.randn(d, d) * col_scale  # scales columns
    bundle = _make_bundle(w, d, d, seed=3)
    x = torch.randn(n, d)
    z = (x.double() - bundle["xmu"].double()) / bundle["xsd"].double()
    ref = float((z @ w.double()).norm(dim=-1).pow(2).mean().sqrt())  # lineage y-residual scale
    decision = fm.bind_map_orientation(bundle, x, reference_scale=ref)
    assert decision.orientation == "zW"
    assert decision.criterion == "scale-match"
    # the transposed planting flips the answer
    bundle_t = dict(bundle)
    bundle_t["W"] = w.T
    decision_t = fm.bind_map_orientation(bundle_t, x, reference_scale=ref)
    assert decision_t.orientation == "Wz"


def test_orientation_bind_ambiguous_fails_loud():
    torch.manual_seed(9)
    d = 6
    w = torch.randn(d, d)
    w = (w + w.T) / 2  # symmetric → orientations identical → margin 1.0
    bundle = _make_bundle(w, d, d, seed=4)
    x = torch.randn(4, d)
    z = (x.double() - bundle["xmu"].double()) / bundle["xsd"].double()
    y = (bundle["ymu"].double() + z @ w.double()).float()
    with pytest.raises(ValueError, match="ambiguous map orientation"):
        fm.bind_map_orientation(bundle, x, probe_y=y)


def test_orientation_bind_requires_evidence():
    bundle = _make_bundle(torch.randn(4, 4), 4, 4, seed=5)
    with pytest.raises(ValueError, match="probe_y or reference_scale"):
        fm.bind_map_orientation(bundle, torch.randn(2, 4))


def test_orientation_bind_nonsquare_decided_by_shape():
    torch.manual_seed(10)
    d_in, d_out = 4, 7
    w = torch.randn(d_in, d_out)
    bundle = _make_bundle(w, d_in, d_out, seed=6)
    decision = fm.bind_map_orientation(bundle, torch.randn(3, d_in))
    assert decision.orientation == "zW" and decision.criterion == "shape"


def test_transport_predicted_shift_is_affine_exact():
    """f(v+alphaΔ) - f(v) == alpha * (Δ/xsd) @ W for the affine standardized ridge."""
    torch.manual_seed(11)
    d = 5
    w = torch.randn(d, d)
    bundle = _make_bundle(w, d, d, seed=7)
    v = torch.randn(d)
    delta = torch.randn(d)
    alpha = 2.0
    got = fm.transport_predicted_shift(bundle, v, delta, alpha, orientation="zW")
    expect = ((alpha * delta.double() / bundle["xsd"].double()) @ w.double()).float()
    assert torch.allclose(got, expect, atol=1e-4)
    # transport cosine of a realized shift equal to the prediction is 1
    assert fm.safe_cosine(got, expect).item() == pytest.approx(1.0, abs=1e-6)


# ── homogeneity / linearity ───────────────────────────────────────────


def test_pairwise_shift_cosines_linear_response_all_ones():
    u = torch.tensor([1.0, 2.0, -1.0])
    alphas = torch.tensor([0.5, 1.0, 2.0, 4.0])
    shifts = alphas.unsqueeze(-1) * u  # perfectly homogeneous
    cos = fm.pairwise_shift_cosines(shifts)
    assert torch.allclose(cos, torch.ones(4, 4), atol=1e-6)


def test_pairwise_shift_cosines_rotating_response_below_one():
    shifts = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    cos = fm.pairwise_shift_cosines(shifts)
    assert cos[0, 2].item() == pytest.approx(0.0, abs=1e-6)
    assert cos[0, 1].item() == pytest.approx(1 / math.sqrt(2), abs=1e-6)
    # zero-norm row → NaN row/col
    cos_z = fm.pairwise_shift_cosines(torch.tensor([[1.0, 0.0], [0.0, 0.0]]))
    assert torch.isnan(cos_z[1]).all() and torch.isnan(cos_z[:, 1]).all()
    assert cos_z[0, 0].item() == pytest.approx(1.0, abs=1e-6)


def test_disattenuated_cosines_exact_and_guarded():
    cos = torch.tensor([[1.0, 0.6], [0.6, 1.0]])
    rel = torch.tensor([0.9, 0.4])
    out = fm.disattenuated_cosines(cos, rel)
    assert out[0, 1].item() == pytest.approx(0.6 / math.sqrt(0.9 * 0.4), abs=1e-6)
    assert out[0, 0].item() == pytest.approx(1.0 / 0.9, abs=1e-6)
    out_bad = fm.disattenuated_cosines(cos, torch.tensor([0.9, 0.0]))
    assert torch.isnan(out_bad[0, 1]) and torch.isnan(out_bad[1, 1])
    assert not torch.isnan(out_bad[0, 0])


def test_log_log_magnitude_fit_slopes():
    alphas = torch.tensor([0.5, 1.0, 2.0, 4.0])
    # linear response: ||shift|| = 3*alpha → slope exactly 1, intercept log 3
    norms_lin = 3.0 * alphas
    slope, intercept = fm.log_log_magnitude_fit(alphas, norms_lin)
    assert slope.item() == pytest.approx(1.0, abs=1e-6)
    assert intercept.item() == pytest.approx(math.log(3.0), abs=1e-6)
    # quadratic response: ||shift|| = alpha² → slope exactly 2
    slope2, _ = fm.log_log_magnitude_fit(alphas, alphas**2)
    assert slope2.item() == pytest.approx(2.0, abs=1e-6)
    # batched
    norms_b = torch.stack([norms_lin, alphas**2])
    slopes, _ = fm.log_log_magnitude_fit(alphas, norms_b)
    assert torch.allclose(slopes, torch.tensor([1.0, 2.0]), atol=1e-6)
    # non-positive norms fail loud (log undefined)
    with pytest.raises(AssertionError):
        fm.log_log_magnitude_fit(alphas, torch.tensor([1.0, 0.0, 1.0, 1.0]))


def test_unity_slope_reference_anchored_at_alpha1():
    alphas = torch.tensor([0.5, 1.0, 2.0, 4.0])
    ref = fm.unity_slope_reference(alphas, torch.tensor(3.0))
    assert torch.allclose(ref, torch.tensor([1.5, 3.0, 6.0, 12.0]), atol=1e-6)
    # batched anchor
    ref_b = fm.unity_slope_reference(alphas, torch.tensor([1.0, 2.0]))
    assert ref_b.shape == (2, 4)
    assert ref_b[1, 3].item() == pytest.approx(8.0, abs=1e-6)
