"""Synthetic numerical oracles for workspace summaries, fits, and paired inference."""

import numpy as np
import pytest

from explore_persona_space.analysis.workspace_components import (
    aggregate_rollouts,
    component_metrics,
    fit_affine_ridge,
    match_direction_variances,
    paired_context_bootstrap,
    per_direction_metrics,
    per_direction_scores,
    reconstruction_metrics,
    validate_context_splits,
    workspace_gap_contrasts,
)


def test_equal_rollout_weighting_and_noise_with_unequal_lengths():
    """Longer rollouts must not gain weight, and rollout variance stays measurable."""
    arrays = [
        np.array([[0.0, 2.0]]),
        np.array([[6.0, 4.0]] * 3),
        np.array([[9.0, 8.0]]),
    ]
    result = aggregate_rollouts({"y": arrays}, ["a", "a", "b"], [0, 1, 0])
    np.testing.assert_array_equal(result.rollout_token_counts, [1, 3, 1])
    np.testing.assert_array_equal(result.rollout_counts, [2, 1])
    np.testing.assert_allclose(result.context_means["y"], [[3.0, 3.0], [9.0, 8.0]])
    assert not np.allclose(result.context_means["y"][0], np.concatenate(arrays[:2]).mean(0))
    # Deviations ±(3,1) have sample covariance trace (10+10)/(2-1)=20.
    np.testing.assert_allclose(result.noise_covariance_trace["y"][0], 20.0)
    np.testing.assert_allclose(result.mean_noise_covariance_trace["y"][0], 10.0)
    assert np.isnan(result.noise_covariance_trace["y"][1])
    assert result.noise_status == ("ok", "insufficient_rollouts")
    np.testing.assert_array_equal(result.rollout_means["y"], [[0, 2], [6, 4], [9, 8]])


def test_decomposition_happens_before_token_and_rollout_averaging():
    """A row-dependent projection cannot be replaced by projection after averaging."""
    y = np.array([[2.0, 0.0], [0.0, 4.0]])
    directions = np.array([[[1.0], [0.0]], [[0.0], [1.0]]])
    scores = per_direction_scores(y, directions)
    component = (directions * scores[:, None, :]).sum(-1)
    result = aggregate_rollouts({"y": [y], "J": [component], "restJ": [y - component]}, ["ctx"])
    np.testing.assert_allclose(result.context_means["J"], [[1.0, 2.0]])
    assert not np.allclose(result.context_means["J"][0], np.array([y.mean(0)[0], 0.0]))
    audit = reconstruction_metrics(
        result.context_means["y"],
        result.context_means["J"],
        result.context_means["restJ"],
    )
    assert audit["max_abs_reconstruction_error"] == 0.0


def test_aggregation_rejects_unpaired_token_coverage_and_duplicate_rollouts():
    """Missing component tokens and duplicate draw identities are hard errors."""
    with pytest.raises(ValueError, match="token coverage"):
        aggregate_rollouts({"J": [np.ones((2, 3))], "R": [np.ones((1, 3))]}, ["ctx"])
    with pytest.raises(ValueError, match="unique within"):
        aggregate_rollouts({"J": [np.ones((2, 3))] * 2}, ["ctx", "ctx"], [0, 0])
    with pytest.raises(ValueError, match="nonempty"):
        aggregate_rollouts({"J": [np.ones((0, 3))]}, ["ctx"])


def test_context_leakage_is_checked_at_group_identity():
    """Duplicated rollouts inside a split are legal; the same context across splits is not."""
    validate_context_splits({"train": ["a", "a", "b"], "validation": ["c"], "test": ["d"]})
    with pytest.raises(ValueError, match="Context leakage"):
        validate_context_splits({"train": ["a", "b"], "test": ["b", "c"]})
    with pytest.raises(ValueError, match="string or integer"):
        validate_context_splits({"train": [np.nan], "test": ["c"]})


def test_component_own_centered_r2_covariance_and_bias():
    """Pooled R² has a known denominator and decomposes SSE into variance plus bias."""
    y = np.array([[1.0, 0.0], [3.0, 2.0], [5.0, 4.0]])
    p = 0.5 * (y - y.mean(0)) + y.mean(0) + np.array([1.0, -1.0])
    result = component_metrics(y, p)
    assert result["sst"] == 16.0
    assert result["sse"] == 10.0
    assert result["r2"] == pytest.approx(0.375)
    assert result["target_prediction_covariance_trace"] == pytest.approx(8.0 / 3.0)
    assert result["squared_mean_bias"] == 2.0
    assert result["mse_per_context"] == pytest.approx(
        result["residual_variance_trace"] + result["squared_mean_bias"]
    )
    np.testing.assert_allclose(component_metrics(y * 100, p * 100)["r2"], result["r2"])
    tiny = component_metrics(y * 1e-20, p * 1e-20)
    assert tiny["status"] == "ok"
    assert tiny["r2"] == pytest.approx(result["r2"])


def test_zero_variance_never_gets_fake_r2():
    """A perfectly predicted constant has undefined R², not one or zero."""
    result = component_metrics(np.full((5, 2), 7.0), np.full((5, 2), 7.0))
    assert result["r2"] is None
    assert result["status"] == "zero_target_variance"
    assert result["sse"] == 0
    result = component_metrics(np.full((5, 2), 7.0), np.zeros((5, 2)))
    assert result["r2"] is None
    assert result["sse"] > 0


def test_component_covariance_accounting_does_not_assume_orthogonality():
    """Context-dependent/token-averaged components may have nonzero covariance."""
    c = np.array([[0.0, 1.0], [1.0, 3.0], [3.0, 2.0]])
    r = 2 * c + 5
    result = reconstruction_metrics(c + r, c, r)
    assert result["twice_component_rest_covariance_trace"] > 0
    assert result["variance_decomposition_error"] == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError, match="reconstruction failed"):
        reconstruction_metrics(c + r + 1, c, r)


def test_direction_scores_preserve_rows_and_report_zero_variance():
    """Rank-paired normalized direction scores preserve rows without rescaling."""
    j = np.array([[0.0, 2.0], [2.0, 4.0], [4.0, 0.0]])
    np.testing.assert_array_equal(per_direction_scores(j, np.eye(2)), j)
    metrics = per_direction_metrics(np.column_stack([j[:, 0], np.ones(3)]), np.zeros((3, 2)))
    assert metrics["status"] == ("ok", "zero_target_variance")
    assert np.isnan(metrics["r2"][1])
    assert metrics["r2"][0] == component_metrics(j[:, :1], np.zeros((3, 1)))["r2"]
    with pytest.raises(ValueError, match="unit norm"):
        per_direction_scores(j, 2 * np.eye(2))


def test_variance_matching_selects_original_directions_with_explicit_unmatched():
    """Only directions within the calibration caliper match, without reuse or rescaling."""
    base = np.array([-1.0, 0.0, 1.0])[:, None]
    j = base * np.sqrt([1.0, 4.0, 16.0, 0.0])
    r = base * np.sqrt([4.1, 1.05, 100.0, 0.0])
    j_before, r_before = j.copy(), r.copy()
    match = match_direction_variances(j, r, log_variance_caliper=0.2)
    np.testing.assert_array_equal(match.j_indices, [1, 0])
    np.testing.assert_array_equal(match.r_indices, [0, 1])
    np.testing.assert_array_equal(match.unmatched_j_indices, [2, 3])
    np.testing.assert_array_equal(match.unmatched_r_indices, [2, 3])
    np.testing.assert_array_equal(match.zero_variance_j_indices, [3])
    np.testing.assert_array_equal(match.zero_variance_r_indices, [3])
    assert match.status == "partial_match"
    assert np.all(match.log_variance_distance <= 0.2)
    np.testing.assert_array_equal(j, j_before)
    np.testing.assert_array_equal(r, r_before)
    repeated = match_direction_variances(j, r, log_variance_caliper=0.2)
    np.testing.assert_array_equal(match.j_indices, repeated.j_indices)
    np.testing.assert_array_equal(match.r_indices, repeated.r_indices)
    with pytest.raises(ValueError, match="paired"):
        match_direction_variances(j[:2], r)


def test_sparse_variance_matching_agrees_with_dense_greedy_oracle():
    """Adjacent heap matching has the same unique-distance choices as all-pairs search."""
    rng = np.random.default_rng(773)
    base = np.array([-1.0, 0.0, 1.0])[:, None]
    j = base * np.exp(rng.normal(size=31))
    r = base * np.exp(rng.normal(size=24))
    match = match_direction_variances(j, r, log_variance_caliper=0.2)
    distance = np.abs(np.log(np.var(j, axis=0))[:, None] - np.log(np.var(r, axis=0))[None])
    pairs = []
    while np.isfinite(distance).any():
        ji, ri = np.unravel_index(np.argmin(distance), distance.shape)
        if distance[ji, ri] > 0.2:
            break
        pairs.append((ji, ri))
        distance[ji, :] = np.inf
        distance[:, ri] = np.inf
    np.testing.assert_array_equal(match.j_indices, [p[0] for p in pairs])
    np.testing.assert_array_equal(match.r_indices, [p[1] for p in pairs])


@pytest.mark.parametrize("n,d", [(9, 15), (30, 4)])
def test_ridge_shared_factorization_matches_direct_oracle(n, d):
    """Both dual and primal production paths agree with independently solved ridge."""
    rng = np.random.default_rng(61)
    xtr = rng.normal(size=(n, d)) * np.linspace(1, 3, d) + 4
    xtr[:, -1] = 7.0
    xv = rng.normal(size=(11, d)) * np.linspace(1, 3, d) + 8
    xv[:, -1] = 7.0
    ytr = {"J": rng.normal(size=(n, 3)), "restJ": rng.normal(size=(n, 2)) * 10 + 11}
    yv = {"J": rng.normal(size=(11, 3)), "restJ": rng.normal(size=(11, 2)) * 10 + 11}
    alphas = np.array([0.01, 1.0, 100.0])
    results = fit_affine_ridge(
        xtr,
        ytr,
        xv,
        yv,
        alphas=alphas,
        train_context_ids=list(range(n)),
        validation_context_ids=list(range(n, n + 11)),
        alpha_chunk_size=2,
        output_chunk_size=1,
    )
    xmu, xsd = xtr.mean(0), xtr.std(0)
    xsd[xsd == 0] = 1.0
    xn = (xtr - xmu) / xsd
    for name, fit in results.items():
        ym = ytr[name].mean(0)
        weights = [
            np.linalg.solve(xn.T @ xn + a * np.eye(d), xn.T @ (ytr[name] - ym)) for a in alphas
        ]
        preds = [(xv - xmu) / xsd @ w + ym for w in weights]
        sse = np.array([np.square(p - yv[name]).sum() for p in preds])
        selected = np.argmin(sse)
        np.testing.assert_allclose(fit.validation_sse, sse, rtol=1e-10, atol=1e-9)
        assert fit.selected_alpha == alphas[selected]
        np.testing.assert_allclose(fit.predict(xv), preds[selected], rtol=1e-10, atol=1e-9)
        np.testing.assert_allclose(fit.weights, weights[selected] / xsd[:, None], atol=1e-9)
        np.testing.assert_allclose(fit.x_mean, xmu, atol=1e-14)
        np.testing.assert_allclose(fit.x_scale, xsd, atol=1e-14)
        assert fit.factorization == ("dual_gram_eigh" if n <= d else "primal_gram_eigh")


def test_ridge_linear_recovery_geometry_invariance_and_validation_selection():
    """Exact affine targets recover their intercept and selecting alpha uses validation."""
    xtr = np.arange(12.0)[:, None]
    xv = np.arange(12.0, 17.0)[:, None]
    yt, yv = 3 * xtr + 9, 3 * xv + 9
    kwargs = dict(
        alphas=[1e-10, 1e6],
        train_context_ids=list(range(12)),
        validation_context_ids=list(range(12, 17)),
    )
    results = fit_affine_ridge(
        xtr, {"J": yt, "scaled": 10 * yt}, xv, {"J": yv, "scaled": 10 * yv}, **kwargs
    )
    assert results["J"].selected_alpha == 1e-10
    np.testing.assert_allclose(results["J"].weights, [[3]], atol=1e-8)
    np.testing.assert_allclose(results["J"].intercept, [9], atol=1e-8)
    np.testing.assert_allclose(results["scaled"].predict(xv), results["J"].predict(xv) * 10)
    assert results["scaled"].y_scale == pytest.approx(results["J"].y_scale * 10)
    assert component_metrics(yv, results["J"].predict(xv))["r2"] == pytest.approx(1.0)
    # Train data unchanged; validation favors shrinkage toward the train mean.
    alternative = fit_affine_ridge(xtr, {"J": yt}, xv, {"J": np.full_like(yv, yt.mean())}, **kwargs)
    assert alternative["J"].selected_alpha == 1e6
    constant = fit_affine_ridge(
        xtr, {"J": np.ones_like(yt)}, xv, {"J": np.ones_like(yv)}, **kwargs
    )["J"]
    assert constant.target_status == "zero_train_variance"
    np.testing.assert_array_equal(constant.predict(xv), np.ones_like(yv))


def test_ridge_context_overlap_and_repeated_contexts_fail_before_fitting():
    """The production fitting API owns the context-disjoint split boundary."""
    x = np.arange(3.0)[:, None]
    with pytest.raises(ValueError, match="Context leakage"):
        fit_affine_ridge(
            x,
            {"J": x},
            x,
            {"J": x},
            alphas=[1],
            train_context_ids=[0, 1, 2],
            validation_context_ids=[2, 3, 4],
        )
    with pytest.raises(ValueError, match="repeated context"):
        fit_affine_ridge(
            x,
            {"J": x},
            x,
            {"J": x},
            alphas=[1],
            train_context_ids=[0, 0, 1],
            validation_context_ids=[2, 3, 4],
        )


def test_paired_bootstrap_matches_explicit_resampling_and_chunk_invariance():
    """Each statistic uses the SAME contextual counts and re-centers its own target."""
    rng = np.random.default_rng(7)
    targets = {
        name: rng.normal(size=(15, 4)) * scale + 20
        for name, scale in [("J", 1), ("restJ", 2), ("R", 3), ("restR", 4)]
    }
    predictions = {
        "ridge": {name: y + rng.normal(size=y.shape) for name, y in targets.items()},
        "mlp": {name: y + rng.normal(size=y.shape) / 2 for name, y in targets.items()},
    }
    contrasts = workspace_gap_contrasts()
    kwargs = dict(n_bootstrap=31, seed=43, contrasts=contrasts)
    result = paired_context_bootstrap(
        targets,
        predictions,
        list(range(15)),
        chunk_size=3,
        output_chunk_size=1,
        **kwargs,
    )
    other = paired_context_bootstrap(
        targets,
        predictions,
        list(range(15)),
        chunk_size=16,
        output_chunk_size=3,
        **kwargs,
    )
    indices = np.random.default_rng(43).integers(0, 15, size=(31, 15))
    counts = np.zeros((31, 15), dtype=np.int64)
    np.add.at(counts, (np.arange(31)[:, None], indices), 1)
    supplied = paired_context_bootstrap(
        targets, predictions, list(range(15)), counts=counts, **kwargs
    )
    for model, by_target in predictions.items():
        for target, p in by_target.items():
            expected = [component_metrics(targets[target][draw], p[draw])["r2"] for draw in indices]
            np.testing.assert_allclose(result["samples"][f"{model}/{target}"], expected, rtol=1e-11)
    for name, samples in result["samples"].items():
        np.testing.assert_allclose(samples, other["samples"][name], atol=1e-12)
        np.testing.assert_allclose(samples, supplied["samples"][name], atol=1e-12)
    assert result["counts_sha256"] == other["counts_sha256"] == supplied["counts_sha256"]
    samples = result["samples"]
    np.testing.assert_allclose(samples["G_R_minus_G_J"], samples["G_R"] - samples["G_J"])
    np.testing.assert_allclose(samples["MLP_gain/J"], samples["mlp/J"] - samples["ridge/J"])


def test_bootstrap_identical_models_have_exact_zero_paired_gain():
    """Independent model bootstraps would introduce spurious nonzero differences."""
    rng = np.random.default_rng(87)
    y = rng.normal(size=(20, 3))
    p = 0.3 * y
    result = paired_context_bootstrap(
        {"J": y},
        {"ridge": {"J": p}, "mlp": {"J": p}},
        list(range(20)),
        n_bootstrap=70,
        seed=5,
        contrasts={"gain": {"mlp/J": 1, "ridge/J": -1}},
    )
    np.testing.assert_array_equal(result["samples"]["gain"], np.zeros(70))
    assert result["summary"]["gain"]["interval"] == [0.0, 0.0]


def test_bootstrap_degenerate_draws_are_explicit_and_not_dropped():
    """An unavailable component invalidates its contrast instead of becoming zero."""
    target = np.array([[0.0], [2.0]])
    result = paired_context_bootstrap(
        {"J": target, "R": np.ones_like(target)},
        {"ridge": {"J": target, "R": np.ones_like(target)}},
        ["a", "b"],
        n_bootstrap=3,
        seed=4,
        counts=np.array([[2, 0], [1, 1], [0, 2]]),
        contrasts={"gap": {"ridge/J": 1, "ridge/R": -1}},
    )
    assert result["summary"]["ridge/J"]["n_valid_draws"] == 1
    assert result["summary"]["ridge/J"]["interval"] is None
    assert result["summary"]["ridge/R"]["estimate"] is None
    assert result["summary"]["gap"]["estimate"] is None
    assert np.isnan(result["samples"]["gap"]).all()
    with pytest.raises(ValueError, match="unique context"):
        paired_context_bootstrap(
            {"J": target}, {"ridge": {"J": target}}, ["a", "a"], n_bootstrap=3, seed=4
        )


def test_nonrepresentable_constant_has_exactly_zero_variance_everywhere():
    """The decimal 0.1 must not gain variance from floating-point mean summation."""
    constant = np.full((17, 2), 0.1)
    assert component_metrics(constant, constant)["r2"] is None
    assert per_direction_metrics(constant, constant)["status"] == (
        "zero_target_variance",
        "zero_target_variance",
    )
    assert match_direction_variances(constant, constant).status == "no_eligible_pairs"
    result = paired_context_bootstrap(
        {"J": constant},
        {"ridge": {"J": constant}},
        list(range(17)),
        n_bootstrap=5,
        seed=1,
    )
    assert result["summary"]["ridge/J"]["estimate"] is None
    assert result["summary"]["ridge/J"]["n_valid_draws"] == 0
    fitted = fit_affine_ridge(
        constant,
        {"J": constant},
        constant,
        {"J": constant},
        alphas=[1.0],
        train_context_ids=list(range(17)),
        validation_context_ids=list(range(17, 34)),
    )["J"]
    assert fitted.target_status == "zero_train_variance"
    np.testing.assert_array_equal(fitted.x_scale, np.ones(2))
    np.testing.assert_array_equal(fitted.predict(constant), constant)


def test_synthetic_separability_and_swap_null_have_expected_gap_signs():
    """A deterministic synthetic oracle separates easy rest/hard J; swapping labels reverses it."""
    x = np.linspace(-2, 2, 41)[:, None]
    # Unit fixture: predictions are specified analytically, not an experimental fit.
    targets = {"J": x * 0.2, "restJ": x * 4, "R": x, "restR": x * 2}
    predictions = {"ridge": {"J": np.zeros_like(x), "restJ": x * 4, "R": x * 0.5, "restR": x}}
    result = paired_context_bootstrap(
        targets,
        predictions,
        list(range(41)),
        n_bootstrap=80,
        seed=2,
        contrasts=workspace_gap_contrasts(mlp_predictor=None),
    )
    assert result["summary"]["G_J"]["estimate"] == pytest.approx(1.0)
    assert result["summary"]["G_R"]["estimate"] == pytest.approx(0.0)
    assert result["summary"]["G_R_minus_G_J"]["estimate"] == pytest.approx(-1.0)
    assert result["summary"]["G_R_minus_G_J"]["interval"][1] < 0
    swapped = paired_context_bootstrap(
        targets,
        predictions,
        list(range(41)),
        n_bootstrap=80,
        seed=2,
        contrasts=workspace_gap_contrasts(
            j_component="R",
            j_rest="restR",
            r_component="J",
            r_rest="restJ",
            mlp_predictor=None,
        ),
    )
    np.testing.assert_allclose(
        swapped["samples"]["G_R_minus_G_J"], -result["samples"]["G_R_minus_G_J"]
    )
