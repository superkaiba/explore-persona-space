"""Training-selected readout controls and finite-rollout noise diagnostics."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from explore_persona_space.analysis.workspace_components import (
    _center,
    match_direction_variances,
    per_direction_metrics,
    per_direction_scores,
    validate_context_splits,
)


def direction_redundancy(directions):
    """Describe correlated unit directions without treating them as independent data."""
    gram = np.asarray(directions, dtype=np.float64).T @ directions
    off_diagonal = gram[np.triu_indices(len(gram), 1)]
    return {
        "directions": len(gram),
        "pairwise_cosine_quantiles": np.quantile(off_diagonal, [0, 0.25, 0.5, 0.75, 1]).tolist()
        if len(off_diagonal)
        else [],
        "pairwise_absolute_cosine_quantiles": np.quantile(
            np.abs(off_diagonal), [0, 0.25, 0.5, 0.75, 1]
        ).tolist()
        if len(off_diagonal)
        else [],
        "effective_rank_participation_ratio": float(np.trace(gram) ** 2 / np.square(gram).sum()),
        "effective_rank_definition": "trace_gram_squared_divided_by_trace_gram_squared_matrix",
    }


def _direction_bootstrap(targets, predictions, *, draws, seed, confidence):
    """Resample context rows, re-centering every scalar direction in every draw."""
    import hashlib

    n = len(targets["J"])
    rng = np.random.default_rng(seed)
    centered = {key: _center(value)[0] for key, value in targets.items()}
    samples = {
        f"{predictor}/{arm}": np.empty((draws, value.shape[1]))
        for predictor in predictions
        for arm, value in targets.items()
    }
    digest = hashlib.sha256()
    for start in range(0, draws, 64):
        count = min(64, draws - start)
        indices = rng.integers(0, n, size=(count, n))
        weights = np.zeros((count, n), dtype=np.int64)
        np.add.at(weights, (np.arange(count)[:, None], indices), 1)
        digest.update(np.asarray(weights, dtype="<i8", order="C").tobytes())
        for arm, y in targets.items():
            first = weights @ np.square(centered[arm])
            second = np.square(weights @ centered[arm]) / n
            denominator = first - second
            tolerance = 32 * np.finfo(np.float64).eps * np.maximum(first, second)
            if np.any(denominator < -tolerance):
                raise FloatingPointError("Direction bootstrap variance is materially negative")
            for predictor, by_arm in predictions.items():
                error = weights @ np.square(y - by_arm[arm])
                ratio = np.full_like(error, np.nan)
                np.divide(error, denominator, out=ratio, where=denominator > tolerance)
                samples[f"{predictor}/{arm}"][start : start + count] = 1 - ratio
    return samples, {
        "counts_sha256": digest.hexdigest(),
        "unit": "context",
        "draws": draws,
        "confidence": confidence,
    }


def _direction_intervals(samples, confidence):
    """Keep undefined scalar-direction resamples visible, including contrasts."""
    tail = (1 - confidence) / 2
    intervals = {}
    for key, values in samples.items():
        valid = np.isfinite(values).all(0)
        limits = np.full((2, values.shape[1]), np.nan)
        limits[:, valid] = np.quantile(values[:, valid], [tail, 1 - tail], axis=0)
        intervals[key] = {
            "low": limits[0],
            "high": limits[1],
            "undefined_bootstrap_draws": (~np.isfinite(values)).sum(0),
        }
    return intervals


def evaluate_direction_readouts(
    y_train, y_test, predictions, directions, token_ids, train_ids, test_ids, config
):
    """Score paired J/R tokens and training-variance-matched random/PCA controls.

    Predictions are already fitted full-answer vectors. No direction, control,
    or matching decision uses held-out target values or prediction errors.
    """
    validate_context_splits({"train": train_ids, "test": test_ids})
    if any(len(ids) < 2 or len(ids) != len(set(ids)) for ids in (train_ids, test_ids)):
        raise ValueError("Readouts require at least two unique context IDs per split")
    if set(directions) != {"J", "R"} or not predictions:
        raise ValueError("Paired J/R directions and at least one predictor are required")
    y_train, y_test = np.asarray(y_train, dtype=np.float64), np.asarray(y_test, dtype=np.float64)
    if len(y_train) != len(train_ids) or len(y_test) != len(test_ids):
        raise ValueError("Readout target rows and context IDs differ")
    if len(token_ids) != len(set(token_ids)) or any(
        direction.shape[1] != len(token_ids) for direction in directions.values()
    ):
        raise ValueError("J/R columns must have the same unique token IDs")
    settings = config["diagnostics"]
    train_scores = {key: per_direction_scores(y_train, basis) for key, basis in directions.items()}
    centered = _center(y_train)[0]
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    tolerance = np.finfo(np.float64).eps * max(centered.shape) * singular[0]
    rank = int((singular > tolerance).sum())
    if rank == 0:
        raise ValueError("Training target has no nonzero PCA control directions")
    rng = np.random.default_rng(config["seed"])
    random = rng.normal(
        size=(y_train.shape[1], len(token_ids) * settings["controls_pool_multiplier"])
    )
    random /= np.linalg.norm(random, axis=0)
    bases = {**directions, "random": random, "pca": vh[:rank].T}
    train_scores.update(
        {key: per_direction_scores(y_train, bases[key]) for key in ("random", "pca")}
    )
    matching = {
        f"{arm}_vs_{control}": asdict(
            match_direction_variances(
                train_scores[arm],
                train_scores[control],
                log_variance_caliper=settings["maximum_absolute_log_variance_mismatch"],
            )
        )
        for arm in ("J", "R")
        for control in ("random", "pca")
    }
    test_scores = {key: per_direction_scores(y_test, basis) for key, basis in bases.items()}
    projected = {
        predictor: {key: per_direction_scores(value, basis) for key, basis in bases.items()}
        for predictor, value in predictions.items()
    }
    metrics = {
        predictor: {
            key: per_direction_metrics(test_scores[key], value) for key, value in by_arm.items()
        }
        for predictor, by_arm in projected.items()
    }
    report = {
        "schema": "workspace-jr-direction-readouts-v1",
        "token_ids": list(token_ids),
        "training_context_ids": list(train_ids),
        "test_context_ids": list(test_ids),
        "label": "J-aligned and R-aligned full-predictor readouts",
        "interpretation": (
            "Random/PCA controls are not established non-workspace directions; "
            "scores do not decompose the predictor."
        ),
        "training_variance": {
            key: np.square(_center(value)[0]).mean(0) for key, value in train_scores.items()
        },
        "matching": matching,
        "metrics": metrics,
        "redundancy": {key: direction_redundancy(value) for key, value in directions.items()},
        "paired_token_cosine": (directions["J"] * directions["R"]).sum(0),
        "pca_nonzero_rank": rank,
        "pca_numerical_rank_tolerance": float(tolerance),
        "paired_lens_r2_difference": {
            predictor: by_arm["R"]["r2"] - by_arm["J"]["r2"]
            for predictor, by_arm in metrics.items()
        },
        "matched_control_r2_difference": {
            predictor: {
                key: by_arm[key.split("_vs_")[0]]["r2"][match["j_indices"]]
                - by_arm[key.split("_vs_")[1]]["r2"][match["r_indices"]]
                for key, match in matching.items()
            }
            for predictor, by_arm in metrics.items()
        },
    }
    bootstrap = config["statistics"]["bootstrap"]
    samples, report["paired_context_bootstrap"] = _direction_bootstrap(
        test_scores,
        projected,
        draws=bootstrap["draws"],
        seed=config["seed"],
        confidence=bootstrap["confidence"],
    )
    for predictor in predictions:
        samples[f"{predictor}/R_minus_J"] = samples[f"{predictor}/R"] - samples[f"{predictor}/J"]
        for key, match in matching.items():
            arm, control = key.split("_vs_")
            samples[f"{predictor}/{key}"] = (
                samples[f"{predictor}/{arm}"][:, match["j_indices"]]
                - samples[f"{predictor}/{control}"][:, match["r_indices"]]
            )
    ridge_names = [name for name in predictions if "ridge" in name]
    mlp_names = [name for name in predictions if "mlp" in name]
    report["mlp_improvement"] = {}
    for mlp in mlp_names:
        for ridge in ridge_names:
            key = f"{mlp}_minus_{ridge}"
            report["mlp_improvement"][key] = {
                arm: metrics[mlp][arm]["r2"] - metrics[ridge][arm]["r2"] for arm in ("J", "R")
            }
            for arm in ("J", "R"):
                samples[f"{key}/{arm}"] = samples[f"{mlp}/{arm}"] - samples[f"{ridge}/{arm}"]
    report["paired_context_bootstrap"]["intervals"] = _direction_intervals(
        samples, bootstrap["confidence"]
    )
    arrays = {f"direction__{key}": value for key, value in bases.items()}
    arrays.update({f"target__{key}": value for key, value in test_scores.items()})
    arrays.update(
        {
            f"prediction__{p}__{key}": value
            for p, by_arm in projected.items()
            for key, value in by_arm.items()
        }
    )
    return report, arrays, samples


def rollout_noise_report(rollouts, context_ids, seeds, *, threshold=0.1):
    """Estimate variance of equal-K context means and paired component noise covariance."""
    names = ("full", "J", "restJ", "R", "restR")
    if set(rollouts) != set(names) or len(seeds) < 2 or len(seeds) != len(set(seeds)):
        raise ValueError("Five paired targets and at least two unique rollout seeds are required")
    values = {name: np.asarray(rollouts[name], dtype=np.float64) for name in names}
    shape = values["full"].shape
    if len(shape) != 3 or shape[:2] != (len(context_ids), len(seeds)) or len(context_ids) < 2:
        raise ValueError("Rollout arrays must be (unique contexts, exact K seeds, features)")
    if len(context_ids) != len(set(context_ids)) or any(
        value.shape != shape or not np.isfinite(value).all() for value in values.values()
    ):
        raise ValueError("Noise arrays must share finite paired values and unique contexts")
    for arm in ("J", "R"):
        if not np.allclose(
            values["full"], values[arm] + values[f"rest{arm}"], rtol=1e-10, atol=1e-10
        ):
            raise ValueError("Rollout component reconstruction mismatch")
    shifted = {name: value - value[:, :1] for name, value in values.items()}
    shifts = {name: value.mean(1) for name, value in shifted.items()}
    means = {name: values[name][:, 0] + shift for name, shift in shifts.items()}
    residuals = {name: value - shifts[name][:, None, :] for name, value in shifted.items()}
    n, k, _ = shape
    noise = {
        name: np.square(value).sum((1, 2)) / (k * (k - 1)) for name, value in residuals.items()
    }
    cells = {}
    for name, mean in means.items():
        observed = float(np.square(_center(mean)[0]).sum() / (n - 1))
        expected_noise = float(noise[name].mean())
        cells[name] = {
            "mean_target_noise_trace": expected_noise,
            "observed_between_context_variance_trace_unbiased": observed,
            "noise_fraction": expected_noise / observed if observed > 0 else None,
            "noise_subtracted_signal_variance_trace": observed - expected_noise,
            "status": "ok" if observed > 0 else "zero_observed_variance",
            "higher_k_trigger": bool(observed > 0 and expected_noise / observed > threshold),
        }
    covariance = {
        arm: (residuals[arm] * residuals[f"rest{arm}"]).sum((1, 2)) / (k * (k - 1))
        for arm in ("J", "R")
    }
    for arm in ("J", "R"):
        if not np.allclose(noise["full"], noise[arm] + noise[f"rest{arm}"] + 2 * covariance[arm]):
            raise ValueError("Noise covariance identity failed")
    return {
        "schema": "workspace-jr-rollout-noise-v1",
        "context_ids": list(context_ids),
        "seeds": list(seeds),
        "components": cells,
        "trigger_threshold": threshold,
        "higher_k_trigger": any(value["higher_k_trigger"] for value in cells.values()),
        "component_remainder_mean_noise_covariance": {
            key: float(value.mean()) for key, value in covariance.items()
        },
        "interpretation": (
            "Finite-rollout noise is sampling variability, not evidence of reasoning. "
            "Noise-subtracted variance is descriptive and may be negative."
        ),
    }, {
        **{f"noise__{key}": value for key, value in noise.items()},
        **{f"noise_covariance__{key}": value for key, value in covariance.items()},
    }
