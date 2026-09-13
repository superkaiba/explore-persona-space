"""Rescore supplementary diagnostics on the same completed primary test cohort."""

from __future__ import annotations

import numpy as np

from explore_persona_space.analysis.mapping_baselines import knn_retrieval
from explore_persona_space.analysis.workspace_components import (
    _center,
    component_metrics,
    paired_context_bootstrap,
    per_direction_metrics,
    reconstruction_metrics,
)
from explore_persona_space.analysis.workspace_diagnostics import (
    _direction_bootstrap,
    _direction_intervals,
)


def mapping_references(targets, predictions, context_ids, config):
    """Rescore saved identity+bias and retrieval references on the primary population."""
    if not {"ridge", "mlp", "identity_bias"} <= set(predictions):
        raise ValueError("Mapping references require the saved identity+bias, ridge and MLP fits")
    ks = tuple(config["fit"]["retrieval"]["k"])
    if len(context_ids) < max(ks):
        raise ValueError("Primary cohort is smaller than a registered retrieval pool cutoff")
    settings = config["statistics"]["bootstrap"]
    bootstrap = paired_context_bootstrap(
        targets,
        predictions,
        context_ids,
        n_bootstrap=settings["draws"],
        seed=config["seed"],
        confidence=settings["confidence"],
    )
    metrics = {}
    for predictor, by_target in predictions.items():
        metrics[predictor] = {}
        for name, prediction in by_target.items():
            target = targets[name]
            metrics[predictor][name] = {
                **component_metrics(target, prediction),
                "r2_interval": bootstrap["summary"][f"{predictor}/{name}"],
                "retrieval": {
                    metric: knn_retrieval(prediction, target, ks=ks, metric=metric)
                    for metric in config["fit"]["retrieval"]["metrics"]
                },
            }
    return {
        "context_ids": context_ids,
        "metrics": metrics,
        "counts_sha256": bootstrap["counts_sha256"],
        "identity_bias_recipe": "saved x + training_mean(y - x); no refitting",
        "retrieval_pool": "same completed joint-cohort targets, in the same context order",
        "retrieval_uncertainty": "descriptive point estimates; intervals shown only for R2",
    }, bootstrap["samples"]


def row_indices(original, selected):
    """Resolve explicitly paired contexts without changing the selected population."""
    if (
        len(original) != len(set(original))
        or len(selected) != len(set(selected))
        or len(selected) < 2
    ):
        raise ValueError("Supplementary diagnostics require unique paired context IDs")
    lookup = {context: i for i, context in enumerate(original)}
    if not set(selected) <= set(lookup):
        raise ValueError("Supplementary source lacks a primary scoring context")
    return np.array([lookup[context] for context in selected])


def agreement_statistics(targets, context_ids, norm_floor):
    """Keep raw J/R differences, reconstruction cross terms and near-zero exclusions."""
    a, b = targets["J"], targets["R"]
    if (
        a.shape != b.shape
        or len(a) != len(context_ids)
        or not np.isfinite(norm_floor)
        or norm_floor < 0
    ):
        raise ValueError("Agreement inputs or training-fixed norm floor differ")
    norms = [np.linalg.norm(value, axis=1) for value in (a, b)]
    eligible = (norms[0] > norm_floor) & (norms[1] > norm_floor)
    cosine = np.full(len(a), np.nan)
    cosine[eligible] = (a[eligible] * b[eligible]).sum(1) / (
        norms[0][eligible] * norms[1][eligible]
    )
    differences = np.square(a - b).sum(1)
    return {
        "context_ids": context_ids,
        "training_fixed_norm_floor": norm_floor,
        "component_cosine": cosine,
        "squared_component_difference": differences,
        "mean_component_cosine": float(cosine[eligible].mean()) if eligible.any() else None,
        "mean_squared_component_difference": float(differences.mean()),
        "eligible_cosine_contexts": int(eligible.sum()),
        "excluded_near_zero_contexts": int((~eligible).sum()),
        "reconstruction": {
            arm: reconstruction_metrics(targets["full"], targets[arm], targets[f"rest{arm}"])
            for arm in ("J", "R")
        },
    }


def paired_noise(report, arrays, targets, context_ids):
    """Recompute variance-of-mean fractions using the exact primary scoring rows."""
    index = row_indices(report["context_ids"], context_ids)
    n = len(index)
    components, selected = {}, {}
    for name, target in targets.items():
        if target.ndim != 2 or len(target) != n or not np.isfinite(target).all():
            raise ValueError("Paired noise targets must use the selected context rows")
        noise = np.asarray(arrays[f"noise__{name}"], dtype=np.float64)
        if (
            noise.shape != (len(report["context_ids"]),)
            or not np.isfinite(noise).all()
            or (noise < 0).any()
        ):
            raise ValueError("Invalid per-context rollout noise trace")
        selected[name] = noise[index]
        variance = float(np.square(_center(target)[0]).sum() / (n - 1))
        expected = float(selected[name].mean())
        fraction = expected / variance if variance > 0 else None
        components[name] = {
            "mean_target_noise_trace": expected,
            "observed_between_context_variance_trace_unbiased": variance,
            "noise_fraction": fraction,
            "noise_subtracted_signal_variance_trace": variance - expected,
            "higher_k_trigger": fraction is not None and fraction > report["trigger_threshold"],
        }
    covariance = {}
    for arm in ("J", "R"):
        value = np.asarray(arrays[f"noise_covariance__{arm}"], dtype=np.float64)
        if value.shape != (len(report["context_ids"]),) or not np.isfinite(value).all():
            raise ValueError("Invalid per-context rollout noise covariance")
        value = value[index]
        if not np.allclose(selected["full"], selected[arm] + selected[f"rest{arm}"] + 2 * value):
            raise ValueError("Paired noise covariance identity failed")
        covariance[arm] = float(value.mean())
    return {
        "context_ids": context_ids,
        "seeds": report["seeds"],
        "components": components,
        "trigger_threshold": report["trigger_threshold"],
        "higher_k_trigger": any(value["higher_k_trigger"] for value in components.values()),
        "component_remainder_mean_noise_covariance": covariance,
        "scope": "completed joint primary test cohort; sampling variability is not reasoning",
    }


def paired_readouts(report, arrays, context_ids, config):
    """Keep training-fixed directions/matches, changing only the test cohort."""
    index = row_indices(report["test_context_ids"], context_ids)
    names = ("J", "R", "random", "pca")
    targets = {name: arrays[f"target__{name}"][index] for name in names}
    predictions = {
        predictor: {name: arrays[f"prediction__{predictor}__{name}"][index] for name in names}
        for predictor in report["metrics"]
    }
    metrics = {
        predictor: {
            name: per_direction_metrics(targets[name], value) for name, value in values.items()
        }
        for predictor, values in predictions.items()
    }
    settings = config["statistics"]["bootstrap"]
    samples, bootstrap = _direction_bootstrap(
        targets,
        predictions,
        draws=settings["draws"],
        seed=config["seed"],
        confidence=settings["confidence"],
    )
    paired, matched, gains = {}, {}, {}
    for predictor, values in metrics.items():
        paired[predictor] = values["R"]["r2"] - values["J"]["r2"]
        samples[f"{predictor}/R_minus_J"] = samples[f"{predictor}/R"] - samples[f"{predictor}/J"]
        matched[predictor] = {}
        for key, match in report["matching"].items():
            arm, control = key.split("_vs_")
            a, b = (
                np.asarray(match["j_indices"], dtype=int),
                np.asarray(match["r_indices"], dtype=int),
            )
            matched[predictor][key] = values[arm]["r2"][a] - values[control]["r2"][b]
            samples[f"{predictor}/{key}"] = (
                samples[f"{predictor}/{arm}"][:, a] - samples[f"{predictor}/{control}"][:, b]
            )
        if predictor.startswith("mlp"):
            gains[f"{predictor}_minus_ridge"] = {
                arm: values[arm]["r2"] - metrics["ridge"][arm]["r2"] for arm in ("J", "R")
            }
            for arm in ("J", "R"):
                samples[f"{predictor}_minus_ridge/{arm}"] = (
                    samples[f"{predictor}/{arm}"] - samples[f"ridge/{arm}"]
                )
    bootstrap["intervals"] = _direction_intervals(samples, settings["confidence"])
    return {
        **report,
        "test_context_ids": context_ids,
        "metrics": metrics,
        "paired_context_bootstrap": bootstrap,
        "paired_lens_r2_difference": paired,
        "matched_control_r2_difference": matched,
        "mlp_improvement": gains,
        "rescore_policy": (
            "same completed primary cohort; directions and matching frozen on training/calibration"
        ),
    }, samples
