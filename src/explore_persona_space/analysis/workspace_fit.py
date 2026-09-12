"""Fixed-split component fits using the repository's ridge and batched MLP engines."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from explore_persona_space.analysis.mapping_baselines import identity_bias_predict, knn_retrieval
from explore_persona_space.analysis.workspace_components import (
    component_metrics,
    fit_affine_ridge,
    paired_context_bootstrap,
    reconstruction_metrics,
    validate_context_splits,
    workspace_gap_contrasts,
)
from explore_persona_space.analysis.workspace_runtime import file_sha256, save_json


def finite_json(value):
    """Convert NumPy results to strict JSON, retaining undefined metrics as null."""
    if isinstance(value, Mapping):
        return {str(key): finite_json(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def _check_fixed_recipes(fixed_recipes, targets, settings):
    """Reject missing targets or recipes outside the registered grid."""
    if fixed_recipes is not None:
        if set(fixed_recipes) != set(targets["train"]):
            raise ValueError("Fixed MLP recipes must cover every target exactly")
        for recipe in fixed_recipes.values():
            if (
                recipe["hidden"] not in settings["mlp_hidden"]
                or recipe["lr"] not in settings["mlp_learning_rates"]
            ):
                raise ValueError("Fixed MLP recipe is outside the frozen tuning grid")


def fit_mlp_targets(
    x, targets, config, output: Path, *, device: str, training_logger=None, fixed_recipes=None
) -> dict:
    """Tune on mean validation SSE, or use each target's fixed full-data recipe.

    Each seed remains a separate predictor. The first registered seed is the
    primary MLP; there is no test-selected seed or prediction ensemble. Inputs
    and target scalar scales use train rows only and match the ridge convention.
    """
    from explore_persona_space.analysis.vectorized_mlp_skill import (
        SplitMLPGroup,
        fit_batched_split_mlp,
    )

    settings = config["fit"]
    _check_fixed_recipes(fixed_recipes, targets, settings)
    xmu, xsd = x["train"].mean(0), x["train"].std(0)
    xsd = np.where(xsd > 0, xsd, 1.0)
    xn = {split: ((values - xmu) / xsd).astype(np.float32) for split, values in x.items()}
    scales = {}
    for name, values in targets["train"].items():
        mu = values.mean(0)
        scale = float(np.sqrt(np.square(values - mu).mean()))
        scales[name] = (mu, scale if scale > 0 else 1.0)
    candidates = []
    joined_eval = np.concatenate([xn["validation"], xn["test"]])
    for hidden in settings["mlp_hidden"]:
        for lr in settings["mlp_learning_rates"]:
            names = [
                name
                for name in targets["train"]
                if fixed_recipes is None
                or (fixed_recipes[name]["hidden"], fixed_recipes[name]["lr"]) == (hidden, lr)
            ]
            if not names:
                continue
            seed_results = {}
            for seed in settings["mlp_seeds"]:

                def log_epoch(*, hidden=hidden, lr=lr, seed=seed, **record):
                    """Stream only train/validation losses, never held-out labels."""
                    if training_logger is not None:
                        training_logger(
                            {"hidden": hidden, "learning_rate": lr, "seed": seed, **record}
                        )

                groups = [
                    SplitMLPGroup(
                        key=(name,),
                        X_train=xn["train"],
                        Y_train=((values - scales[name][0]) / scales[name][1]).astype(np.float32),
                        X_val=xn["validation"],
                        Y_val=(
                            (targets["validation"][name] - scales[name][0]) / scales[name][1]
                        ).astype(np.float32),
                        X_eval=joined_eval,
                    )
                    for name, values in targets["train"].items()
                    if name in names
                ]
                fitted = fit_batched_split_mlp(
                    groups,
                    seed=seed,
                    hidden=hidden,
                    lr=lr,
                    wd=settings["mlp_weight_decay"],
                    max_epochs=settings["mlp_max_epochs"],
                    patience=settings["mlp_patience"],
                    device=device,
                    chunk_size=4,
                    loss="mse",
                    standardize_inputs=False,
                    epoch_callback=log_epoch if training_logger is not None else None,
                )
                seed_results[seed] = fitted
            for name in names:
                mu, scale = scales[name]
                errors = []
                for fitted in seed_results.values():
                    val = fitted.preds_by_key[(name,)][: len(xn["validation"])] * scale + mu
                    errors.append(float(np.square(val - targets["validation"][name]).sum()))
                candidates.append(
                    {
                        "target": name,
                        "hidden": hidden,
                        "lr": lr,
                        "validation_sse": float(np.mean(errors)),
                    }
                )
                for seed, fitted in seed_results.items():
                    folder = output / f"h{hidden}-lr{lr}-seed{seed}"
                    folder.mkdir(parents=True, exist_ok=True)
                    prediction = fitted.preds_by_key[(name,)][len(xn["validation"]) :] * scale + mu
                    np.savez(
                        folder / f"{name}.npz",
                        prediction=prediction,
                        **fitted.params_by_key[(name,)],
                        x_train_mean=xmu,
                        x_train_scale=xsd,
                        y_train_mean=mu,
                        y_train_scale=scale,
                        best_epoch=fitted.best_val_epoch_by_key[(name,)],
                    )
            print(f"mlp hidden={hidden} lr={lr} seeds={settings['mlp_seeds']} saved", flush=True)
    chosen = {
        name: min((c for c in candidates if c["target"] == name), key=lambda c: c["validation_sse"])
        for name in targets["train"]
    }
    predictions = {f"mlp_seed{seed}": {} for seed in settings["mlp_seeds"]}
    for name, candidate in chosen.items():
        for seed in settings["mlp_seeds"]:
            folder = output / f"h{candidate['hidden']}-lr{candidate['lr']}-seed{seed}"
            with np.load(folder / f"{name}.npz", allow_pickle=False) as values:
                predictions[f"mlp_seed{seed}"][name] = values["prediction"]
    predictions["mlp"] = predictions[f"mlp_seed{settings['mlp_seeds'][0]}"]
    save_json(
        output / "selection.json",
        {
            "candidates": candidates,
            "selected": chosen,
            "primary_seed": settings["mlp_seeds"][0],
            "selection_reads_test_loss": False,
            "recipe_mode": "fixed_from_full_training_validation" if fixed_recipes else "grid",
        },
    )
    return predictions


def _input_fingerprints(x, targets, ids):
    """Bind all realized rows, exact raw array bytes, shapes and dtypes."""
    import hashlib

    from explore_persona_space.analysis.workspace_runtime import content_sha256

    def array_hash(value):
        a = np.ascontiguousarray(value)
        return {
            "shape": list(a.shape),
            "dtype": str(a.dtype),
            "sha256": hashlib.sha256(a.tobytes()).hexdigest(),
        }

    return {
        split: {
            "ids_sha256": content_sha256(list(ids[split])),
            "x": array_hash(x[split]),
            "targets": {name: array_hash(value) for name, value in targets[split].items()},
        }
        for split in x
    }


def _fit_contract(config):
    """Bind tuning, stopping, seeds and evaluation conventions for full-fit reuse."""
    return {key: config[key] for key in ("fit", "statistics", "seed")}


def evaluate_component_fits(
    x,
    targets,
    ids,
    config,
    output: Path,
    *,
    mlp_device=None,
    training_logger=None,
    fixed_mlp_recipes=None,
) -> dict:
    """Fit original-unit targets, save every example and compute paired intervals."""
    validate_context_splits(ids)
    if set(x) != {"train", "validation", "test"} or set(targets) != set(x):
        raise ValueError("Exactly train/validation/test are required")
    for split in x:
        if set(targets[split]) != {"full", "J", "restJ", "R", "restR"}:
            raise ValueError("Complete full and paired J/R component targets are required")
        for name in ("J", "R"):
            reconstruction_metrics(
                targets[split]["full"], targets[split][name], targets[split][f"rest{name}"]
            )
        if len(x[split]) != len(ids[split]):
            raise ValueError(f"Input rows/IDs differ: {split}")
        if targets[split].keys() != targets["train"].keys():
            raise ValueError("All target names must be present in every split")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed evaluation: {output}")
    ridge = fit_affine_ridge(
        x["train"],
        targets["train"],
        x["validation"],
        targets["validation"],
        alphas=config["fit"]["ridge_alpha_grid"],
        train_context_ids=ids["train"],
        validation_context_ids=ids["validation"],
    )
    predictions = {"ridge": {}, "identity_bias": {}}
    fit_info = {}
    for name, fitted in ridge.items():
        predictions["ridge"][name] = fitted.predict(x["test"])
        predictions["identity_bias"][name] = identity_bias_predict(
            x["train"], targets["train"][name], x["test"]
        )
        np.savez(output / f"ridge-{name}.npz", weights=fitted.weights, bias=fitted.intercept)
        fit_info[name] = {
            "alpha": fitted.selected_alpha,
            "validation_sse": fitted.validation_sse,
            "status": fitted.target_status,
        }
    if mlp_device is not None:
        predictions.update(
            fit_mlp_targets(
                x,
                targets,
                config,
                output / "mlp",
                device=mlp_device,
                training_logger=training_logger,
                fixed_recipes=fixed_mlp_recipes,
            )
        )
    arrays = {f"target__{key}": value for key, value in targets["test"].items()}
    arrays["context_ids"] = np.asarray(ids["test"])
    arrays["x"] = x["test"]
    metrics = {}
    if config["statistics"]["near_zero_component_norm_floor"] != (
        "1e-6_times_training_median_full_target_norm"
    ):
        raise ValueError("Unsupported near-zero component convention")
    norm_floor = 1e-6 * np.median(np.linalg.norm(targets["train"]["full"], axis=1))
    norm_diagnostics = {
        name: {
            "norm_floor": float(norm_floor),
            "near_zero_contexts": int((np.linalg.norm(value, axis=1) <= norm_floor).sum()),
            "near_zero_variation": bool(
                np.linalg.norm(value - value.mean(0), axis=1).mean() <= norm_floor
            ),
        }
        for name, value in targets["test"].items()
    }
    for predictor, values in predictions.items():
        metrics[predictor] = {}
        for name, prediction in values.items():
            arrays[f"prediction__{predictor}__{name}"] = prediction
            cell = component_metrics(targets["test"][name], prediction)
            cell["near_zero_diagnostics"] = norm_diagnostics[name]
            cell["retrieval"] = {
                metric: knn_retrieval(
                    prediction,
                    targets["test"][name],
                    ks=tuple(config["fit"]["retrieval"]["k"]),
                    metric=metric,
                )
                for metric in config["fit"]["retrieval"]["metrics"]
            }
            metrics[predictor][name] = cell
    np.savez(output / "per_example.npz", **arrays)
    boot = config["statistics"]["bootstrap"]
    contrasts = workspace_gap_contrasts(mlp_predictor="mlp" if mlp_device else None)
    for seed in config["fit"]["mlp_seeds"] if mlp_device else []:
        for name, coefficients in workspace_gap_contrasts(
            predictor=f"mlp_seed{seed}", mlp_predictor=None
        ).items():
            contrasts[f"mlp_seed{seed}/{name}"] = coefficients
    intervals = paired_context_bootstrap(
        targets["test"],
        predictions,
        ids["test"],
        n_bootstrap=boot["draws"],
        seed=config["seed"],
        confidence=boot["confidence"],
        contrasts=contrasts,
    )
    np.savez(output / "bootstrap_samples.npz", **intervals.pop("samples"))
    result = {
        "schema": "workspace-jr-component-fit-v1",
        "input_fingerprints": _input_fingerprints(x, targets, ids),
        "fit_contract": _fit_contract(config),
        "mlp_selection_sha256": file_sha256(output / "mlp/selection.json")
        if mlp_device is not None
        else None,
        "metrics": metrics,
        "ridge_selection": fit_info,
        "paired_bootstrap": intervals,
        "split_counts": {s: len(rows) for s, rows in ids.items()},
        "smallest_practical_gap": config["statistics"]["smallest_practical_gap"],
        "reconstruction": {
            name: reconstruction_metrics(
                targets["test"]["full"], targets["test"][name], targets["test"][f"rest{name}"]
            )
            for name in ("J", "R")
        },
    }
    a, b = targets["test"]["J"], targets["test"]["R"]
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    valid = (na > norm_floor) & (nb > norm_floor)
    result["decomposition_agreement"] = {
        "mean_squared_component_difference": float(np.square(a - b).sum(1).mean()),
        "eligible_cosine_contexts": int(valid.sum()),
        "excluded_near_zero_contexts": int((~valid).sum()),
        "mean_component_cosine": float(
            ((a[valid] * b[valid]).sum(1) / (na[valid] * nb[valid])).mean()
        )
        if valid.any()
        else None,
        "training_fixed_norm_floor": float(norm_floor),
    }
    save_json(output / "results.json", finite_json(result))
    return result


def evaluate_learning_curves(
    x, targets, ids, config, output: Path, full_fit: Path, *, device: str, training_logger=None
) -> dict:
    """Refit frozen training prefixes; keep full-data MLP recipes and shared test rows.

    Ridge alpha retains its registered validation selection. MLP hidden width
    and learning rate come from the full training fit's validation-only choice;
    only early stopping reads validation at each smaller training size.
    """
    import json

    from explore_persona_space.analysis.workspace_runtime import file_sha256

    validate_context_splits(ids)
    fractions = config["fit"]["learning_curve_train_fractions"]
    if fractions != sorted(set(fractions)) or fractions[-1] != 1.0 or fractions[0] <= 0:
        raise ValueError("Learning fractions must be increasing, unique, and end at 1")
    n_train = len(ids["train"])
    counts = [int(n_train * fraction) for fraction in fractions]
    if min(counts) < 2 or len(set(counts)) != len(counts):
        raise ValueError("Training prefixes need at least two distinct context counts")
    selection_path = full_fit / "mlp/selection.json"
    full_results_path = full_fit / "results.json"
    recipes = json.loads(selection_path.read_text())["selected"]
    full_results = json.loads(full_results_path.read_text())
    if "input_fingerprints" not in full_results or "fit_contract" not in full_results:
        raise ValueError(
            "Legacy full fit needs a separately attested rerun; never retrofit provenance"
        )
    if full_results["fit_contract"] != _fit_contract(config):
        raise ValueError("Full fit and learning curves must share the exact fit contract")
    if full_results["mlp_selection_sha256"] != file_sha256(selection_path):
        raise ValueError("Full fit MLP recipe selection bytes changed")
    if full_results["input_fingerprints"] != _input_fingerprints(x, targets, ids):
        raise ValueError("Full fit and learning curves must use identical input bytes and IDs")
    with np.load(full_fit / "per_example.npz", allow_pickle=False) as saved:
        if saved["context_ids"].tolist() != ids["test"] or not np.array_equal(
            saved["x"], x["test"]
        ):
            raise ValueError("Full fit and learning curves must use identical test rows")
        for name, value in targets["test"].items():
            if not np.array_equal(saved[f"target__{name}"], value):
                raise ValueError("Full fit and learning curve test targets differ")
    if full_results["split_counts"] != {key: len(value) for key, value in ids.items()}:
        raise ValueError("Full fit and learning curve split counts differ")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "workspace-jr-learning-curves-v1",
        "training_selection": "prefix_of_frozen_hash_order_after_explicit_exclusions",
        "full_fit_result_sha256": file_sha256(full_results_path),
        "full_fit_recipe_sha256": file_sha256(selection_path),
        "mlp_recipe_source": "full_training_validation_selection_no_new_grid_at_smaller_n",
        "shared_validation_ids": ids["validation"],
        "shared_test_ids": ids["test"],
        "cells": [],
    }
    for fraction, count in zip(fractions, counts, strict=True):
        cell = {"fraction": fraction, "n_train": count, "train_ids": ids["train"][:count]}
        if fraction == 1.0:
            result = full_results
            cell["fit_path"] = str(full_fit.resolve())
            cell["reused_full_fit"] = True
        else:
            folder = output / f"train-{count}"

            def log_epoch(record, fraction=fraction, count=count):
                if training_logger is not None:
                    training_logger({"train_fraction": fraction, "train_contexts": count, **record})

            result = evaluate_component_fits(
                {key: value[:count] if key == "train" else value for key, value in x.items()},
                {
                    key: {name: v[:count] for name, v in values.items()}
                    if key == "train"
                    else values
                    for key, values in targets.items()
                },
                {key: value[:count] if key == "train" else value for key, value in ids.items()},
                config,
                folder,
                mlp_device=device,
                fixed_mlp_recipes=recipes,
                training_logger=log_epoch if training_logger is not None else None,
            )
            cell["fit_path"] = str(folder.resolve())
            cell["reused_full_fit"] = False
        cell["metrics"] = result["metrics"]
        cell["paired_bootstrap"] = result["paired_bootstrap"]
        report["cells"].append(cell)
        save_json(output / "learning_curves.json", finite_json(report))
        print(f"learning curve saved train={count}/{n_train}", flush=True)
    return report
