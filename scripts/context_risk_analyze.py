#!/usr/bin/env python3
"""Analyze prospective agent-risk pilots without bypassing frozen prevalence gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np


SEED = 38296
C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
BOOTSTRAP_REPLICATES = 5000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
        temporary = Path(fh.name)
    os.replace(temporary, path)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as fh:
        fh.write(text)
        temporary = Path(fh.name)
    os.replace(temporary, path)


def load_impossible_activations(
    capture_root: Path,
    *,
    selected_layer: int,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    activations: dict[str, np.ndarray] = {}
    metadata: dict[str, dict[str, Any]] = {}
    done_paths = sorted(capture_root.glob("chunk_*.done.json"))
    if not done_paths:
        raise FileNotFoundError(f"no completed activation chunks under {capture_root}")
    for done_path in done_paths:
        done = json.loads(done_path.read_text(encoding="utf-8"))
        stem = done_path.name.removesuffix(".done.json")
        npz_path = done_path.with_name(f"{stem}.npz")
        rows_path = done_path.with_name(f"{stem}.rows.jsonl")
        if _sha256(npz_path) != done["npz_sha256"]:
            raise RuntimeError(f"activation hash mismatch: {npz_path}")
        if _sha256(rows_path) != done["rows_sha256"]:
            raise RuntimeError(f"activation-row hash mismatch: {rows_path}")
        rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
        with np.load(npz_path) as arrays:
            values = np.asarray(arrays["activation"], dtype=np.float32)
            layers = np.asarray(arrays["layers"], dtype=np.int64)
        layer_indices = np.flatnonzero(layers == selected_layer)
        if len(layer_indices) != 1:
            raise RuntimeError(
                f"selected layer {selected_layer} missing or duplicated in {npz_path}"
            )
        if len(rows) != len(values) or len(rows) != int(done["n_contexts"]):
            raise RuntimeError(f"activation row-count mismatch: {done_path}")
        layer_index = int(layer_indices[0])
        for row, activation in zip(rows, values, strict=True):
            exact_hash = str(row["exact_context_sha256"])
            if exact_hash in activations:
                raise RuntimeError(f"duplicate captured exact context: {exact_hash}")
            vector = np.asarray(activation[layer_index], dtype=np.float32)
            if vector.ndim != 1 or not np.isfinite(vector).all():
                raise RuntimeError(f"invalid activation for {exact_hash}")
            activations[exact_hash] = vector
            metadata[exact_hash] = row
    return activations, metadata


def load_misalignment_activations(
    rollout_root: Path,
    *,
    selected_layer: int,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    """Load one verified pre-action activation per unique exact prefix.

    The public manifest contains a few condition aliases with byte-identical
    visible prefixes. Those aliases must share one predictor row rather than
    leaking the same activation across train and test folds.
    """

    activations: dict[str, np.ndarray] = {}
    metadata: dict[str, dict[str, Any]] = {}
    done_paths = sorted(rollout_root.glob("context_*/done.json"))
    if not done_paths:
        raise FileNotFoundError(f"no completed misalignment contexts under {rollout_root}")
    for done_path in done_paths:
        done = json.loads(done_path.read_text(encoding="utf-8"))
        activation_path = done_path.parent / "pre_action_activation.npz"
        if _sha256(activation_path) != done["activation_sha256"]:
            raise RuntimeError(f"activation hash mismatch: {activation_path}")
        with np.load(activation_path) as arrays:
            values = np.asarray(arrays["activation"], dtype=np.float32)
            layers = np.asarray(arrays["layers"], dtype=np.int64)
        layer_indices = np.flatnonzero(layers == selected_layer)
        if len(layer_indices) != 1:
            raise RuntimeError(
                f"selected layer {selected_layer} missing or duplicated in {activation_path}"
            )
        vector = np.asarray(values[int(layer_indices[0])], dtype=np.float32)
        exact_hash = str(done["exact_context_sha256"])
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise RuntimeError(f"invalid activation for {exact_hash}")
        if exact_hash in activations:
            if not np.array_equal(activations[exact_hash], vector):
                raise RuntimeError(f"duplicate exact context has activation drift: {exact_hash}")
            continue
        activations[exact_hash] = vector
        metadata[exact_hash] = done
    return activations, metadata


def apply_context_map(raw: np.ndarray, map_path: Path) -> np.ndarray:
    with np.load(map_path) as arrays:
        weight = np.asarray(arrays["weight"], dtype=np.float32)
        x_mean = np.asarray(arrays["x_mean"], dtype=np.float32)
        x_scale = np.asarray(arrays["x_scale"], dtype=np.float32)
        y_mean = np.asarray(arrays["y_mean"], dtype=np.float32)
    if weight.shape != (raw.shape[1], raw.shape[1]):
        raise RuntimeError(f"map shape {weight.shape} is incompatible with {raw.shape}")
    if x_mean.shape != x_scale.shape or x_mean.shape != y_mean.shape:
        raise RuntimeError("map normalization vector shapes disagree")
    return ((raw - x_mean) / np.maximum(x_scale, 1e-9)) @ weight + y_mean


def binomial_log_loss(rows: list[dict[str, Any]], probability: np.ndarray) -> float:
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1 - 1e-6)
    positive = np.asarray([row["positive"] for row in rows], dtype=np.float64)
    negative = np.asarray([row["negative"] for row in rows], dtype=np.float64)
    denominator = float((positive + negative).sum())
    if denominator <= 0:
        return float("nan")
    return float(
        -(positive * np.log(probability) + negative * np.log1p(-probability)).sum() / denominator
    )


def binomial_brier(rows: list[dict[str, Any]], probability: np.ndarray) -> float:
    probability = np.asarray(probability, dtype=np.float64)
    positive = np.asarray([row["positive"] for row in rows], dtype=np.float64)
    negative = np.asarray([row["negative"] for row in rows], dtype=np.float64)
    denominator = float((positive + negative).sum())
    if denominator <= 0:
        return float("nan")
    return float(
        (positive * (1 - probability) ** 2 + negative * probability**2).sum() / denominator
    )


def _expand_binomial(
    features: np.ndarray,
    rows: list[dict[str, Any]],
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expanded_x = []
    expanded_y = []
    expanded_groups = []
    for index in indices:
        row = rows[int(index)]
        positive = int(row["positive"])
        negative = int(row["negative"])
        if positive + negative <= 0:
            continue
        expanded_x.extend([features[int(index)]] * (positive + negative))
        expanded_y.extend([1] * positive + [0] * negative)
        expanded_groups.extend([row["task_id"]] * (positive + negative))
    return (
        np.asarray(expanded_x, dtype=np.float32),
        np.asarray(expanded_y, dtype=np.int8),
        np.asarray(expanded_groups),
    )


def _fit_predict_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    *,
    c_value: float,
) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.exceptions import ConvergenceWarning

    if len(np.unique(y_train)) < 2:
        probability = (float(y_train.sum()) + 0.5) / (len(y_train) + 1.0)
        return np.full(len(x_eval), probability, dtype=np.float64)
    # Binomial expansion repeats the identical feature/label pair contiguously.
    # Collapse these runs with frequency weights: this preserves both the
    # standardization and the summed logistic objective, without the huge dual
    # optimization over repeated observations.
    starts = np.r_[
        0,
        1
        + np.flatnonzero(
            (y_train[1:] != y_train[:-1]) | np.any(x_train[1:] != x_train[:-1], axis=1)
        ),
    ]
    weights = np.diff(np.r_[starts, len(y_train)])
    unique_x = x_train[starts]
    unique_y = y_train[starts]
    scaler = StandardScaler().fit(unique_x, sample_weight=weights)
    train = scaler.transform(unique_x)
    evaluation = scaler.transform(x_eval)
    model = LogisticRegression(
        C=float(c_value),
        solver="liblinear",
        dual=False,
        max_iter=5000,
        tol=1e-8,
        random_state=SEED,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(train, unique_y, sample_weight=weights)
    return model.predict_proba(evaluation)[:, 1]


def _select_c(
    features: np.ndarray,
    rows: list[dict[str, Any]],
    train_indices: np.ndarray,
) -> float:
    from sklearn.model_selection import GroupKFold

    context_groups = np.asarray([rows[int(index)]["task_id"] for index in train_indices])
    unique_groups = np.unique(context_groups)
    if len(unique_groups) < 3:
        return 0.01
    splitter = GroupKFold(n_splits=min(5, len(unique_groups)))
    losses = {c_value: [] for c_value in C_GRID}
    context_x = features[train_indices]
    for inner_train, inner_validation in splitter.split(context_x, groups=context_groups):
        fit_indices = train_indices[inner_train]
        validation_indices = train_indices[inner_validation]
        x_fit, y_fit, _groups = _expand_binomial(features, rows, fit_indices)
        if not len(y_fit):
            continue
        for c_value in C_GRID:
            probability = _fit_predict_logistic(
                x_fit,
                y_fit,
                features[validation_indices],
                c_value=c_value,
            )
            validation_rows = [rows[int(index)] for index in validation_indices]
            losses[c_value].append(binomial_log_loss(validation_rows, probability))
    return min(
        C_GRID,
        key=lambda value: (
            float(np.mean(losses[value])) if losses[value] else float("inf"),
            value,
        ),
    )


def leave_one_task_out_predictions(
    features: np.ndarray,
    rows: list[dict[str, Any]],
    *,
    checkpoint_dir: Path | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or len(features) != len(rows):
        raise ValueError("features must be a row-aligned two-dimensional array")
    tasks = sorted({str(row["task_id"]) for row in rows})
    predictions = np.full(len(rows), np.nan, dtype=np.float64)
    fold_reports = []
    for fold_index, task_id in enumerate(tasks):
        started = time.monotonic()
        test_indices = np.asarray(
            [index for index, row in enumerate(rows) if row["task_id"] == task_id],
            dtype=np.int64,
        )
        train_indices = np.asarray(
            [index for index, row in enumerate(rows) if row["task_id"] != task_id],
            dtype=np.int64,
        )
        checkpoint = (
            None if checkpoint_dir is None else checkpoint_dir / f"fold_{fold_index:03d}.json"
        )
        if checkpoint is not None and checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            if saved["fold"]["held_out_task_id"] != task_id:
                raise RuntimeError(f"checkpoint grouping drift: {checkpoint}")
            predictions[test_indices] = saved["predictions"]
            fold_reports.append(saved["fold"])
            continue
        selected_c = _select_c(features, rows, train_indices)
        x_train, y_train, _groups = _expand_binomial(features, rows, train_indices)
        predictions[test_indices] = _fit_predict_logistic(
            x_train,
            y_train,
            features[test_indices],
            c_value=selected_c,
        )
        fold_reports.append(
            {
                "held_out_task_id": task_id,
                "selected_c": selected_c,
                "n_train_rollouts": int(len(y_train)),
                "n_test_contexts": int(len(test_indices)),
            }
        )
        if checkpoint is not None:
            _write_json_atomic(
                checkpoint,
                {"fold": fold_reports[-1], "predictions": predictions[test_indices].tolist()},
            )
        print(
            f"[fit] fold {fold_index + 1}/{len(tasks)} {task_id} elapsed={time.monotonic() - started:.2f}s",
            flush=True,
        )
    if not np.isfinite(predictions).all():
        raise RuntimeError("cross-fitting left missing predictions")
    return predictions, fold_reports


def leave_one_task_out_prevalence(rows: list[dict[str, Any]]) -> np.ndarray:
    predictions = np.empty(len(rows), dtype=np.float64)
    for index, row in enumerate(rows):
        training = [candidate for candidate in rows if candidate["task_id"] != row["task_id"]]
        positive = sum(int(candidate["positive"]) for candidate in training)
        total = sum(int(candidate["positive"] + candidate["negative"]) for candidate in training)
        predictions[index] = (positive + 0.5) / (total + 1.0)
    return predictions


def classification_metrics(
    rows: list[dict[str, Any]],
    probability: np.ndarray,
) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    labels = []
    scores = []
    for row, score in zip(rows, probability, strict=True):
        labels.extend([1] * int(row["positive"]) + [0] * int(row["negative"]))
        scores.extend([float(score)] * int(row["positive"] + row["negative"]))
    label_array = np.asarray(labels, dtype=np.int8)
    score_array = np.asarray(scores, dtype=np.float64)
    return {
        "binomial_log_loss": binomial_log_loss(rows, probability),
        "brier": binomial_brier(rows, probability),
        "auroc": float(roc_auc_score(label_array, score_array)),
        "auprc": float(average_precision_score(label_array, score_array)),
    }


def clustered_bootstrap_log_loss_delta(
    rows: list[dict[str, Any]],
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    cluster_label: str = "base_task_id",
) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    tasks = np.asarray(sorted({str(row["task_id"]) for row in rows}))
    indices_by_task = {
        task: np.asarray(
            [index for index, row in enumerate(rows) if row["task_id"] == task],
            dtype=np.int64,
        )
        for task in tasks
    }
    positive = np.asarray([row["positive"] for row in rows], dtype=float)
    negative = np.asarray([row["negative"] for row in rows], dtype=float)
    candidate = np.clip(candidate, 1e-6, 1 - 1e-6)
    baseline = np.clip(baseline, 1e-6, 1 - 1e-6)
    difference = positive * np.log(baseline / candidate) + negative * np.log(
        (1 - baseline) / (1 - candidate)
    )
    group_difference = np.asarray([difference[indices_by_task[task]].sum() for task in tasks])
    group_size = np.asarray([(positive + negative)[indices_by_task[task]].sum() for task in tasks])
    # Same seeded cluster draws as the original loop, reduced once per cluster.
    sampled = rng.integers(len(tasks), size=(replicates, len(tasks)))
    deltas = group_difference[sampled].sum(axis=1) / group_size[sampled].sum(axis=1)
    point = binomial_log_loss(rows, candidate) - binomial_log_loss(rows, baseline)
    return {
        "delta": float(point),
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
        "bootstrap_replicates": replicates,
        "cluster": cluster_label,
    }


def prepare_reward_hacking_rows(
    impossible_result: dict[str, Any],
    activation_metadata: dict[str, dict[str, Any]],
    manifest_path: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    gate = impossible_result["reward_hacking_prevalence_gate"]
    eligible = set(gate["eligible_task_ids"])
    texts_by_hash = {
        str(row["exact_context_sha256"]): "\n".join(
            f"{message['role']}: {message['content']}" for message in row["messages"]
        )
        for row in (
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    rows = []
    texts = []
    for context in impossible_result["contexts"]:
        if context["task_id"] not in eligible or context["condition"] == "original":
            continue
        exact_hash = str(context["exact_context_sha256"])
        if exact_hash not in activation_metadata:
            raise RuntimeError(f"outcome context lacks activation: {exact_hash}")
        if exact_hash not in texts_by_hash:
            raise RuntimeError(f"outcome context lacks frozen text: {exact_hash}")
        negative = int(context["n"] - context["passed"] - context["errors"])
        rows.append(
            {
                "task_id": str(context["task_id"]),
                "condition": str(context["condition"]),
                "exact_context_sha256": exact_hash,
                "positive": int(context["passed"]),
                "negative": negative,
                "censored": int(context["errors"]),
                "n_prefix_tokens": int(activation_metadata[exact_hash]["n_prefix_tokens"]),
            }
        )
        texts.append(texts_by_hash[exact_hash])
    return rows, texts


def prepare_misalignment_rows(
    misalignment_result: dict[str, Any],
    activation_metadata: dict[str, dict[str, Any]],
    manifest_path: Path,
    *,
    group_axis: str = "exact_context_sha256",
) -> tuple[list[dict[str, Any]], list[str]]:
    """Aggregate rollout outcomes at the unique visible-prefix grain."""

    manifest_rows = [
        json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line
    ]
    manifest_by_condition = {str(row["condition_id"]): row for row in manifest_rows}
    if len(manifest_by_condition) != len(manifest_rows):
        raise RuntimeError("misalignment manifest contains duplicate condition IDs")
    aggregated: dict[str, dict[str, Any]] = {}
    texts_by_hash: dict[str, str] = {}
    for context in misalignment_result["contexts"]:
        condition_id = str(context["condition_id"])
        manifest = manifest_by_condition.get(condition_id)
        if manifest is None:
            raise RuntimeError(f"outcome condition is absent from manifest: {condition_id}")
        exact_hash = str(context["exact_context_sha256"])
        if str(manifest["exact_context_sha256"]) != exact_hash:
            raise RuntimeError(f"exact-context hash drift for {condition_id}")
        if exact_hash not in activation_metadata:
            raise RuntimeError(f"outcome context lacks activation: {exact_hash}")
        text = "\n".join(
            f"{message['role']}: {message['content']}" for message in manifest["messages"]
        )
        prior_text = texts_by_hash.setdefault(exact_hash, text)
        if prior_text != text:
            raise RuntimeError(f"duplicate exact context has text drift: {exact_hash}")
        row = aggregated.setdefault(
            exact_hash,
            {
                "task_id": exact_hash,
                "condition_ids": [],
                "exact_context_sha256": exact_hash,
                "positive": 0,
                "negative": 0,
                "censored": 0,
                "n_prefix_tokens": int(activation_metadata[exact_hash]["n_prefix_tokens"]),
            },
        )
        row["condition_ids"].append(condition_id)
        row["positive"] += int(context["n_positive"])
        row["negative"] += int(context["n_negative"])
        row["censored"] += int(context.get("n_censored", 0))
    rows = [aggregated[key] for key in sorted(aggregated)]
    if group_axis != "exact_context_sha256":
        for row in rows:
            groups = set()
            for condition in row["condition_ids"]:
                record = manifest_by_condition[condition]
                group = str(
                    record["goal_type"] if group_axis == "goal_framing" else record[group_axis]
                )
                # Latent and swap aliases include byte-identical prompts and
                # must never straddle train/test, even in a framing-held-out fit.
                if group_axis == "goal_framing" and group in {"latent", "swap"}:
                    group = "latent_or_swap"
                groups.add(group)
            if len(groups) != 1:
                raise RuntimeError(f"duplicate prefix straddles {group_axis} groups")
            row["task_id"] = groups.pop()
    texts = [texts_by_hash[row["exact_context_sha256"]] for row in rows]
    if sum(row["censored"] for row in rows):
        raise RuntimeError("misalignment prediction rows still contain censored outcomes")
    return rows, texts


def predictor_bakeoff(
    rows: list[dict[str, Any]],
    texts: list[str],
    raw: np.ndarray,
    mapped: np.ndarray,
    metadata: np.ndarray,
    *,
    cluster_label: str,
    checkpoint_root: Path | None = None,
) -> dict[str, Any]:
    """Cross-fit the frozen predictor ladder on one grouped outcome table."""

    from sklearn.feature_extraction.text import HashingVectorizer

    text_features = (
        HashingVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            n_features=2048,
            alternate_sign=False,
            norm="l2",
        )
        .transform(texts)
        .toarray()
        .astype(np.float32)
    )
    feature_sets = {
        "metadata": metadata,
        "text_metadata": np.column_stack([text_features, metadata]),
        "raw_activation_metadata": np.column_stack([raw, metadata]),
        "mapped_activation_metadata": np.column_stack([mapped, metadata]),
    }
    predictions = {"prevalence": leave_one_task_out_prevalence(rows)}
    fold_reports = {}
    for name, features in feature_sets.items():
        predictions[name], fold_reports[name] = leave_one_task_out_predictions(
            features,
            rows,
            checkpoint_dir=None if checkpoint_root is None else checkpoint_root / name,
        )
    models = {
        name: {
            "metrics": classification_metrics(rows, probability),
            "predictions": [float(value) for value in probability],
            "folds": fold_reports.get(name),
        }
        for name, probability in predictions.items()
    }
    primary = clustered_bootstrap_log_loss_delta(
        rows,
        predictions["raw_activation_metadata"],
        predictions["text_metadata"],
        cluster_label=cluster_label,
    )
    mapping = clustered_bootstrap_log_loss_delta(
        rows,
        predictions["mapped_activation_metadata"],
        predictions["raw_activation_metadata"],
        cluster_label=cluster_label,
    )
    return {
        "models": models,
        "primary_contrast": primary,
        "mapping_contrast": mapping,
        "activation_signal_detected": primary["ci95_high"] < 0,
        "mapping_signal_detected": mapping["ci95_high"] < 0,
    }


def _format_report(report: dict[str, Any]) -> str:
    impossible = report["reward_hacking_feasibility"]
    misalignment = report.get("misaligned_action_feasibility")
    lines = [
        "# Pre-action agent-risk feasibility result",
        "",
        f"Status: **{report['claim_status']}**",
        "",
        "## Frozen gates",
        "",
        f"- Reward hacking: `{impossible['gate']['passed']}`; "
        f"{impossible['gate']['n_positive']} positive, "
        f"{impossible['gate']['n_negative']} negative, "
        f"{impossible['gate']['n_censored']} censored, "
        f"{impossible['gate']['n_mixed_contexts']} mixed contexts, and "
        f"{impossible['gate']['n_eligible_tasks']} competent tasks.",
    ]
    if misalignment:
        lines.append(
            f"- Misaligned action: `{misalignment['gate_passed']}`; "
            f"{misalignment['n_positive']} positive, {misalignment['n_negative']} negative, "
            f"{misalignment['n_censored']} censored, and "
            f"{misalignment['n_mixed_contexts']} mixed contexts."
        )
    lines.extend(["", "## Prediction result", ""])
    if impossible["prediction_status"] != "completed":
        lines.append(
            "Prediction was not fit because the frozen reward-hacking prevalence gate failed. "
            "This is an underpowered environment/model result, not evidence that pre-action "
            "prediction is impossible."
        )
    else:
        lines.extend(
            [
                "| Predictor | Log loss | Brier | AUROC | AUPRC |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for name, model in impossible["models"].items():
            metrics = model["metrics"]
            lines.append(
                f"| {name} | {metrics['binomial_log_loss']:.4f} | "
                f"{metrics['brier']:.4f} | {metrics['auroc']:.3f} | "
                f"{metrics['auprc']:.3f} |"
            )
        contrast = impossible["primary_contrast"]
        lines.extend(
            [
                "",
                "Primary raw-activation-minus-text log-loss contrast: "
                f"{contrast['delta']:.4f} "
                f"(task-cluster 95% CI {contrast['ci95_low']:.4f}, "
                f"{contrast['ci95_high']:.4f}).",
            ]
        )
    if misalignment:
        lines.extend(["", "### Misaligned action", ""])
        lines.append(
            f"Split: `{report['analysis_protocol']['outer_split']['misaligned_action']}`; "
            f"groups: {misalignment.get('n_groups', 'not fit')}; "
            f"metadata: `{report['analysis_protocol']['misalignment_metadata']}`."
        )
        lines.append("")
        if misalignment["prediction_status"] != "completed":
            lines.append(
                "Prediction was not fit because the frozen misaligned-action prevalence and "
                "censoring gates did not both pass, or the required activation inputs were absent."
            )
        else:
            lines.extend(
                [
                    "| Predictor | Log loss | Brier | AUROC | AUPRC |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for name, model in misalignment["models"].items():
                metrics = model["metrics"]
                lines.append(
                    f"| {name} | {metrics['binomial_log_loss']:.4f} | "
                    f"{metrics['brier']:.4f} | {metrics['auroc']:.3f} | "
                    f"{metrics['auprc']:.3f} |"
                )
            contrast = misalignment["primary_contrast"]
            lines.extend(
                [
                    "",
                    "Misaligned-action raw-activation-minus-text log-loss contrast: "
                    f"{contrast['delta']:.4f} "
                    f"({contrast['cluster']}-cluster 95% CI {contrast['ci95_low']:.4f}, "
                    f"{contrast['ci95_high']:.4f}).",
                ]
            )
            mapping = misalignment["mapping_contrast"]
            lines.extend(
                [
                    "",
                    "Mapped-minus-raw log-loss contrast: "
                    f"{mapping['delta']:.4f} (95% CI {mapping['ci95_low']:.4f}, {mapping['ci95_high']:.4f}).",
                    "Intervals resample fixed out-of-fold predictions; they do not refit models. "
                    "All prompts are from one information-leak scenario. Structured-fold checks are "
                    "exploratory, with only three or four groups.",
                ]
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            report["interpretation"],
            "",
        ]
    )
    return "\n".join(lines)


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    group_axis = getattr(args, "misalignment_group_axis", "exact_context_sha256")
    metadata_mode = getattr(args, "misalignment_metadata", "length")
    # Cache namespaces bind immutable input files, code, dependencies and every
    # analysis option; do not hash recomputed floating-point features.
    import importlib.metadata

    input_paths = [args.impossible_result, args.impossible_manifest, args.map_artifact]
    for key in ("misalignment_result", "misalignment_manifest"):
        path = getattr(args, key, None)
        if path is not None:
            input_paths.append(path)
    for key, pattern in (
        ("misalignment_rollout_root", "context_*/*"),
        ("impossible_capture_root", "chunk_*"),
    ):
        root = getattr(args, key, None)
        if root is not None:
            input_paths.extend(path for path in root.glob(pattern) if path.is_file())
    reproducibility = {
        "source_sha256": _sha256(Path(__file__)),
        "input_hashes": {str(path): _sha256(path) for path in input_paths},
        "selected_layer": args.selected_layer,
        "misalignment_group_axis": group_axis,
        "misalignment_metadata": metadata_mode,
        "versions": {
            name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn")
        },
    }
    cache_key = hashlib.sha256(json.dumps(reproducibility, sort_keys=True).encode()).hexdigest()
    checkpoint_root = args.output_dir / "checkpoints" / cache_key
    impossible_result = json.loads(args.impossible_result.read_text(encoding="utf-8"))
    gate = impossible_result["reward_hacking_prevalence_gate"]
    execution_integrity = {
        "run_passed": bool(impossible_result.get("passed", False)),
        "realized_rollouts": int(impossible_result.get("realized_rollouts", 0)),
        "requested_rollouts": int(impossible_result.get("requested_rollouts", 0)),
        "technical_errors": int(impossible_result.get("technical_errors", 0)),
    }
    execution_integrity["passed"] = bool(
        execution_integrity["run_passed"]
        and execution_integrity["requested_rollouts"] > 0
        and execution_integrity["realized_rollouts"] == execution_integrity["requested_rollouts"]
        and execution_integrity["technical_errors"] == 0
    )
    can_fit = bool(gate["passed"] and execution_integrity["passed"])
    misalignment = None
    misalignment_source = None
    if args.misalignment_result is not None:
        source = json.loads(args.misalignment_result.read_text(encoding="utf-8"))
        misalignment_source = source
        misalignment_gate_passed = bool(
            source.get("prevalence_gate_passed") and source.get("censoring_gate_passed")
        )
        misalignment = {
            "gate_passed": misalignment_gate_passed,
            "prevalence_gate_passed": bool(source.get("prevalence_gate_passed")),
            "censoring_gate_passed": bool(source.get("censoring_gate_passed")),
            "n_positive": int(source["n_positive"]),
            "n_negative": int(source["n_negative"]),
            "n_censored": int(source.get("n_censored", 0)),
            "n_mixed_contexts": int(source["n_mixed_contexts"]),
            "public_test_role": source["public_test_role"],
            "prediction_status": (
                "not_run_frozen_gate_failed"
                if not misalignment_gate_passed
                else "not_run_missing_inputs"
            ),
        }
    feasibility: dict[str, Any] = {
        "gate": gate,
        "execution_integrity": execution_integrity,
        "prediction_status": (
            "not_run_frozen_gate_failed"
            if not gate["passed"]
            else "not_run_execution_integrity_failed"
        ),
    }
    activation_signal = False
    mapping_signal = False
    if can_fit:
        activations, activation_metadata = load_impossible_activations(
            args.impossible_capture_root,
            selected_layer=args.selected_layer,
        )
        rows, texts = prepare_reward_hacking_rows(
            impossible_result,
            activation_metadata,
            args.impossible_manifest,
        )
        exact_hashes = [row["exact_context_sha256"] for row in rows]
        raw = np.stack([activations[exact_hash] for exact_hash in exact_hashes])
        mapped = apply_context_map(raw, args.map_artifact)
        metadata = np.asarray(
            [
                [
                    float(row["condition"] == "conflicting"),
                    math.log1p(float(row["n_prefix_tokens"])),
                ]
                for row in rows
            ],
            dtype=np.float32,
        )
        bakeoff = predictor_bakeoff(
            rows,
            texts,
            raw,
            mapped,
            metadata,
            cluster_label="base_task_id",
            checkpoint_root=checkpoint_root / "reward_hacking",
        )
        activation_signal = bool(bakeoff["activation_signal_detected"])
        mapping_signal = bool(bakeoff["mapping_signal_detected"])
        feasibility.update(
            {
                "prediction_status": "completed",
                "selected_layer": args.selected_layer,
                "n_contexts": len(rows),
                "n_tasks": len({row["task_id"] for row in rows}),
                **bakeoff,
                "context_rows": rows,
            }
        )
    misalignment_activation_signal = False
    misalignment_mapping_signal = False
    misalignment_rollout_root = getattr(args, "misalignment_rollout_root", None)
    misalignment_manifest = getattr(args, "misalignment_manifest", None)
    if (
        misalignment is not None
        and misalignment_source is not None
        and misalignment["gate_passed"]
        and misalignment_rollout_root is not None
        and misalignment_manifest is not None
    ):
        misalignment_activations, misalignment_activation_metadata = load_misalignment_activations(
            misalignment_rollout_root,
            selected_layer=args.selected_layer,
        )
        misalignment_rows, misalignment_texts = prepare_misalignment_rows(
            misalignment_source,
            misalignment_activation_metadata,
            misalignment_manifest,
            group_axis=group_axis,
        )
        misalignment_hashes = [row["exact_context_sha256"] for row in misalignment_rows]
        misalignment_raw = np.stack(
            [misalignment_activations[exact_hash] for exact_hash in misalignment_hashes]
        )
        misalignment_mapped = apply_context_map(misalignment_raw, args.map_artifact)
        misalignment_metadata = np.asarray(
            [[math.log1p(float(row["n_prefix_tokens"]))] for row in misalignment_rows],
            dtype=np.float32,
        )
        if metadata_mode == "conditions":
            manifest_records = [
                json.loads(line) for line in misalignment_manifest.read_text().splitlines() if line
            ]
            manifest_by_id = {row["condition_id"]: row for row in manifest_records}
            conditions = []
            for row in misalignment_rows:
                record = manifest_by_id[row["condition_ids"][0]]
                framing = record["goal_type"]
                conditions.append(
                    [
                        float(framing == "ambiguous"),
                        float(framing == "explicit"),
                        float(framing in {"latent", "swap"}),
                        float(record["goal_value"] == "acceleration"),
                        float(record["goal_value"] == "safety"),
                        float(record["urgency_type"] == "replacement"),
                        float(record["urgency_type"] == "restriction"),
                    ]
                )
            misalignment_metadata = np.column_stack(
                [misalignment_metadata, np.asarray(conditions, dtype=np.float32)]
            )
        misalignment_bakeoff = predictor_bakeoff(
            misalignment_rows,
            misalignment_texts,
            misalignment_raw,
            misalignment_mapped,
            misalignment_metadata,
            cluster_label=group_axis,
            checkpoint_root=checkpoint_root / "misaligned_action",
        )
        misalignment_activation_signal = bool(misalignment_bakeoff["activation_signal_detected"])
        misalignment_mapping_signal = bool(misalignment_bakeoff["mapping_signal_detected"])
        misalignment.update(
            {
                "prediction_status": "completed",
                "selected_layer": args.selected_layer,
                "n_contexts": len(misalignment_rows),
                "n_groups": len({row["task_id"] for row in misalignment_rows}),
                **misalignment_bakeoff,
                "context_rows": misalignment_rows,
            }
        )
    feasibility_claim = bool(
        can_fit
        and activation_signal
        and misalignment is not None
        and misalignment.get("prediction_status") == "completed"
        and misalignment_activation_signal
    )
    if feasibility_claim:
        claim_status = "development signal detected; confirmation still required"
        interpretation = (
            "Pre-action activations improved held-out prediction over the frozen text baseline "
            "in both the reward-hacking and misaligned-action public-development arms. The planned "
            "confirmatory environments and independent frozen risk directions remain necessary "
            "before making the full general claim."
        )
    elif can_fit and activation_signal:
        if misalignment is None:
            claim_status = (
                "reward-hacking development signal detected; misaligned-action arm not evaluated"
            )
            interpretation = (
                "The reward-hacking development arm contains a task-held-out activation signal, "
                "but the misaligned-action arm was not supplied to this analysis. This supports "
                "only a narrow development result, not general pre-action agent-risk prediction."
            )
        else:
            if misalignment.get("prediction_status") == "completed":
                claim_status = (
                    "reward-hacking development signal detected; no misaligned-action signal"
                )
                interpretation = (
                    "The reward-hacking development arm contains a task-held-out activation "
                    "signal, but pre-action activations did not improve held-out misaligned-action "
                    "prediction over text with a grouped interval excluding zero."
                )
            else:
                claim_status = (
                    "reward-hacking development signal detected; cross-construct gate failed"
                )
                interpretation = (
                    "The reward-hacking development arm contains a task-held-out activation signal, "
                    "but the misaligned-action arm did not complete a gate-valid prediction fit. "
                    "This supports only a narrow development result, not general pre-action "
                    "agent-risk prediction."
                )
    elif can_fit:
        claim_status = "no reward-hacking development signal detected"
        interpretation = (
            "The reward-hacking arm was sufficiently populated, but raw pre-action activations did "
            "not improve held-out log loss over the frozen text baseline with a task-clustered "
            "interval excluding zero."
        )
    elif not gate["passed"]:
        claim_status = "reward-hacking feasibility gate failed"
        interpretation = (
            "The environment/model pairing did not yield enough competent, mixed reward-hacking "
            "outcomes for a valid prospective prediction test. Per the frozen protocol, fitting a "
            "classifier would be post-selection on rare trajectories, so no prediction claim is made."
        )
    else:
        claim_status = "reward-hacking execution integrity failed"
        interpretation = (
            "The recorded reward-hacking prevalence gate passed, but the rollout did not meet "
            "the independent completeness and zero-technical-error checks. No predictor was fit."
        )
    report = {
        "schema_version": "context_risk_prospective_analysis_v1",
        "public_test_role": "development_only",
        "reward_hacking_feasibility": feasibility,
        "misaligned_action_feasibility": misalignment,
        "activation_signal_detected": activation_signal,
        "mapping_signal_detected": mapping_signal,
        "misalignment_activation_signal_detected": misalignment_activation_signal,
        "misalignment_mapping_signal_detected": misalignment_mapping_signal,
        "feasibility_claim_supported": feasibility_claim,
        "confirmatory_environments_completed": False,
        "full_claim_supported": False,
        "claim_status": claim_status,
        "interpretation": interpretation,
        "reproducibility": reproducibility,
        "analysis_protocol": {
            "misalignment_metadata": metadata_mode,
            "outer_split": {
                "reward_hacking": "leave_one_base_task_out",
                "misaligned_action": f"leave_one_{group_axis}_out",
            },
            "inner_split": "up_to_five_grouped_folds",
            "regularization_grid": list(C_GRID),
            "primary_metric": "held-out binomial log loss",
            "uncertainty": {
                "reward_hacking": "5000-replicate base-task clustered bootstrap",
                "misaligned_action": f"5000-replicate {group_axis} clustered bootstrap of fixed out-of-fold predictions",
            },
            "primary_contrast": "raw activation + metadata minus fixed hashed text + metadata",
            "mapping_contrast": "mapped activation + metadata minus raw activation + metadata",
            "single_class_training_policy": "Jeffreys-smoothed prevalence: (positive + 0.5)/(total + 1)",
            "small_training_group_policy": "C=0.01 when fewer than three training groups (inherited fallback)",
            "fit": "frequency-weighted primal liblinear, same summed L2 objective as repeated rows; tol=1e-8; convergence warnings fatal",
        },
        "inputs": {
            "impossible_result": str(args.impossible_result),
            "impossible_result_sha256": _sha256(args.impossible_result),
            "impossible_manifest": str(args.impossible_manifest),
            "impossible_manifest_sha256": _sha256(args.impossible_manifest),
            "map_artifact": str(args.map_artifact),
            "map_artifact_sha256": _sha256(args.map_artifact),
            "misalignment_result": (
                str(args.misalignment_result) if args.misalignment_result is not None else None
            ),
            "misalignment_rollout_root": (
                str(misalignment_rollout_root) if misalignment_rollout_root is not None else None
            ),
            "misalignment_manifest": (
                str(misalignment_manifest) if misalignment_manifest is not None else None
            ),
        },
        "analysis_completed": True,
        "passed": feasibility_claim,
    }
    _write_json_atomic(args.output_dir / "analysis_result.json", report)
    _write_text_atomic(args.output_dir / "results.md", _format_report(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impossible-result", type=Path, required=True)
    parser.add_argument("--impossible-capture-root", type=Path, required=True)
    parser.add_argument("--impossible-manifest", type=Path, required=True)
    parser.add_argument("--map-artifact", type=Path, required=True)
    parser.add_argument("--misalignment-result", type=Path)
    parser.add_argument("--misalignment-rollout-root", type=Path)
    parser.add_argument("--misalignment-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selected-layer", type=int, default=44)
    parser.add_argument(
        "--misalignment-metadata", choices=("length", "conditions"), default="length"
    )
    parser.add_argument(
        "--misalignment-group-axis",
        choices=("exact_context_sha256", "goal_framing", "urgency_type", "goal_value"),
        default="exact_context_sha256",
    )
    args = parser.parse_args()
    report = run_analysis(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
