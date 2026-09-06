#!/usr/bin/env python3
"""Fit and evaluate grouped Qwen3.8 context→answer ridge maps."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import hydra  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402


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


def _write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as fh:
        np.savez(fh, **arrays)
        temporary = Path(fh.name)
    os.replace(temporary, path)


def pooled_r2(prediction: np.ndarray, target: np.ndarray) -> float:
    """Variance-weighted multi-output R² against the evaluation-set mean."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    residual = float(np.square(target - prediction).sum())
    centered = target - target.mean(axis=0, keepdims=True)
    total = float(np.square(centered).sum())
    return float("nan") if total <= 0 else 1.0 - residual / total


def normalized_rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    rmse = float(np.sqrt(np.square(target - prediction).mean()))
    scale = float(np.sqrt(np.square(target - target.mean(axis=0, keepdims=True)).mean()))
    return float("nan") if scale <= 0 else rmse / scale


def load_capture_bank(
    capture_root: Path,
    *,
    layers: list[int],
    hidden_dim: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    """Load only sentinel-complete chunks, verifying hashes and row alignment."""
    done_paths = sorted(capture_root.rglob("chunk_*.done.json"))
    if not done_paths:
        raise FileNotFoundError(f"no completed capture chunks under {capture_root}")
    contexts = []
    answers = []
    metadata: list[dict[str, Any]] = []
    provenance = []
    seen_pair_ids = set()
    expected_tail = (len(layers), hidden_dim)
    for done_path in done_paths:
        done = json.loads(done_path.read_text(encoding="utf-8"))
        stem = done_path.name.removesuffix(".done.json")
        npz_path = done_path.parent / f"{stem}.npz"
        rows_path = done_path.parent / f"{stem}.rows.jsonl"
        if _sha256(npz_path) != done["npz_sha256"] or _sha256(rows_path) != done["rows_sha256"]:
            raise RuntimeError(f"capture hash mismatch for {done_path}")
        with np.load(npz_path) as arrays:
            context = np.asarray(arrays["context"], dtype=np.float32)
            answer = np.asarray(arrays["answer"], dtype=np.float32)
        rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
        if context.shape != answer.shape or context.shape[1:] != expected_tail:
            raise RuntimeError(
                f"shape mismatch for {npz_path}: context={context.shape}, answer={answer.shape}"
            )
        if len(rows) != len(context) or len(rows) != int(done["accepted_rows"]):
            raise RuntimeError(f"row-count mismatch for {done_path}")
        for row in rows:
            pair_id = str(row["pair_id"])
            if pair_id in seen_pair_ids:
                raise RuntimeError(f"duplicate pair_id across capture chunks: {pair_id}")
            seen_pair_ids.add(pair_id)
        contexts.append(context)
        answers.append(answer)
        metadata.extend(rows)
        provenance.append(
            {
                "done_path": str(done_path),
                "fingerprint": done["fingerprint"],
                "npz_sha256": done["npz_sha256"],
                "rows_sha256": done["rows_sha256"],
                "accepted_rows": done["accepted_rows"],
            }
        )
    return np.concatenate(contexts), np.concatenate(answers), metadata, provenance


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {
        "pooled_r2": pooled_r2(prediction, target),
        "normalized_rmse": normalized_rmse(prediction, target),
    }


def _knn_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    *,
    k: int,
    query_rows: int,
) -> np.ndarray:
    """Cosine kNN target retrieval with bounded query-side memory."""
    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    x_eval = np.asarray(x_eval, dtype=np.float32)
    center = x_train.mean(axis=0, keepdims=True)
    train = x_train - center
    train /= np.linalg.norm(train, axis=1, keepdims=True).clip(1e-12)
    output = np.empty((len(x_eval), y_train.shape[1]), dtype=np.float32)
    k = min(k, len(train))
    for start in range(0, len(x_eval), query_rows):
        query = x_eval[start : start + query_rows] - center
        query /= np.linalg.norm(query, axis=1, keepdims=True).clip(1e-12)
        similarities = query @ train.T
        neighbors = np.argpartition(similarities, -k, axis=1)[:, -k:]
        output[start : start + len(query)] = y_train[neighbors].mean(axis=1)
    return output


def fit_ridge_layer(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    lambdas: list[float],
    device: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Select a primal ridge penalty on grouped validation rows and test once."""
    import torch

    dev = torch.device(device)
    xtr = torch.as_tensor(x_train, dtype=torch.float64, device=dev)
    ytr = torch.as_tensor(y_train, dtype=torch.float64, device=dev)
    xval = torch.as_tensor(x_validation, dtype=torch.float64, device=dev)
    xtest = torch.as_tensor(x_test, dtype=torch.float64, device=dev)
    x_mean = xtr.mean(dim=0)
    x_scale = xtr.std(dim=0, correction=0).clamp_min(1e-9)
    y_mean = ytr.mean(dim=0)
    xtr = (xtr - x_mean) / x_scale
    xval = (xval - x_mean) / x_scale
    xtest = (xtest - x_mean) / x_scale
    y_centered = ytr - y_mean
    gram = xtr.T @ xtr
    cross = xtr.T @ y_centered
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    eigenvalues = eigenvalues.clamp_min(0)
    projected_cross = eigenvectors.T @ cross
    projected_validation = xval @ eigenvectors
    projected_test = xtest @ eigenvectors
    validation_scores = []
    best_lambda = None
    best_validation_r2 = -float("inf")
    for penalty in lambdas:
        prediction = (
            projected_validation / (eigenvalues + float(penalty))
        ) @ projected_cross + y_mean
        prediction_np = prediction.float().cpu().numpy()
        metrics = _metrics(prediction_np, y_validation)
        validation_scores.append({"lambda": float(penalty), **metrics})
        if metrics["pooled_r2"] > best_validation_r2:
            best_validation_r2 = metrics["pooled_r2"]
            best_lambda = float(penalty)
    if best_lambda is None:
        raise RuntimeError("ridge validation failed to select a penalty")
    inverse = 1.0 / (eigenvalues + best_lambda)
    test_prediction = (projected_test * inverse) @ projected_cross + y_mean
    weights = eigenvectors @ (projected_cross * inverse[:, None])
    test_prediction_np = test_prediction.float().cpu().numpy()
    report = {
        "selected_lambda": best_lambda,
        "validation": validation_scores,
        "test": _metrics(test_prediction_np, y_test),
        "effective_dof": float((eigenvalues / (eigenvalues + best_lambda)).sum().cpu()),
    }
    artifact = {
        "weight": weights.float().cpu().numpy(),
        "x_mean": x_mean.float().cpu().numpy(),
        "x_scale": x_scale.float().cpu().numpy(),
        "y_mean": y_mean.float().cpu().numpy(),
        "test_prediction": test_prediction_np,
    }
    return report, artifact


def run_fit(cfg: DictConfig) -> dict[str, Any]:
    capture_root = Path(str(cfg.capture_root))
    output_dir = Path(str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    layers = [int(layer) for layer in cfg.capture_layers]
    context, answer, rows, provenance = load_capture_bank(
        capture_root,
        layers=layers,
        hidden_dim=int(cfg.hidden_dim),
    )
    partitions = np.asarray([row["map_partition"] for row in rows])
    indices = {name: np.flatnonzero(partitions == name) for name in ("train", "validation", "test")}
    if any(len(value) == 0 for value in indices.values()):
        raise RuntimeError(
            f"empty grouped map partition: { {key: len(value) for key, value in indices.items()} }"
        )
    minimum_ratio = float(cfg.minimum_train_rows_per_dimension)
    if len(indices["train"]) < minimum_ratio * int(cfg.hidden_dim):
        raise RuntimeError(
            f"under-determined map refused: train_rows={len(indices['train'])}, "
            f"hidden_dim={cfg.hidden_dim}, required_ratio={minimum_ratio}"
        )
    split_groups: dict[str, set[tuple[str, str]]] = {
        name: {(rows[index]["source"], rows[index]["group_id"]) for index in split_indices}
        for name, split_indices in indices.items()
    }
    if any(
        split_groups[left] & split_groups[right]
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise RuntimeError("trajectory group leakage across map partitions")
    started = time.monotonic()
    layer_reports = {}
    artifacts = {}
    for layer_index, layer in enumerate(layers):
        x = context[:, layer_index]
        y = answer[:, layer_index]
        train = indices["train"]
        validation = indices["validation"]
        test = indices["test"]
        fit_report, artifact = fit_ridge_layer(
            x[train],
            y[train],
            x[validation],
            y[validation],
            x[test],
            y[test],
            lambdas=[float(value) for value in cfg.lambdas],
            device=str(cfg.fit_device),
        )
        y_mean = y[train].mean(axis=0, keepdims=True)
        identity_bias = (y[train] - x[train]).mean(axis=0, keepdims=True)
        knn_prediction = _knn_predict(
            x[train],
            y[train],
            x[test],
            k=int(cfg.knn_k),
            query_rows=int(cfg.knn_query_rows),
        )
        fit_report["test_baselines"] = {
            "mean": _metrics(np.repeat(y_mean, len(test), axis=0), y[test]),
            "identity_plus_bias": _metrics(x[test] + identity_bias, y[test]),
            f"cosine_knn_k{int(cfg.knn_k)}": _metrics(knn_prediction, y[test]),
        }
        layer_reports[str(layer)] = fit_report
        artifacts[layer] = artifact
        print(
            f"[context-risk-map] layer={layer} lambda={fit_report['selected_lambda']:g} "
            f"test_r2={fit_report['test']['pooled_r2']:.4f} "
            f"elapsed={time.monotonic() - started:.1f}s",
            flush=True,
        )
    selected_layer = max(
        layers,
        key=lambda layer: max(
            item["pooled_r2"] for item in layer_reports[str(layer)]["validation"]
        ),
    )
    artifact = artifacts[selected_layer]
    artifact_path = output_dir / f"map_layer_{selected_layer}.npz"
    _write_npz_atomic(
        artifact_path,
        weight=artifact["weight"],
        x_mean=artifact["x_mean"],
        x_scale=artifact["x_scale"],
        y_mean=artifact["y_mean"],
    )
    source_counts = {}
    for row in rows:
        source_counts[row["source"]] = source_counts.get(row["source"], 0) + 1
    report = {
        "schema_version": "context_risk_qwen38_map_fit_v1",
        "model_id": str(cfg.model_id),
        "model_revision": str(cfg.model_revision),
        "capture_root": str(capture_root),
        "capture_chunks": provenance,
        "n_rows": len(rows),
        "partition_rows": {key: len(value) for key, value in indices.items()},
        "partition_groups": {key: len(value) for key, value in split_groups.items()},
        "source_rows": source_counts,
        "layers": layer_reports,
        "selected_layer": selected_layer,
        "selection_rule": "highest grouped-validation pooled R2; safety labels never used",
        "map_artifact": str(artifact_path),
        "map_artifact_sha256": _sha256(artifact_path),
        "elapsed_seconds": time.monotonic() - started,
        "passed": layer_reports[str(selected_layer)]["test"]["pooled_r2"] > 0,
    }
    _write_json_atomic(output_dir / "run_result.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


@hydra.main(
    version_base="1.3",
    config_path="../configs/eval",
    config_name="context_risk_qwen38_fit_map",
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True), flush=True)
    run_fit(cfg)


if __name__ == "__main__":
    main()
