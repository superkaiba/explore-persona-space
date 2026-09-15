"""CPU-only, conversation-held-out K=1/K=5 turn-1→12 calibration pilot.

The K=1 target is draw zero of the same bank used for K=5. A K=5 target is
the equal-weight mean of five answer-token means, never a pooled-token mean.
Lambda selection uses the corrected #825 four-inner-group-fold implementation.
All bootstrap intervals condition on the fitted maps and captured draw bank.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import issue825_fit_cells as fit825  # noqa: E402
from explore_persona_space.analysis.turn_transfer_calibration import (  # noqa: E402
    adapted_predictions,
    calibrate,
)

MODELS = ("instruct", "pretrained")
TURNS = (1, 12)
KS = (1, 5)
METHODS = ("raw", "bias", "bias_scale", "identity_bias", "own12")
N_FOLDS = 6
N_INNER = 4
SEED = 0
INNER_SEED = 4242
BOOTSTRAPS = 1000
CONTEXT_COS_MIN = 0.995
RECIPE = {
    "version": 1,
    "turns": TURNS,
    "draw_ids": list(range(5)),
    "k_values": KS,
    "outer_folds": N_FOLDS,
    "outer_seed": SEED,
    "inner_folds": N_INNER,
    "inner_seed": "4242+outer_fold",
    "lambda_grid": {"log10_start": -2, "log10_stop": 4, "count": 13},
    "selector": "issue825_fit_cells._inner_cv_rss_curve_batched",
    "dtype": "float64",
    "calibration_target_k": "train_k",
    "bootstrap": {"replicates": BOOTSTRAPS, "seed": SEED, "refit": False},
}


def sha256(path: Path) -> str:
    """Hash a local artifact without loading its complete bytes into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    """Commit JSON atomically and reject non-finite JSON numbers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_npz(path: Path, **arrays: np.ndarray) -> dict:
    """Persist plain numeric/Unicode arrays atomically, returning a hash receipt."""
    if any(np.asarray(value).dtype.hasobject for value in arrays.values()):
        raise ValueError("object arrays are forbidden")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)
    return {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}


def validate_capture(arrays: dict, *, expected_dim: int = 3584) -> dict:
    """Validate one capture bank and expose only complete, draw-consistent panels."""
    required = {"conv_id", "turn", "draw_id", "context", "answer"}
    if not required <= arrays.keys():
        raise ValueError(f"capture is missing fields: {sorted(required - arrays.keys())}")
    a = {name: np.asarray(arrays[name]) for name in required}
    n = len(a["conv_id"])
    if not n or a["conv_id"].dtype.kind != "U" or a["conv_id"].shape != (n,):
        raise ValueError("conv_id must be nonempty one-dimensional Unicode")
    for name in ("turn", "draw_id"):
        if a[name].shape != (n,) or a[name].dtype.kind not in "iu":
            raise ValueError(f"{name} must contain one integer per row")
    if not set(a["turn"]) <= set(TURNS) or not set(a["draw_id"]) <= set(range(5)):
        raise ValueError("unexpected turn or draw id")
    for name in ("context", "answer"):
        if a[name].shape != (n, expected_dim) or a[name].dtype.kind != "f":
            raise ValueError(f"invalid {name} feature shape/dtype")
        if not np.isfinite(a[name]).all():
            raise ValueError(f"nonfinite {name} capture")
    keys = list(zip(a["conv_id"].tolist(), a["turn"].tolist(), a["draw_id"].tolist(), strict=True))
    if len(set(keys)) != n:
        raise ValueError("duplicate conversation/turn/draw capture")
    lookup = {key: row for row, key in enumerate(keys)}
    complete, excluded, banks, parity = [], {}, {}, []
    for conv in np.unique(a["conv_id"]):
        missing = [
            (turn, draw) for turn in TURNS for draw in range(5) if (conv, turn, draw) not in lookup
        ]
        if missing:
            excluded[str(conv)] = {"missing_turn_draw": missing}
            continue
        complete.append(str(conv))
        banks[str(conv)] = {}
        for turn in TURNS:
            rows = np.array([lookup[conv, turn, draw] for draw in range(5)])
            contexts = a["context"][rows]
            # Identical prefixes can differ slightly under bf16 batch kernels.
            # Match the GPU numeric-parity gate, then freeze draw-zero X for BOTH K arms.
            c64 = contexts.astype(np.float64)
            norms = np.linalg.norm(c64, axis=1)
            if np.any(norms == 0):
                raise ValueError("zero-norm context prevents numeric-parity verification")
            cos_min = float(np.min((c64 @ c64[0]) / (norms * norms[0])))
            if cos_min < CONTEXT_COS_MIN:
                raise ValueError(f"context differs across draws: {conv}, turn {turn}")
            parity.append(
                {
                    "conv_id": str(conv),
                    "turn": turn,
                    "cos_min": cos_min,
                    "max_abs": float(np.abs(c64 - c64[0]).max()),
                }
            )
            answers = a["answer"][rows].astype(np.float64)
            banks[str(conv)][turn] = (contexts[0].copy(), answers[0].copy(), answers.mean(axis=0))
    return {
        "complete": complete,
        "excluded": excluded,
        "banks": banks,
        "rows": n,
        "context_parity": parity,
    }


def load_capture_chunk(chunk: Path, expected_hash: str, fingerprint: str) -> tuple[dict, list]:
    """Verify a chunk's manifest and raw files, preserving row-to-vector alignment."""
    manifest_path = chunk / "manifest.json"
    if sha256(manifest_path) != expected_hash:
        raise ValueError(f"capture hash mismatch: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest["fingerprint"] != fingerprint or manifest["kind"] != "capture":
        raise ValueError("capture chunk fingerprint/kind mismatch")
    if set(manifest["files"]) != {"vectors.npz", "rows.jsonl"}:
        raise ValueError("capture chunk must contain exactly vectors.npz and rows.jsonl")
    for name, info in manifest["files"].items():
        path = chunk / name
        if path.stat().st_size != info["bytes"] or sha256(path) != info["sha256"]:
            raise ValueError(f"capture hash mismatch: {path}")
    with np.load(chunk / "vectors.npz", allow_pickle=False) as saved:
        values = {name: saved[name] for name in ("conv_id", "turn", "draw_id", "context", "answer")}
    if len(values["conv_id"]) != manifest["n_rows"]:
        raise ValueError("capture chunk row count mismatch")
    with (chunk / "rows.jsonl").open(encoding="utf-8") as handle:
        metadata = [json.loads(line) for line in handle]
    if len(metadata) != manifest["n_rows"]:
        raise ValueError("capture row-metadata count mismatch")
    for row, info in enumerate(metadata):
        if (
            info["conv_id"] != str(values["conv_id"][row])
            or info["turn"] != int(values["turn"][row])
            or info["draw_id"] != int(values["draw_id"][row])
        ):
            raise ValueError("capture row metadata does not match vector ordering")
    return values, metadata


def validate_capture_config(config: dict, manifest: dict, expected_model: str | None) -> None:
    """Bind the capture to its declared model, five-draw generation and layer recipe."""
    if config["layer"] != 19 or config["context_cos_min"] != CONTEXT_COS_MIN:
        raise ValueError("capture layer/context-parity recipe mismatch")
    generation = manifest["generation_config"]
    generation_hash = hashlib.sha256(json.dumps(generation, sort_keys=True).encode()).hexdigest()
    if (
        generation_hash != manifest["generation_fingerprint"]
        or generation_hash != config["generation_fingerprint"]
    ):
        raise ValueError("capture generation provenance mismatch")
    if generation["n"] != 5 or generation["turns"] != list(TURNS):
        raise ValueError("capture generation draw/turn recipe mismatch")
    if expected_model is not None and generation["model"] != expected_model:
        raise ValueError("capture belongs to a different model")


def load_capture(
    root: Path, *, expected_dim: int = 3584, expected_model: str | None = None
) -> tuple[dict, dict]:
    """Load exactly the hash-verified shards named by a completed capture manifest."""
    manifest_path = root / "summary.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete":
        raise ValueError(f"capture manifest is not complete: {manifest_path}")
    config_path = root / "config.json"
    if sha256(config_path) != manifest["config_sha256"]:
        raise ValueError("capture config hash mismatch")
    config = json.loads(config_path.read_text())
    if (
        hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        != manifest["fingerprint"]
    ):
        raise ValueError("capture config fingerprint mismatch")
    validate_capture_config(config, manifest, expected_model)
    gates = manifest["hook_gates"]
    if not gates or any(
        g["status"] != "pass"
        or g["same_forward_equal"] is not True
        or g["layer"] != 19
        or g["max_abs"] != 0
        for g in gates
    ):
        raise ValueError("capture hook/hidden-state layer gate did not pass")
    shards = manifest["chunks"]
    if not shards:
        raise ValueError("capture manifest has no shards")
    collected = {name: [] for name in ("conv_id", "turn", "draw_id", "context", "answer")}
    seen = set()
    prefix_hashes = {}
    for shard in shards:
        chunk = (root / shard["path"]).resolve()
        if not chunk.is_relative_to(root.resolve()) or chunk in seen:
            raise ValueError("capture shard path escapes root or is duplicated")
        seen.add(chunk)
        values, row_metadata = load_capture_chunk(chunk, shard["sha256"], manifest["fingerprint"])
        for metadata in row_metadata:
            key = (metadata["conv_id"], metadata["turn"])
            prefix = metadata["context_prefix_sha256"]
            if key in prefix_hashes and prefix_hashes[key] != prefix:
                raise ValueError("context token prefix differs across answer draws")
            prefix_hashes[key] = prefix
        for name in collected:
            collected[name].append(values[name])
    arrays = {name: np.concatenate(values, axis=0) for name, values in collected.items()}
    bank = validate_capture(arrays, expected_dim=expected_dim)
    if bank["rows"] != manifest["n_captured_draws"]:
        raise ValueError("capture summary row count mismatch")
    if manifest["n_captured_draws"] + manifest["n_excluded_draws"] != manifest["n_expected_draws"]:
        raise ValueError("capture planned-versus-realized counts do not reconcile")
    if (
        sorted(bank["complete"]) != sorted(manifest["complete_conversation_ids"])
        or len(bank["complete"]) != manifest["n_complete_conversations"]
    ):
        raise ValueError("capture summary complete-conversation mismatch")
    bank["reported_exclusions"] = manifest["exclusions"]
    return bank, {
        "manifest": str(manifest_path.resolve()),
        "sha256": sha256(manifest_path),
        "capture": manifest,
    }


def matched_panels(captures: dict[str, dict]) -> tuple[dict, dict]:
    """Intersect whole conversations across models, turns and all five draws."""
    if set(captures) != set(MODELS):
        raise ValueError("both model capture banks are required")
    ids = np.asarray(sorted(set.intersection(*(set(captures[m]["complete"]) for m in MODELS))))
    if len(ids) < 12:
        raise ValueError("fewer than 12 fully matched conversations")
    folds = fit825._cv_folds(ids, N_FOLDS, SEED)
    panels = {}
    exclusions = {}
    for model in MODELS:
        capture = captures[model]
        panels[model] = {"ids": ids, "folds": folds, "turns": {}}
        for turn in TURNS:
            rows = [capture["banks"][str(conv)][turn] for conv in ids]
            panels[model]["turns"][turn] = {
                "x": np.stack([r[0] for r in rows]).astype(np.float64),
                "y": np.stack([np.stack([r[1], r[2]]) for r in rows], axis=1),
            }
        exclusions[model] = {
            "incomplete": capture["excluded"],
            "complete_but_unmatched": sorted(set(capture["complete"]) - set(ids)),
            "captured_rows": capture["rows"],
            "capture_reported_exclusions": capture.get("reported_exclusions", []),
            "context_numeric_parity": capture["context_parity"],
        }
    return panels, {
        "n_conversations": len(ids),
        "conversation_ids": ids.tolist(),
        "outer_fold_ids": folds.tolist(),
        "exclusions": exclusions,
    }


def fit_paired_ridge(x: np.ndarray, y: np.ndarray, groups: np.ndarray, outer_fold: int) -> dict:
    """Fit K=1/K=5 with shared X caches and the existing batched inner selector."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups)
    if x.ndim != 2 or y.shape != (2, len(x), x.shape[1]) or len(groups) != len(x):
        raise ValueError("paired ridge inputs are misaligned")
    if (
        len(np.unique(groups)) != len(groups)
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("paired ridge requires unique groups and finite values")
    device = torch.device("cpu")
    inner = fit825._prep_inner_lambda(x, groups, N_INNER, INNER_SEED + outer_fold, device)
    if inner is None or len(inner) != N_INNER:
        raise RuntimeError("all four inner group-CV caches are required; no GCV fallback")
    yt = torch.as_tensor(y, dtype=torch.float64, device=device)
    curves = fit825._inner_cv_rss_curve_batched(inner, yt)
    if curves.shape != (2, 13) or not torch.isfinite(curves).all():
        raise RuntimeError("invalid inner group-CV RSS curves")
    selected = torch.as_tensor(fit825.LAMBDAS, dtype=torch.float64)[curves.argmin(1)]
    xt = torch.as_tensor(x, dtype=torch.float64, device=device)
    xmu, xsd, ymu = xt.mean(0), xt.std(0) + 1e-9, yt.mean(1)
    xn = (xt - xmu) / xsd
    w, v = torch.linalg.eigh(xn @ xn.T)
    w.clamp_(min=0)
    vy = torch.einsum("ij,kjd->kid", v.T, yt - ymu[:, None])
    dual = torch.einsum("ij,kjd->kid", v, vy / (w[None, :, None] + selected[:, None, None]))
    fitted = {
        "xtrain_standardized": xn.numpy(),
        "xmu": xmu.numpy(),
        "xsd": xsd.numpy(),
        "dual": dual.numpy(),
        "ymu": ymu.numpy(),
        "lambda": selected.numpy(),
        "lambda_grid": np.asarray(fit825.LAMBDAS),
        "inner_cv_rss": curves.numpy(),
        "train_ids": groups,
        "inner_fold_ids": fit825._cv_folds(groups, N_INNER, INNER_SEED + outer_fold),
    }
    for index, cache in enumerate(inner):
        fitted[f"inner_{index}_train_indices"] = cache["fi_idx"].numpy()
        fitted[f"inner_{index}_validation_indices"] = cache["va_idx"].numpy()
    return fitted


def predict_paired(fitted: dict, x: np.ndarray) -> np.ndarray:
    """Apply both K fits using their frozen train-normalized dual representation."""
    xn = (np.asarray(x, dtype=np.float64) - fitted["xmu"]) / fitted["xsd"]
    gram = xn @ fitted["xtrain_standardized"].T
    return np.einsum("ij,kjd->kid", gram, fitted["dual"]) + fitted["ymu"][:, None]


def heldout_predictions(panel: dict, fold: int, source: dict, own: dict) -> dict:
    """Calibrate exclusively on target-turn training rows and predict held-out rows."""
    ids, membership = panel["ids"], panel["folds"]
    train, test = np.flatnonzero(membership != fold), np.flatnonzero(membership == fold)
    if not len(test) or len(train) < 4 or len(set(ids[train]) & set(ids[test])):
        raise ValueError("invalid conversation train/test split")
    for fitted in (source, own):
        if not np.array_equal(fitted["train_ids"], ids[train]):
            raise ValueError("source/own map training conversations differ from outer fold")
    destination = panel["turns"][12]
    raw_all = predict_paired(source, destination["x"])
    own_test = predict_paired(own, destination["x"][test])
    predictions = np.empty((2, len(METHODS), len(test), destination["x"].shape[1]))
    coefficients = {name: [] for name in ("bias", "gain", "prediction_mean", "target_mean")}
    identity_bias_vectors = []
    for k in range(2):
        c = calibrate(raw_all[k, train], destination["y"][k, train])
        adjusted = adapted_predictions(raw_all[k, test], c)
        identity_bias = (destination["y"][k, train] - destination["x"][train]).mean(0)
        identity_bias_vectors.append(identity_bias)
        predictions[k] = np.stack(
            [
                raw_all[k, test],
                adjusted["bias"],
                adjusted["bias_scale"],
                destination["x"][test] + identity_bias,
                own_test[k],
            ]
        )
        for name in coefficients:
            coefficients[name].append(c[name])
    return {
        "test_ids": ids[test],
        "test_indices": test,
        "train_indices": train,
        "predictions": predictions,
        "target": destination["y"][:, test],
        "identity_bias": np.asarray(identity_bias_vectors),
        **{name: np.asarray(values) for name, values in coefficients.items()},
    }


def score_rows(predictions: np.ndarray, targets: np.ndarray) -> dict:
    """Cross-score train-K x eval-K on identical held-out candidate pools."""
    if predictions.ndim != 4 or predictions.shape[:2] != (2, len(METHODS)):
        raise ValueError("invalid prediction bank")
    if targets.shape != (2, *predictions.shape[2:]):
        raise ValueError("targets do not match prediction rows")
    # [train K, eval K, method, held-out conversation].
    residuals = predictions[:, None] - targets[None, :, None]
    sse = np.einsum("kemnd,kemnd->kemn", residuals, residuals)
    centered = targets - targets.mean(axis=1, keepdims=True)
    sst = np.einsum("end,end->en", centered, centered)
    pnorm, tnorm = np.linalg.norm(predictions, axis=-1), np.linalg.norm(targets, axis=-1)
    if np.any(pnorm == 0) or np.any(tnorm == 0):
        raise ValueError("zero-norm vectors make cosine retrieval undefined")
    dot = np.einsum("kmnd,evd->kemnv", predictions, targets)
    cosine = dot / (pnorm[:, None, :, :, None] * tnorm[None, :, None, None, :])
    distances = pnorm[:, None, :, :, None] ** 2 + tnorm[None, :, None, None, :] ** 2 - 2 * dot
    rows = np.arange(predictions.shape[2])
    return {
        "row_sse": sse,
        "row_sst": sst,
        "cosine_top1_hit": (cosine.argmax(-1) == rows).astype(np.uint8),
        "euclidean_top1_hit": (distances.argmin(-1) == rows).astype(np.uint8),
        "pool_size": np.full(len(rows), len(rows), dtype=np.int64),
    }


def calibration_comparisons(point: dict, boot: dict) -> list[dict]:
    """Use the same paired bootstrap draws for changes attributable to calibration."""
    comparisons = []
    for k, train_k in enumerate(KS):
        for e, eval_k in enumerate(KS):
            for label, left, right in (
                ("bias_minus_raw", 1, 0),
                ("bias_scale_minus_bias", 2, 1),
                ("bias_scale_minus_raw", 2, 0),
            ):
                comparison = {"comparison": label, "train_k": train_k, "eval_k": eval_k}
                for metric in boot:
                    delta = boot[metric][:, k, e, left] - boot[metric][:, k, e, right]
                    comparison[metric] = float(
                        point[metric][k, e, left] - point[metric][k, e, right]
                    )
                    comparison[f"{metric}_ci95"] = np.quantile(delta, [0.025, 0.975]).tolist()
                comparisons.append(comparison)
    return comparisons


def bootstrap_summary(rows: dict, *, replicates: int = BOOTSTRAPS) -> dict:
    """Pair conversations across every arm; condition intervals on maps/draw bank."""
    sse, sst = rows["row_sse"], rows["row_sst"]
    n = sse.shape[-1]
    if sse.shape[:3] != (2, 2, len(METHODS)) or sst.shape != (2, n):
        raise ValueError("invalid OOF rows")
    if replicates < 100:
        raise ValueError("bootstrap requires at least 100 replicates")
    weights = np.random.default_rng(SEED).multinomial(n, np.full(n, 1 / n), size=replicates)
    denom = weights @ sst.T
    if np.any(denom <= 0) or np.any(sst.sum(-1) <= 0):
        raise ValueError("nonpositive held-out target variance")
    br2 = 1 - np.einsum("bn,kemn->bkem", weights, sse) / denom[:, None, :, None]
    r2 = 1 - sse.sum(-1) / sst.sum(-1)[None, :, None]
    boot = {"r2": br2}
    point = {"r2": r2}
    for metric in ("cosine_top1", "euclidean_top1"):
        hits = rows[f"{metric}_hit"]
        boot[metric] = np.einsum("bn,kemn->bkem", weights, hits) / n
        point[metric] = hits.mean(-1)
    own = METHODS.index("own12")
    cells = []
    for k, train_k in enumerate(KS):
        for e, eval_k in enumerate(KS):
            own_interval = np.quantile(br2[:, k, e, own], [0.025, 0.975])
            valid_ratio = bool(r2[k, e, own] > 0 and own_interval[0] > 0)
            for m, method in enumerate(METHODS):
                cell = {"train_k": train_k, "eval_k": eval_k, "method": method}
                for metric in boot:
                    cell[metric] = float(point[metric][k, e, m])
                    cell[f"{metric}_ci95"] = np.quantile(
                        boot[metric][:, k, e, m], [0.025, 0.975]
                    ).tolist()
                cell["r2_retention"] = float(r2[k, e, m] / r2[k, e, own]) if valid_ratio else None
                cell["r2_retention_ci95"] = (
                    np.quantile(br2[:, k, e, m] / br2[:, k, e, own], [0.025, 0.975]).tolist()
                    if valid_ratio
                    else None
                )
                cell["ratio_status"] = (
                    "valid" if valid_ratio else "own_r2_not_positive_away_from_zero"
                )
                cells.append(cell)
    comparisons = []
    for m, method in enumerate(METHODS):
        for label, left, right in (
            ("diagonal_k5_minus_k1", (1, 1), (0, 0)),
            ("training_k5_minus_k1_evaluated_k1", (1, 0), (0, 0)),
            ("training_k5_minus_k1_evaluated_k5", (1, 1), (0, 1)),
            ("target_k5_minus_k1_trained_k1", (0, 1), (0, 0)),
            ("target_k5_minus_k1_trained_k5", (1, 1), (1, 0)),
        ):
            comparison = {"comparison": label, "method": method}
            for metric in boot:
                difference = boot[metric][:, *left, m] - boot[metric][:, *right, m]
                comparison[metric] = float(point[metric][*left, m] - point[metric][*right, m])
                comparison[f"{metric}_ci95"] = np.quantile(difference, [0.025, 0.975]).tolist()
            comparisons.append(comparison)
    return {
        "cells": cells,
        "paired_k_comparisons": comparisons,
        "paired_calibration_comparisons": calibration_comparisons(point, boot),
        "n_conversations": n,
        "retrieval_pool_sizes": sorted(set(rows["pool_size"].tolist())),
        "retrieval_chance": float(np.mean(1 / rows["pool_size"])),
        "bootstrap": {
            "replicates": replicates,
            "seed": SEED,
            "unit": "conversation",
            "paired_across_arms": True,
            "conditional_on": ["fitted maps", "captured answer draw bank"],
            "sst_center": "fixed original held-out-fold target mean",
            "refits": False,
        },
    }


def run_analysis(
    panels: dict,
    coverage: dict,
    provenance: dict,
    out_dir: Path,
    store_dir: Path,
    *,
    bootstraps: int = BOOTSTRAPS,
) -> dict:
    """Checkpoint each model/fold, validate resumes, then reduce paired OOF rows."""
    source_files = [Path(__file__), Path(fit825.__file__)]
    import explore_persona_space.analysis.turn_transfer_calibration as calibration_module

    source_files.append(Path(calibration_module.__file__))
    recipe = RECIPE | {"bootstraps": bootstraps}
    identity = {
        "recipe": recipe,
        "coverage": coverage,
        "provenance": provenance,
        "source_sha256": {str(path): sha256(path) for path in source_files},
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    identity_path = out_dir / "inputs_and_recipe.json"
    if (
        identity_path.exists()
        and json.loads(identity_path.read_text())["fingerprint"] != fingerprint
    ):
        raise RuntimeError("output directory belongs to different inputs or recipe")
    write_json(identity_path, identity | {"fingerprint": fingerprint})
    model_summaries = {}
    artifacts = []
    for model in MODELS:
        panel = panels[model]
        assembled = {
            name: []
            for name in (
                "row_sse",
                "row_sst",
                "cosine_top1_hit",
                "euclidean_top1_hit",
                "pool_size",
                "test_indices",
            )
        }
        for fold in range(N_FOLDS):
            began = time.monotonic()
            receipt_path = out_dir / "folds" / f"{model}_fold{fold}.json"
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if receipt["fingerprint"] != fingerprint:
                    raise RuntimeError(f"stale fold checkpoint: {receipt_path}")
                for item in receipt["artifacts"]:
                    if sha256(Path(item["path"])) != item["sha256"]:
                        raise RuntimeError(f"corrupt checkpoint artifact: {item['path']}")
                prediction_path = Path(receipt["predictions"]["path"])
                print(f"[fit] resume {model}/fold{fold}", flush=True)
            else:
                train = panel["folds"] != fold
                maps, map_receipts = {}, []
                for turn in TURNS:
                    bank = panel["turns"][turn]
                    maps[turn] = fit_paired_ridge(
                        bank["x"][train], bank["y"][:, train], panel["ids"][train], fold
                    )
                    map_path = store_dir / "maps" / f"{model}_turn{turn}_fold{fold}.npz"
                    map_receipts.append(write_npz(map_path, **maps[turn]))
                predictions = heldout_predictions(panel, fold, maps[1], maps[12])
                scored = score_rows(predictions["predictions"], predictions["target"])
                prediction_path = store_dir / "predictions" / f"{model}_fold{fold}.npz"
                predicted = write_npz(prediction_path, **predictions, **scored)
                receipt = {
                    "fingerprint": fingerprint,
                    "model": model,
                    "fold": fold,
                    "artifacts": [*map_receipts, predicted],
                    "predictions": predicted,
                    "elapsed_seconds": time.monotonic() - began,
                    "selected_lambdas": {str(t): maps[t]["lambda"].tolist() for t in TURNS},
                }
                write_json(receipt_path, receipt)
                del maps, predictions, scored
                print(
                    f"[fit] unit {MODELS.index(model) * N_FOLDS + fold + 1}/12 "
                    f"{model}/fold{fold} elapsed={time.monotonic() - began:.1f}s",
                    flush=True,
                )
            artifacts.extend(receipt["artifacts"])
            with np.load(prediction_path, allow_pickle=False) as saved:
                for name in assembled:
                    assembled[name].append(saved[name])
        oof = {name: np.concatenate(values, axis=-1) for name, values in assembled.items()}
        order = np.argsort(oof["test_indices"])
        if not np.array_equal(oof["test_indices"][order], np.arange(len(panel["ids"]))):
            raise RuntimeError("OOF coverage is not exactly one prediction per conversation")
        oof = {name: array[..., order] for name, array in oof.items()}
        oof["conv_id"] = panel["ids"]
        oof["fold_id"] = panel["folds"]
        artifacts.append(write_npz(store_dir / f"{model}_oof_rows.npz", **oof))
        model_summaries[model] = bootstrap_summary(oof, replicates=bootstraps)
        write_json(out_dir / f"{model}_results.json", model_summaries[model])
    result = {
        "status": "complete",
        "fingerprint": fingerprint,
        "recipe": recipe,
        "coverage": coverage,
        "models": model_summaries,
        "artifacts": artifacts,
        "fit_recipe_note": "corrected inner-group CV; not the prior legacy-GCV replay",
        "interpretation": "Target-informed affine calibration; not zero-shot transfer.",
    }
    write_json(out_dir / "results.json", result)
    return result


def main() -> None:
    """Load verified captures and execute the fixed, CPU-only endpoint analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruct-root", required=True, type=Path)
    parser.add_argument("--pretrained-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--store-dir", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    captures, provenance = {}, {}
    for model in MODELS:
        captures[model], provenance[model] = load_capture(
            getattr(args, f"{model}_root"), expected_model=model
        )
    panel_hashes = {
        p["capture"]["generation_config"]["selected_ids_sha256"] for p in provenance.values()
    }
    if len(panel_hashes) != 1:
        raise ValueError("model runs did not use the same planned conversation panel")
    panels, coverage = matched_panels(captures)
    run_analysis(panels, coverage, provenance, args.out_dir, args.store_dir)


if __name__ == "__main__":
    main()
