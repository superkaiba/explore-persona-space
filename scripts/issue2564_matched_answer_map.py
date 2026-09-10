"""Grouped out-of-fold context-to-answer map for the #2564 span vectors.

This is an in-bank matched transfer test, independent of the frozen #779 map
(whose physical layer convention is incompatible with the #2054 vectors).
It uses the established #2054 ambient GCV-ridge helper, with each map trained
on the other four outer folds and evaluated only on its held-out fold.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from explore_persona_space.atomic_io import write_json_atomic  # noqa: E402
import issue2054_fits as established  # noqa: E402
from issue2054_fits import (  # noqa: E402
    DEFAULT_LAMBDAS,
    _ridge_gcv_fit_predict,
)


def _sha256_array(value: np.ndarray) -> str:
    """Hash shape, dtype and contiguous bytes so equal bytes cannot hide shape drift."""
    arr = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(repr(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_inputs(rows: list[dict], context: np.ndarray, answer: np.ndarray) -> list[int]:
    if context.ndim != 2 or answer.ndim != 2:
        raise ValueError("context and answer must be two-dimensional matrices")
    if context.shape != answer.shape:
        raise ValueError(f"context/answer shape mismatch: {context.shape} != {answer.shape}")
    if len(rows) != context.shape[0] or context.shape[1] != 3584:
        raise ValueError("expected one row per 3584-wide #2564 vector")
    if not np.isfinite(context).all() or not np.isfinite(answer).all():
        raise ValueError("context and answer must be finite")
    if not rows:
        raise ValueError("empty matched map input")
    fold_of_group: dict[str, int] = {}
    for row in rows:
        if "id" not in row or "fold" not in row or "question_group" not in row:
            raise ValueError("each row needs id, fold and question_group")
        fold = int(row["fold"])
        group = str(row["question_group"])
        prior = fold_of_group.setdefault(group, fold)
        if prior != fold:
            raise ValueError(f"question_group crosses outer folds: {group}")
    folds = sorted({int(row["fold"]) for row in rows})
    if folds != [0, 1, 2, 3, 4]:
        raise ValueError(f"expected exactly outer folds 0..4, got {folds}")
    if len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("row IDs must be unique")
    return folds


def fit_grouped_oof(
    rows: list[dict],
    context: np.ndarray,
    answer: np.ndarray,
    output_dir: str | Path,
    *,
    lambdas: np.ndarray = DEFAULT_LAMBDAS,
    dof_cap: float = 0.9,
) -> dict:
    """Fit five train-fold-only maps and return held-out predictions.

    The selector is the exact ``issue2054_fits._ridge_gcv_fit_predict``
    standardize-X/center-Y GCV path.  The identity-plus-bias arm uses the
    same training rows and is evaluated on the same held-out rows.  No labels
    or behavior targets enter this function.
    """
    folds = _validate_inputs(rows, context, answer)
    if not (0 < dof_cap <= 1):
        raise ValueError("dof_cap must be in (0, 1]")
    lambdas = np.asarray(lambdas, dtype=np.float64)
    if (
        lambdas.ndim != 1
        or len(lambdas) == 0
        or not np.isfinite(lambdas).all()
        or (lambdas <= 0).any()
    ):
        raise ValueError("lambdas must be a finite nonempty positive vector")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    n, width = context.shape
    ids = [str(row["id"]) for row in rows]
    fold_ids = {
        str(fold): [ids[i] for i, row in enumerate(rows) if int(row["fold"]) == fold]
        for fold in folds
    }
    regime = {
        "recipe": "issue2054_fits._ridge_gcv_fit_predict",
        "standardization": "train-fold context mean/std (population std + 1e-9); answer train mean",
        "dof_cap": float(dof_cap),
        "lambdas": lambdas.tolist(),
        "n_rows": n,
        "width": width,
        "rows_sha256": _sha256_array(np.asarray(ids)),
        "context_sha256": _sha256_array(context),
        "answer_sha256": _sha256_array(answer),
        "fold_ids": fold_ids,
        "helper_sha256": _sha256_file(Path(__file__).resolve()),
        "producer_sha256": _sha256_file(Path(established.__file__).resolve()),
    }
    regime_fingerprint = hashlib.sha256(json.dumps(regime, sort_keys=True).encode()).hexdigest()
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("regime_fingerprint") != regime_fingerprint:
            raise ValueError("output directory belongs to another matched-map regime")

    oof_map = np.full((n, width), np.nan, dtype=np.float64)
    oof_identity = np.full((n, width), np.nan, dtype=np.float64)
    fold_meta_by_id: dict[int, dict] = {}
    started = time.monotonic()
    for fold in folds:
        fold_started = time.monotonic()
        test = np.flatnonzero(np.array([int(row["fold"]) == fold for row in rows]))
        train = np.flatnonzero(np.array([int(row["fold"]) != fold for row in rows]))
        train_groups = {str(rows[i]["question_group"]) for i in train}
        test_groups = {str(rows[i]["question_group"]) for i in test}
        if train_groups & test_groups:
            raise ValueError(f"group leakage in fold {fold}")
        fold_json = out / f"fold{fold}.json"
        fold_npz = out / f"fold{fold}.npz"
        if fold_json.exists() or fold_npz.exists():
            if not (fold_json.exists() and fold_npz.exists()):
                raise ValueError(f"incomplete fold checkpoint for fold {fold}")
            evidence = json.loads(fold_json.read_text())
            if evidence.get("regime_fingerprint") != regime_fingerprint:
                raise ValueError(f"fold {fold} checkpoint regime mismatch")
            if evidence.get("train_ids") != [ids[i] for i in train] or evidence.get("test_ids") != [
                ids[i] for i in test
            ]:
                raise ValueError(f"fold {fold} checkpoint IDs mismatch")
            with np.load(fold_npz, allow_pickle=False) as saved:
                if set(saved.files) != {"test_indices", "mapped", "identity_bias"}:
                    raise ValueError(f"fold {fold} checkpoint schema mismatch")
                saved_test = saved["test_indices"]
                mapped = saved["mapped"]
                identity = saved["identity_bias"]
            if (
                not np.array_equal(saved_test, test)
                or mapped.shape != (len(test), width)
                or identity.shape != mapped.shape
            ):
                raise ValueError(f"fold {fold} checkpoint shape/index mismatch")
            if not np.isfinite(mapped).all() or not np.isfinite(identity).all():
                raise ValueError(f"fold {fold} checkpoint contains nonfinite predictions")
            if evidence.get("mapped_test_sha256") != _sha256_array(mapped) or evidence.get(
                "identity_bias_test_sha256"
            ) != _sha256_array(identity):
                raise ValueError(f"fold {fold} checkpoint prediction hash mismatch")
            fit_info = evidence["fit"]
            if fit_info.get("dof_over_cap"):
                raise ValueError(f"fold {fold} checkpoint used a degenerate GCV selection")
            fold_meta_by_id[fold] = evidence
            oof_map[test] = mapped
            oof_identity[test] = identity
            print(f"[matched-map] fold {fold + 1}/5 resumed", flush=True)
            continue

        mapped, fit_info = _ridge_gcv_fit_predict(
            context[train], answer[train], context[test], lambdas=lambdas, dof_cap=dof_cap
        )
        if fit_info.get("dof_over_cap"):
            raise RuntimeError(f"fold {fold} has no nondegenerate GCV lambda under dof cap")
        bias = answer[train].astype(np.float64).mean(axis=0) - context[train].astype(
            np.float64
        ).mean(axis=0)
        identity = context[test].astype(np.float64) + bias
        oof_map[test] = mapped
        oof_identity[test] = identity
        np.savez_compressed(fold_npz, test_indices=test, mapped=mapped, identity_bias=identity)
        evidence = {
            "fold": fold,
            "regime_fingerprint": regime_fingerprint,
            "elapsed_s": time.monotonic() - fold_started,
            "train_ids": [ids[i] for i in train],
            "test_ids": [ids[i] for i in test],
            "train_groups": sorted(train_groups),
            "test_groups": sorted(test_groups),
            "fit": fit_info,
            "lambdas": lambdas.tolist(),
            "context_train_sha256": _sha256_array(context[train]),
            "answer_train_sha256": _sha256_array(answer[train]),
            "context_test_sha256": _sha256_array(context[test]),
            "answer_test_sha256": _sha256_array(answer[test]),
            "mapped_test_sha256": _sha256_array(mapped),
            "identity_bias_test_sha256": _sha256_array(identity),
            "model_fingerprint": hashlib.sha256(
                json.dumps(
                    {
                        "fit": fit_info,
                        "context_train_sha256": _sha256_array(context[train]),
                        "answer_train_sha256": _sha256_array(answer[train]),
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest(),
        }
        write_json_atomic(fold_json, evidence)
        fold_meta_by_id[fold] = evidence
        running = {
            "status": "running",
            "regime": regime,
            "regime_fingerprint": regime_fingerprint,
            "completed_folds": sorted(fold_meta_by_id),
            "folds_meta": [fold_meta_by_id[f] for f in sorted(fold_meta_by_id)],
            "elapsed_s": time.monotonic() - started,
        }
        write_json_atomic(manifest_path, running)
        print(
            f"[matched-map] fold {fold + 1}/5 complete elapsed={evidence['elapsed_s']:.2f}s",
            flush=True,
        )

    final = out / "oof_predictions.npz"
    np.savez_compressed(final, id=np.asarray(ids), mapped=oof_map, identity_bias=oof_identity)
    fold_meta = [fold_meta_by_id[f] for f in folds]
    manifest = {
        "status": "complete",
        "regime": regime,
        "regime_fingerprint": regime_fingerprint,
        "folds": folds,
        "completed_folds": folds,
        "oof_prediction_sha256": _sha256_array(oof_map),
        "oof_identity_bias_sha256": _sha256_array(oof_identity),
        "folds_meta": fold_meta,
    }
    write_json_atomic(out / "manifest.json", manifest)
    return {
        "mapped": oof_map,
        "identity_bias": oof_identity,
        "folds": fold_meta,
        "manifest": manifest,
    }


__all__ = ["fit_grouped_oof"]
