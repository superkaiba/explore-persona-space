#!/usr/bin/env python3
"""Factorial diff of cross-model context-to-answer maps for issue #2569.

The crossed own-answer follow-up fits four native maps at Qwen L14 / Llama L16:

    encoder Qwen x writer Qwen, encoder Qwen x writer Llama,
    encoder Llama x writer Qwen, encoder Llama x writer Llama.

This script puts all four affine maps in one fixed Qwen coordinate system and
decomposes their natural diagonal difference into encoder and writer contrasts,
plus the encoder-by-writer interaction.  All reported prediction reads are on
the frozen held-out rows; the alignments and behavior readout use train rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

from explore_persona_space.atomic_io import atomic_replace  # noqa: E402

import issue2569_atlas as AT  # noqa: E402
import issue2569_operator as OP  # noqa: E402


SOURCE_REVISION = "8d2694f6eedfbad61b9413299bca096370429d7a"
PRIMARY_LAYERS = {"qwen": 14, "llama": 16}
CELL_NAMES = ("q_qwriter", "q_lwriter", "l_qwriter", "l_lwriter")
CONTRAST_NAMES = ("writer", "encoder", "interaction", "diagonal")
REFUSAL_RE = re.compile(
    r"\b(i (?:can(?:not|'t)|won't)|unable to|cannot assist|sorry,? but|as an ai)\b",
    re.I,
)
REPEAT_RE = re.compile(r"(.{1,40})\1{4,}", re.S)


@dataclass(frozen=True)
class Affine:
    A: np.ndarray
    b: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64) @ self.A + self.b


@dataclass(frozen=True)
class FixedAlignment:
    R_context: np.ndarray
    R_answer: np.ndarray
    q_context_mean: np.ndarray
    l_context_mean: np.ndarray
    q_answer_mean: np.ndarray
    l_answer_mean: np.ndarray

    def q_context_to_l(self, x: np.ndarray) -> np.ndarray:
        return (
            np.asarray(x, dtype=np.float64) - self.q_context_mean
        ) @ self.R_context + self.l_context_mean

    def l_answer_to_q(self, y: np.ndarray) -> np.ndarray:
        return (
            np.asarray(y, dtype=np.float64) - self.l_answer_mean
        ) @ self.R_answer.T + self.q_answer_mean


def atomic_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_replace(path) as tmp:
        Path(tmp).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_replace(path) as tmp, open(tmp, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_replace(path) as tmp:
        with open(tmp, "wb") as handle:
            np.savez_compressed(handle, **arrays)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha_int64(values: np.ndarray | list[int]) -> str:
    arr = np.ascontiguousarray(np.asarray(values, dtype=np.int64))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def load_bundle(path: Path, roster: np.ndarray) -> np.ndarray:
    bundle = AT._decode_bundle(path)
    pos = {int(ci): i for i, ci in enumerate(bundle["ci"])}
    missing = [int(ci) for ci in roster if int(ci) not in pos]
    if missing:
        raise RuntimeError(f"{path}: missing {len(missing)} requested rows")
    idx = np.asarray([pos[int(ci)] for ci in roster], dtype=np.int64)
    out = np.asarray(bundle["x"])[idx]
    if not np.isfinite(out).all():
        raise RuntimeError(f"{path}: non-finite activation")
    return out


def fold_indices(split: dict[str, Any]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    roster = np.asarray(split["ci"], dtype=np.int64)
    pos = {int(ci): i for i, ci in enumerate(roster)}
    folds = {
        key: np.asarray([pos[int(ci)] for ci in split[f"{name}_ci"]], dtype=np.int64)
        for key, name in (("tr", "train"), ("va", "val"), ("te", "test"))
    }
    joined = np.concatenate(list(folds.values()))
    if len(np.unique(joined)) != len(roster) or set(joined) != set(range(len(roster))):
        raise RuntimeError("split does not partition the exact roster")
    return roster, folds


def load_payload(path: Path) -> tuple[OP.MapPayload, dict[str, Any]]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return AT.payload_from_dict(obj, path=path), dict(obj["record"])


def make_fixed_alignment(
    q_context: np.ndarray,
    l_context: np.ndarray,
    q_qwriter: np.ndarray,
    l_qwriter: np.ndarray,
    q_lwriter: np.ndarray,
    l_lwriter: np.ndarray,
    tr: np.ndarray,
    *,
    device: str,
) -> FixedAlignment:
    q_answer = np.concatenate([q_qwriter[tr], q_lwriter[tr]], axis=0)
    l_answer = np.concatenate([l_qwriter[tr], l_lwriter[tr]], axis=0)
    return FixedAlignment(
        R_context=AT.orth_procrustes(q_context[tr], l_context[tr], device=device),
        R_answer=AT.orth_procrustes(q_answer, l_answer, device=device),
        q_context_mean=np.asarray(q_context[tr], np.float64).mean(0),
        l_context_mean=np.asarray(l_context[tr], np.float64).mean(0),
        q_answer_mean=np.asarray(q_answer, np.float64).mean(0),
        l_answer_mean=np.asarray(l_answer, np.float64).mean(0),
    )


def payload_affine(payload: OP.MapPayload) -> Affine:
    A, b = OP.row_operator(payload)
    return Affine(np.asarray(A, np.float64), np.asarray(b, np.float64))


def transform_l_affine(payload: OP.MapPayload, alignment: FixedAlignment) -> Affine:
    """Express a Llama native affine map in the fixed Qwen input/output basis."""
    native = payload_affine(payload)
    t_context = alignment.l_context_mean - alignment.q_context_mean @ alignment.R_context
    t_answer = alignment.q_answer_mean - alignment.l_answer_mean @ alignment.R_answer.T
    A = alignment.R_context @ native.A @ alignment.R_answer.T
    b = (t_context @ native.A + native.b) @ alignment.R_answer.T + t_answer
    return Affine(A=A, b=b)


def combine_affines(terms: list[tuple[float, Affine]]) -> Affine:
    return Affine(
        A=sum(weight * cell.A for weight, cell in terms),
        b=sum(weight * cell.b for weight, cell in terms),
    )


def factorial_affines(cells: dict[str, Affine]) -> dict[str, Affine]:
    q_q, q_l, l_q, l_l = (cells[name] for name in CELL_NAMES)
    out = {
        "writer": combine_affines(
            [(0.5, q_q), (-0.5, q_l), (0.5, l_q), (-0.5, l_l)]
        ),
        "encoder": combine_affines(
            [(0.5, q_q), (0.5, q_l), (-0.5, l_q), (-0.5, l_l)]
        ),
        "interaction": combine_affines(
            [(0.5, q_q), (-0.5, q_l), (-0.5, l_q), (0.5, l_l)]
        ),
        "diagonal": combine_affines([(1.0, q_q), (-1.0, l_l)]),
    }
    if not np.allclose(out["diagonal"].A, out["writer"].A + out["encoder"].A):
        raise AssertionError("factorial operator algebra failed")
    if not np.allclose(out["diagonal"].b, out["writer"].b + out["encoder"].b):
        raise AssertionError("factorial bias algebra failed")
    return out


def factorial_arrays(cells: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    q_q, q_l, l_q, l_l = (np.asarray(cells[name], np.float64) for name in CELL_NAMES)
    out = {
        "writer": 0.5 * (q_q - q_l + l_q - l_l),
        "encoder": 0.5 * (q_q + q_l - l_q - l_l),
        "interaction": 0.5 * (q_q - q_l - l_q + l_l),
        "diagonal": q_q - l_l,
    }
    if not np.allclose(out["diagonal"], out["writer"] + out["encoder"]):
        raise AssertionError("factorial prediction algebra failed")
    return out


def row_cosine(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    p = np.asarray(pred, np.float64)
    t = np.asarray(truth, np.float64)
    return np.sum(p * t, axis=1) / (
        np.linalg.norm(p, axis=1) * np.linalg.norm(t, axis=1) + 1e-30
    )


def flat_cosine(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, np.float64).reshape(-1)
    y = np.asarray(b, np.float64).reshape(-1)
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-30))


def bootstrap_median_ci(values: np.ndarray, draws: int, seed: int) -> list[float]:
    x = np.asarray(values, np.float64)
    rng = np.random.default_rng(seed)
    stats = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        stats[draw] = np.median(x[rng.integers(0, len(x), size=len(x))])
    return [float(v) for v in np.quantile(stats, [0.025, 0.975])]


def permutation_null(
    pred: np.ndarray,
    truth: np.ndarray,
    *,
    draws: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    """Row-pairing null for prompt-specific contrast prediction."""
    p = np.asarray(pred, np.float64)
    t = np.asarray(truth, np.float64)
    dev = torch.device(device)
    cross = (
        torch.as_tensor(p, dtype=torch.float64, device=dev)
        @ torch.as_tensor(t, dtype=torch.float64, device=dev).T
    ).cpu().numpy()
    p2 = float(np.sum(p * p))
    t2 = float(np.sum(t * t))
    denom_cos = math.sqrt(p2 * t2) + 1e-30
    centered = t - t.mean(0)
    ss_tot = float(np.sum(centered * centered))
    rng = np.random.default_rng(seed)
    null_cos = np.empty(draws, dtype=np.float64)
    null_r2 = np.empty(draws, dtype=np.float64)
    rows = np.arange(len(p))
    for draw in range(draws):
        perm = rng.permutation(len(p))
        dot = float(cross[rows, perm].sum())
        null_cos[draw] = dot / denom_cos
        null_r2[draw] = 1.0 - (p2 + t2 - 2.0 * dot) / max(ss_tot, 1e-30)
    observed_cos = flat_cosine(p, t)
    observed_r2 = AT.pooled_r2(p, t)
    return {
        "kind": "held-out row-pairing permutation; prompt correspondence destroyed",
        "n_draws": int(draws),
        "flat_cosine": {
            "observed": observed_cos,
            "null_mean": float(null_cos.mean()),
            "null_p025_p975": [float(v) for v in np.quantile(null_cos, [0.025, 0.975])],
            "p_ge": float((1 + np.sum(null_cos >= observed_cos)) / (draws + 1)),
        },
        "pooled_r2": {
            "observed": observed_r2,
            "null_mean": float(null_r2.mean()),
            "null_p025_p975": [float(v) for v in np.quantile(null_r2, [0.025, 0.975])],
            "p_ge": float((1 + np.sum(null_r2 >= observed_r2)) / (draws + 1)),
        },
    }


def matrix_metrics(
    pred: np.ndarray,
    truth: np.ndarray,
    *,
    permutation_draws: int,
    bootstrap_draws: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    p = np.asarray(pred, np.float64)
    t = np.asarray(truth, np.float64)
    rc = row_cosine(p, t)
    return {
        "n": int(len(p)),
        "d": int(p.shape[1]),
        "pooled_r2": AT.pooled_r2(p, t),
        "flat_cosine": flat_cosine(p, t),
        "relative_l2": float(np.linalg.norm(p - t) / (np.linalg.norm(t) + 1e-30)),
        "prediction_over_truth_energy": float(
            np.sum(p * p) / (np.sum(t * t) + 1e-30)
        ),
        "row_cosine": {
            "mean": float(np.mean(rc)),
            "median": float(np.median(rc)),
            "q05_q25_q75_q95": [
                float(v) for v in np.quantile(rc, [0.05, 0.25, 0.75, 0.95])
            ],
            "median_bootstrap_ci95": bootstrap_median_ci(
                rc, bootstrap_draws, seed + 100_000
            ),
        },
        "permutation_null": permutation_null(
            p, t, draws=permutation_draws, seed=seed, device=device
        ),
    }


def safe_corr(x: np.ndarray, y: np.ndarray, kind: str) -> dict[str, Any]:
    a = np.asarray(x, np.float64)
    b = np.asarray(y, np.float64)
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return {"value": None, "p": None, "reason": "constant or too few rows"}
    result = pearsonr(a, b) if kind == "pearson" else spearmanr(a, b)
    return {"value": float(result.statistic), "p": float(result.pvalue)}


def scalar_metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    p = np.asarray(pred, np.float64)
    t = np.asarray(truth, np.float64)
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    ss_res = float(np.sum((t - p) ** 2))
    return {
        "n": int(len(p)),
        "r2": None if ss_tot < 1e-12 else float(1.0 - ss_res / ss_tot),
        "pearson": safe_corr(p, t, "pearson"),
        "spearman": safe_corr(p, t, "spearman"),
        "rmse": float(np.sqrt(np.mean((p - t) ** 2))),
    }


def behavior_arrays(rows: list[dict[str, Any]], roster: np.ndarray) -> dict[str, np.ndarray]:
    by_ci = {int(row["ci"]): row for row in rows}
    missing = [int(ci) for ci in roster if int(ci) not in by_ci]
    if missing:
        raise RuntimeError(f"behavior rows missing {len(missing)} roster items")
    ordered = [by_ci[int(ci)] for ci in roster]
    return {
        "log_length_delta": np.asarray(
            [math.log1p(row["qwen_words"]) - math.log1p(row["llama_words"]) for row in ordered],
            np.float64,
        ),
        "refusal_delta": np.asarray(
            [float(row["qwen_refusal_flag"]) - float(row["llama_refusal_flag"]) for row in ordered],
            np.float64,
        ),
        "repetition_delta": np.asarray(
            [float(row["qwen_repetition_flag"]) - float(row["llama_repetition_flag"]) for row in ordered],
            np.float64,
        ),
        "semantic_divergence": np.asarray(
            [1.0 - float(row["embedding_cosine"]) for row in ordered], np.float64
        ),
    }


def load_completion_rows(root: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    for path in sorted(root.rglob("shard*.json")):
        obj = json.loads(path.read_text())
        for row in obj.get("rows", []):
            if row.get("drop_reason") is None and str(row.get("response", "")).strip():
                out[int(row["ci"])] = str(row["response"])
    return out


def reliability_behavior_arrays(
    q_root: Path,
    l_root: Path,
    semantic_rows_path: Path,
    roster: np.ndarray,
) -> dict[str, np.ndarray]:
    q = load_completion_rows(q_root)
    l = load_completion_rows(l_root)
    sem = {int(row["ci"]): row for row in read_jsonl(semantic_rows_path)}
    missing = [int(ci) for ci in roster if int(ci) not in q or int(ci) not in l or int(ci) not in sem]
    if missing:
        raise RuntimeError(f"seed-137 behavior inputs missing {len(missing)} test rows")
    qt = [q[int(ci)] for ci in roster]
    lt = [l[int(ci)] for ci in roster]
    return {
        "log_length_delta": np.asarray(
            [math.log1p(len(a.split())) - math.log1p(len(b.split())) for a, b in zip(qt, lt, strict=True)],
            np.float64,
        ),
        "refusal_delta": np.asarray(
            [float(bool(REFUSAL_RE.search(a))) - float(bool(REFUSAL_RE.search(b))) for a, b in zip(qt, lt, strict=True)],
            np.float64,
        ),
        "repetition_delta": np.asarray(
            [float(bool(REPEAT_RE.search(a))) - float(bool(REPEAT_RE.search(b))) for a, b in zip(qt, lt, strict=True)],
            np.float64,
        ),
        "semantic_divergence": np.asarray(
            [1.0 - float(sem[int(ci)]["qwen_vs_llama_seed137_cosine"]) for ci in roster],
            np.float64,
        ),
    }


def fit_behavior_readout(
    observed_writer: np.ndarray,
    predicted_writer: np.ndarray,
    behavior: dict[str, np.ndarray],
    folds: dict[str, np.ndarray],
    *,
    device: str,
    seed137_predicted_writer: np.ndarray | None,
    seed137_behavior: dict[str, np.ndarray] | None,
) -> tuple[dict[str, Any], OP.MapPayload]:
    names = list(behavior)
    raw = np.column_stack([behavior[name] for name in names])
    tr, te = folds["tr"], folds["te"]
    mu = raw[tr].mean(0)
    sd = raw[tr].std(0) + 1e-9
    target = (raw - mu) / sd
    fit = AT._fit_map(
        "q14_l16_writer_contrast_to_behavior",
        observed_writer,
        target,
        folds,
        torch.device(device),
        payload_device=device,
    )
    payload = fit["payload"]
    oracle_z = np.asarray(fit["pred_te"], np.float64)
    mediated_z = OP.predict(payload, predicted_writer[te])
    oracle = oracle_z * sd + mu
    mediated = mediated_z * sd + mu
    out: dict[str, Any] = {
        "axes": names,
        "target_train_mean": {name: float(mu[i]) for i, name in enumerate(names)},
        "target_train_sd": {name: float(sd[i]) for i, name in enumerate(names)},
        "fit": fit["record"],
        "heldout": {},
    }
    for i, name in enumerate(names):
        out["heldout"][name] = {
            "observed_activation_readout": scalar_metrics(oracle[:, i], raw[te, i]),
            "mapping_mediated": scalar_metrics(mediated[:, i], raw[te, i]),
            "target_counts": {
                str(value): int(count)
                for value, count in zip(*np.unique(raw[te, i], return_counts=True), strict=True)
            }
            if name in {"refusal_delta", "repetition_delta"}
            else None,
        }
    if seed137_predicted_writer is not None and seed137_behavior is not None:
        seed_raw = np.column_stack([seed137_behavior[name] for name in names])
        seed_pred = OP.predict(payload, seed137_predicted_writer) * sd + mu
        out["seed137_frozen_readout"] = {
            name: scalar_metrics(seed_pred[:, i], seed_raw[:, i]) for i, name in enumerate(names)
        }
    return out, payload


def bh_adjust(rows: list[dict[str, Any]], p_key: str = "p") -> None:
    valid = [(i, float(row[p_key])) for i, row in enumerate(rows) if row.get(p_key) is not None]
    if not valid:
        return
    ordered = sorted(valid, key=lambda item: item[1])
    m = len(ordered)
    adjusted = [0.0] * m
    running = 1.0
    for rank_from_end in range(m - 1, -1, -1):
        _, p = ordered[rank_from_end]
        rank = rank_from_end + 1
        running = min(running, p * m / rank)
        adjusted[rank_from_end] = running
    for (item, _), q in zip(ordered, adjusted, strict=True):
        rows[item]["q_bh"] = float(min(1.0, q))


def mode_analysis(
    predicted_writer: np.ndarray,
    observed_writer: np.ndarray,
    behavior: dict[str, np.ndarray],
    folds: dict[str, np.ndarray],
    *,
    top_modes: int,
    device: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    tr, te = folds["tr"], folds["te"]
    train_mean = np.asarray(predicted_writer[tr], np.float64).mean(0)
    train_centered = np.asarray(predicted_writer[tr], np.float64) - train_mean
    dev = torch.device(device)
    _, singular, vh = torch.linalg.svd(
        torch.as_tensor(train_centered, dtype=torch.float64, device=dev),
        full_matrices=False,
    )
    singular_np = singular.cpu().numpy()
    modes = vh[:top_modes].cpu().numpy()
    del singular, vh
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    pred_scores = (np.asarray(predicted_writer[te], np.float64) - train_mean) @ modes.T
    obs_scores = (np.asarray(observed_writer[te], np.float64) - train_mean) @ modes.T
    associations: list[dict[str, Any]] = []
    for mode in range(top_modes):
        for name, values in behavior.items():
            result = safe_corr(pred_scores[:, mode], values[te], "spearman")
            associations.append(
                {
                    "mode": mode + 1,
                    "behavior": name,
                    "rho": result["value"],
                    "p": result["p"],
                }
            )
    bh_adjust(associations)
    per_mode = []
    total_energy = float(np.sum(singular_np**2))
    for mode in range(top_modes):
        per_mode.append(
            {
                "mode": mode + 1,
                "singular_value": float(singular_np[mode]),
                "train_energy_fraction": float(singular_np[mode] ** 2 / total_energy),
                "train_cumulative_energy": float(
                    np.sum(singular_np[: mode + 1] ** 2) / total_energy
                ),
                "heldout_predicted_vs_observed": scalar_metrics(
                    pred_scores[:, mode], obs_scores[:, mode]
                ),
            }
        )
    return (
        {
            "basis": "exact SVD of centered train-row predicted writer contrast",
            "top_modes": int(top_modes),
            "per_mode": per_mode,
            "behavior_associations": associations,
        },
        {
            "writer_train_mean": train_mean.astype(np.float32),
            "writer_output_modes": modes.astype(np.float32),
            "writer_singular_values": singular_np.astype(np.float64),
            "test_predicted_mode_scores": pred_scores.astype(np.float32),
            "test_observed_mode_scores": obs_scores.astype(np.float32),
        },
    )


def affine_similarity(a: Affine, b: Affine) -> dict[str, float]:
    return {
        "operator_cosine": flat_cosine(a.A, b.A),
        "bias_cosine": flat_cosine(a.b, b.b),
        "operator_relative_l2": float(np.linalg.norm(a.A - b.A) / (np.linalg.norm(b.A) + 1e-30)),
        "bias_relative_l2": float(np.linalg.norm(a.b - b.b) / (np.linalg.norm(b.b) + 1e-30)),
    }


def contrast_geometry(
    contrasts: dict[str, Affine], q_context: np.ndarray, te: np.ndarray
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, affine in contrasts.items():
        exercised = affine.predict(q_context[te])
        out[name] = {
            "operator_frobenius": float(np.linalg.norm(affine.A)),
            "bias_l2": float(np.linalg.norm(affine.b)),
            "heldout_prediction_rms_norm": float(
                np.sqrt(np.mean(np.sum(exercised * exercised, axis=1)))
            ),
        }
    out["writer_vs_encoder"] = affine_similarity(contrasts["writer"], contrasts["encoder"])
    out["interaction_vs_writer"] = affine_similarity(
        contrasts["interaction"], contrasts["writer"]
    )
    return out


def refit_half_cells(
    arrays: dict[str, np.ndarray],
    records: dict[str, dict[str, Any]],
    half: np.ndarray,
    alignment: FixedAlignment,
    *,
    device: str,
) -> dict[str, Affine]:
    specs = {
        "q_qwriter": ("q_context", "q_qwriter"),
        "q_lwriter": ("q_context", "q_lwriter"),
        "l_qwriter": ("l_context", "l_qwriter"),
        "l_lwriter": ("l_context", "l_lwriter"),
    }
    cells: dict[str, Affine] = {}
    for name, (x_name, y_name) in specs.items():
        lam = float(records[name]["fit_meta"]["selected_lambda"])
        payload = AT.ridge_beta_at_lambda(
            arrays[x_name], arrays[y_name], half, lam, device=device
        )
        cells[name] = (
            payload_affine(payload)
            if name.startswith("q_")
            else transform_l_affine(payload, alignment)
        )
    return cells


def split_half_stability(
    arrays: dict[str, np.ndarray],
    records: dict[str, dict[str, Any]],
    tr: np.ndarray,
    alignment: FixedAlignment,
    full: dict[str, Affine],
    *,
    device: str,
) -> dict[str, Any]:
    h1, h2 = tr[0::2], tr[1::2]
    cells1 = refit_half_cells(arrays, records, h1, alignment, device=device)
    cells2 = refit_half_cells(arrays, records, h2, alignment, device=device)
    contrast1 = factorial_affines(cells1)
    contrast2 = factorial_affines(cells2)
    out: dict[str, Any] = {"n_half": [int(len(h1)), int(len(h2))], "contrasts": {}}
    for name in CONTRAST_NAMES:
        p1 = contrast1[name].predict(arrays["q_context"])
        p2 = contrast2[name].predict(arrays["q_context"])
        pf = full[name].predict(arrays["q_context"])
        out["contrasts"][name] = {
            "operator_half1_vs_half2_cosine": flat_cosine(contrast1[name].A, contrast2[name].A),
            "bias_half1_vs_half2_cosine": flat_cosine(contrast1[name].b, contrast2[name].b),
            "data_weighted_half1_vs_half2_cosine": flat_cosine(p1, p2),
            "data_weighted_full_vs_half1_cosine": flat_cosine(pf, p1),
            "data_weighted_full_vs_half2_cosine": flat_cosine(pf, p2),
        }
    return out


def load_primary(args: argparse.Namespace) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, OP.MapPayload],
    dict[str, dict[str, Any]],
]:
    split = json.loads(Path(args.split_json).read_text())
    roster, folds = fold_indices(split)
    qroot = Path(args.qwriter_dir)
    lroot = Path(args.lwriter_dir)
    arrays = {
        "q_context": load_bundle(qroot / "qwen_vc_L14.pt", roster),
        "l_context": load_bundle(qroot / "llama_vc_L16.pt", roster),
        "q_qwriter": load_bundle(qroot / "qwen_va_L14.pt", roster),
        "l_qwriter": load_bundle(qroot / "llama_va_L16.pt", roster),
        "q_lwriter": load_bundle(lroot / "qwen_va_L14.pt", roster),
        "l_lwriter": load_bundle(lroot / "llama_va_L16.pt", roster),
    }
    map_dir = Path(args.map_dir)
    names = {
        "q_qwriter": "q14_l16_map_qwen_qwriter.pt",
        "q_lwriter": "q14_l16_map_qwen_lwriter.pt",
        "l_qwriter": "q14_l16_map_llama_qwriter.pt",
        "l_lwriter": "q14_l16_map_llama_lwriter.pt",
    }
    payloads: dict[str, OP.MapPayload] = {}
    records: dict[str, dict[str, Any]] = {}
    for name, filename in names.items():
        payloads[name], records[name] = load_payload(map_dir / filename)
    return roster, folds, arrays, payloads, records


def build_cells(
    payloads: dict[str, OP.MapPayload], alignment: FixedAlignment
) -> dict[str, Affine]:
    return {
        name: payload_affine(payload) if name.startswith("q_") else transform_l_affine(payload, alignment)
        for name, payload in payloads.items()
    }


def observed_cells(arrays: dict[str, np.ndarray], alignment: FixedAlignment) -> dict[str, np.ndarray]:
    return {
        "q_qwriter": np.asarray(arrays["q_qwriter"], np.float64),
        "q_lwriter": np.asarray(arrays["q_lwriter"], np.float64),
        "l_qwriter": alignment.l_answer_to_q(arrays["l_qwriter"]),
        "l_lwriter": alignment.l_answer_to_q(arrays["l_lwriter"]),
    }


def predicted_cells(cells: dict[str, Affine], q_context: np.ndarray) -> dict[str, np.ndarray]:
    return {name: cell.predict(q_context) for name, cell in cells.items()}


def alignment_checks(
    arrays: dict[str, np.ndarray], alignment: FixedAlignment, tr: np.ndarray
) -> dict[str, Any]:
    context_pred = alignment.q_context_to_l(arrays["q_context"][tr])
    answer_q_pred = (
        (np.asarray(arrays["q_qwriter"][tr], np.float64) - alignment.q_answer_mean)
        @ alignment.R_answer
        + alignment.l_answer_mean
    )
    answer_l_pred = (
        (np.asarray(arrays["q_lwriter"][tr], np.float64) - alignment.q_answer_mean)
        @ alignment.R_answer
        + alignment.l_answer_mean
    )
    return {
        "kind": "single fixed train-fitted semi-orthogonal Procrustes; answer fit pooled over writers",
        "n_context_train": int(len(tr)),
        "n_answer_train_pooled": int(2 * len(tr)),
        "context_q_to_l_flat_cosine": flat_cosine(context_pred, arrays["l_context"][tr]),
        "answer_qwriter_q_to_l_flat_cosine": flat_cosine(answer_q_pred, arrays["l_qwriter"][tr]),
        "answer_lwriter_q_to_l_flat_cosine": flat_cosine(answer_l_pred, arrays["l_lwriter"][tr]),
    }


def seed137_analysis(
    args: argparse.Namespace,
    test_roster: np.ndarray,
    alignment: FixedAlignment,
    contrasts: dict[str, Affine],
    cells: dict[str, Affine],
    *,
    device: str,
) -> tuple[dict[str, Any] | None, np.ndarray | None, dict[str, np.ndarray] | None]:
    required = [args.qseed137_dir, args.lseed137_dir, args.reliability_semantic_rows]
    if not all(required):
        return None, None, None
    qroot = Path(args.qseed137_dir)
    lroot = Path(args.lseed137_dir)
    q_context = load_bundle(qroot / "qwen_vc_L14.pt", test_roster)
    q_answer = load_bundle(qroot / "qwen_va_L14.pt", test_roster)
    l_answer = load_bundle(lroot / "llama_va_L16.pt", test_roster)
    pred_writer = contrasts["writer"].predict(q_context)
    pred_diagonal = contrasts["diagonal"].predict(q_context)
    observed_diagonal = q_answer - alignment.l_answer_to_q(l_answer)
    result = {
        "n": int(len(test_roster)),
        "frozen_seed42_diagonal_map_on_seed137": matrix_metrics(
            pred_diagonal,
            observed_diagonal,
            permutation_draws=args.permutation_draws,
            bootstrap_draws=args.bootstrap_draws,
            seed=args.seed + 137,
            device=device,
        ),
        "factorial_identity_max_abs": float(
            np.max(
                np.abs(
                    pred_diagonal
                    - contrasts["writer"].predict(q_context)
                    - contrasts["encoder"].predict(q_context)
                )
            )
        ),
    }
    behavior = None
    if args.qseed137_raw and args.lseed137_raw:
        behavior = reliability_behavior_arrays(
            Path(args.qseed137_raw),
            Path(args.lseed137_raw),
            Path(args.reliability_semantic_rows),
            test_roster,
        )
    return result, pred_writer, behavior


def phase_analyze(args: argparse.Namespace) -> None:
    started = time.time()
    roster, folds, arrays, payloads, records = load_primary(args)
    tr, te = folds["tr"], folds["te"]
    print(f"[mapping-diff] loaded primary matrices n={len(roster)}", flush=True)
    alignment = make_fixed_alignment(
        arrays["q_context"],
        arrays["l_context"],
        arrays["q_qwriter"],
        arrays["l_qwriter"],
        arrays["q_lwriter"],
        arrays["l_lwriter"],
        tr,
        device=args.device,
    )
    print("[mapping-diff] fitted fixed pooled Procrustes alignments", flush=True)
    cells = build_cells(payloads, alignment)
    # Exact affine transform check against the explicit centered coordinate path.
    probe = arrays["q_context"][te[: min(16, len(te))]]
    for name in ("l_qwriter", "l_lwriter"):
        explicit = alignment.l_answer_to_q(
            OP.predict(payloads[name], alignment.q_context_to_l(probe))
        )
        if not np.allclose(cells[name].predict(probe), explicit, rtol=1e-10, atol=1e-8):
            raise AssertionError(f"{name}: transformed affine prediction mismatch")
    contrasts = factorial_affines(cells)
    pred = factorial_arrays(predicted_cells(cells, arrays["q_context"]))
    obs = factorial_arrays(observed_cells(arrays, alignment))
    print("[mapping-diff] formed factorial contrasts", flush=True)

    contrast_reads = {
        name: matrix_metrics(
            pred[name][te],
            obs[name][te],
            permutation_draws=args.permutation_draws,
            bootstrap_draws=args.bootstrap_draws,
            seed=args.seed + i,
            device=args.device,
        )
        for i, name in enumerate(CONTRAST_NAMES)
    }
    behavior_rows = read_jsonl(Path(args.semantic_rows))
    behavior = behavior_arrays(behavior_rows, roster)

    seed_result, seed_writer, seed_behavior = seed137_analysis(
        args,
        roster[te],
        alignment,
        contrasts,
        cells,
        device=args.device,
    )
    behavior_readout, behavior_payload = fit_behavior_readout(
        obs["writer"],
        pred["writer"],
        behavior,
        folds,
        device=args.device,
        seed137_predicted_writer=seed_writer,
        seed137_behavior=seed_behavior,
    )
    print("[mapping-diff] completed held-out behavior readouts", flush=True)

    modes, mode_arrays = mode_analysis(
        pred["writer"],
        obs["writer"],
        behavior,
        folds,
        top_modes=args.top_modes,
        device=args.device,
    )
    print("[mapping-diff] completed exact writer-contrast SVD", flush=True)

    split_half = None
    if not args.skip_split_half:
        split_half = split_half_stability(
            arrays,
            records,
            tr,
            alignment,
            contrasts,
            device=args.device,
        )
        print("[mapping-diff] completed split-half refits", flush=True)

    test_rows: list[dict[str, Any]] = []
    row_cos = {name: row_cosine(pred[name][te], obs[name][te]) for name in CONTRAST_NAMES}
    pred_norm = {name: np.linalg.norm(pred[name][te], axis=1) for name in CONTRAST_NAMES}
    obs_norm = {name: np.linalg.norm(obs[name][te], axis=1) for name in CONTRAST_NAMES}
    for i, idx in enumerate(te):
        row: dict[str, Any] = {"ci": int(roster[idx])}
        for name in CONTRAST_NAMES:
            row[f"{name}_row_cosine"] = float(row_cos[name][i])
            row[f"{name}_predicted_norm"] = float(pred_norm[name][i])
            row[f"{name}_observed_norm"] = float(obs_norm[name][i])
        for name, values in behavior.items():
            row[name] = float(values[idx])
        for mode in range(min(4, args.top_modes)):
            row[f"writer_mode{mode + 1}_predicted_score"] = float(
                mode_arrays["test_predicted_mode_scores"][i, mode]
            )
            row[f"writer_mode{mode + 1}_observed_score"] = float(
                mode_arrays["test_observed_mode_scores"][i, mode]
            )
        test_rows.append(row)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload_obj = {
        **AT.payload_to_dict(behavior_payload),
        "record": behavior_readout["fit"],
        "behavior_axes": behavior_readout["axes"],
        "target_train_mean": behavior_readout["target_train_mean"],
        "target_train_sd": behavior_readout["target_train_sd"],
    }
    with atomic_replace(out_dir / "behavior_readout.pt") as tmp:
        torch.save(payload_obj, tmp)
    atomic_npz(out_dir / "writer_modes.npz", **mode_arrays)
    atomic_jsonl(out_dir / "heldout_rows.jsonl", test_rows)

    result = {
        "issue": 2569,
        "followup_label": "cross-model-mapping-diff",
        "analysis_scope": "primary Qwen L14 / Llama L16; exploratory post-hoc factorial map diff",
        "source_revision": SOURCE_REVISION,
        "layers": PRIMARY_LAYERS,
        "n": {name: int(len(folds[key])) for name, key in (("train", "tr"), ("validation", "va"), ("test", "te"))},
        "roster_sha256": sha_int64(roster),
        "test_roster_sha256": sha_int64(roster[te]),
        "fixed_alignment": alignment_checks(arrays, alignment, tr),
        "factorial_definitions": {
            "writer": "0.5 * [(Qenc,Qwriter - Qenc,Lwriter) + (Lenc,Qwriter - Lenc,Lwriter)]",
            "encoder": "0.5 * [(Qenc,Qwriter + Qenc,Lwriter) - (Lenc,Qwriter + Lenc,Lwriter)]",
            "interaction": "0.5 * [(Qenc,Qwriter - Qenc,Lwriter) - (Lenc,Qwriter - Lenc,Lwriter)]",
            "diagonal": "Qenc,Qwriter - Lenc,Lwriter = writer + encoder exactly",
        },
        "cell_native_test_r2": {name: float(records[name]["test_r2"]) for name in CELL_NAMES},
        "contrast_geometry": contrast_geometry(contrasts, arrays["q_context"], te),
        "heldout_contrast_prediction": contrast_reads,
        "split_half_stability": split_half,
        "writer_modes": modes,
        "behavior_readout": behavior_readout,
        "seed137_reliability": seed_result,
        "caveats": [
            "Exploratory post-hoc analysis on the existing LMSYS-only pilot.",
            "Writer contrast is an answer-policy/content contrast, not a causal mechanism label.",
            "Behavior readouts use objective length/refusal/repetition flags and semantic distance; they do not exhaust behavior.",
            "Fixed pooled Procrustes removes one shared coordinate mismatch but cannot prove complete identifiability across architectures.",
        ],
        "elapsed_s": round(time.time() - started, 2),
    }
    atomic_json(out_dir / "mapping_diff.json", result)
    print(f"[mapping-diff] wrote {out_dir} elapsed={result['elapsed_s']:.1f}s", flush=True)


def phase_selftest(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(2569)
    dq, dl = 5, 7
    q_context = rng.normal(size=(80, dq))
    rc, _ = np.linalg.qr(rng.normal(size=(dl, dq)))
    rc = rc.T
    ra, _ = np.linalg.qr(rng.normal(size=(dl, dq)))
    ra = ra.T
    qcm = rng.normal(size=dq)
    lcm = rng.normal(size=dl)
    qam = rng.normal(size=dq)
    lam = rng.normal(size=dl)
    alignment = FixedAlignment(rc, ra, qcm, lcm, qam, lam)
    native = OP.MapPayload(
        layer=16,
        path=Path("<selftest>"),
        W=rng.normal(size=(dl, dl)),
        xmu=rng.normal(size=dl),
        xsd=np.exp(rng.normal(size=dl)),
        ymu=rng.normal(size=dl),
        selected_lambda=1.0,
        raw={},
    )
    transformed = transform_l_affine(native, alignment)
    explicit = alignment.l_answer_to_q(
        OP.predict(native, alignment.q_context_to_l(q_context))
    )
    assert np.allclose(transformed.predict(q_context), explicit, rtol=1e-11, atol=1e-11)
    cells = {
        "q_qwriter": Affine(rng.normal(size=(dq, dq)), rng.normal(size=dq)),
        "q_lwriter": Affine(rng.normal(size=(dq, dq)), rng.normal(size=dq)),
        "l_qwriter": Affine(rng.normal(size=(dq, dq)), rng.normal(size=dq)),
        "l_lwriter": Affine(rng.normal(size=(dq, dq)), rng.normal(size=dq)),
    }
    contrasts = factorial_affines(cells)
    arrays = factorial_arrays({name: cell.predict(q_context) for name, cell in cells.items()})
    for name in CONTRAST_NAMES:
        assert np.allclose(contrasts[name].predict(q_context), arrays[name])
    truth = rng.normal(size=(30, 8))
    pred = truth + 0.1 * rng.normal(size=truth.shape)
    null = permutation_null(pred, truth, draws=99, seed=42, device="cpu")
    assert null["flat_cosine"]["p_ge"] <= 0.02
    rows = [{"p": p} for p in (0.01, 0.03, 0.2)]
    bh_adjust(rows)
    assert np.allclose([row["q_bh"] for row in rows], [0.03, 0.045, 0.2])
    print("[mapping-diff] selftest PASS")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phase", choices=("analyze", "selftest"), required=True)
    base = PROJECT_ROOT / "data" / "issue_2569" / "ownanswers"
    parser.add_argument("--qwriter-dir", default=str(base / "qwriter_final"))
    parser.add_argument("--lwriter-dir", default=str(base / "writer_llama" / "final"))
    parser.add_argument("--map-dir", default=str(base / "analysis" / "maps"))
    parser.add_argument("--split-json", default=str(base / "analysis" / "split.json"))
    parser.add_argument("--semantic-rows", default=str(base / "analysis" / "semantic" / "per_row.jsonl"))
    parser.add_argument("--qseed137-dir", default=str(base / "reliability" / "qwen_seed137" / "final"))
    parser.add_argument("--lseed137-dir", default=str(base / "reliability" / "llama_seed137" / "final"))
    parser.add_argument("--qseed137-raw", default=str(base / "reliability" / "gen_qwen_s137"))
    parser.add_argument("--lseed137-raw", default=str(base / "reliability" / "gen_llama_s137"))
    parser.add_argument(
        "--reliability-semantic-rows",
        default=str(base / "analysis" / "reliability_semantic_rows.jsonl"),
    )
    parser.add_argument("--out-dir", default=str(base / "mapping_diff"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--permutation-draws", type=int, default=1000)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--top-modes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2569)
    parser.add_argument("--skip-split-half", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.permutation_draws < 99 or args.bootstrap_draws < 99:
        raise ValueError("production analysis requires at least 99 resamples")
    if not 1 <= args.top_modes <= 64:
        raise ValueError("top-modes must be in [1, 64]")
    {"analyze": phase_analyze, "selftest": phase_selftest}[args.phase](args)


if __name__ == "__main__":
    main()
