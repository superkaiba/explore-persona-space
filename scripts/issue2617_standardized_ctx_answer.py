"""Standardized context-vs-answer separation on the exact #2617 pairs.

This CPU-only follow-up implements the preregistration in
``eval_results/issue_2617/standardized_ctx_answer/preregistration.json``.
The pair is the independent unit.  Each representation receives the same
family-grouped folds, fixed-rank PCA-whitening recipe, held-out scoring, and
paired bootstrap; covariance is estimated separately within each space.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold


REPO_ROOT = Path(__file__).resolve().parents[1]
PREREG_PATH = (
    REPO_ROOT / "eval_results" / "issue_2617" / "standardized_ctx_answer" / "preregistration.json"
)
OUT_DIR = REPO_ROOT / "eval_results" / "issue_2617" / "standardized_ctx_answer"
FIGURE_PATH = REPO_ROOT / "figures" / "issue_2617" / "standardized_ctx_answer_effect.png"
PRIMARY_LAYER = 19
PRIMARY_RANK = 32
RANKS = (16, 32, 64)
N_SPLITS = 5
SPLIT_SEED = 2617
BOOT_SEED = 2_617_001
N_BOOT = 10_000
EIGEN_FLOOR_REL = 1e-10
EPS = 1e-12


@dataclass(frozen=True)
class PairRow:
    pair_id: str
    pair_class: str
    pair_source: str
    artifact_family_id: str
    a_idx: int
    b_idx: int
    refusal_rate_a: float
    refusal_rate_b: float
    behavior_gap: float
    outcome_group: str
    hi_idx: int
    lo_idx: int
    orientation: str


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def _paths(data_root: Path) -> dict[str, Path]:
    return {
        "issue2617_svmp/manifests/svmp_bank.json": data_root / "manifests" / "svmp_bank.json",
        "issue2617_svmp/analysis_tensors/vc/vc_langow_bank.pt": data_root
        / "analysis_tensors"
        / "vc"
        / "vc_langow_bank.pt",
        "issue2617_svmp/analysis_tensors/va/va_langow_query_svmp.pt": data_root
        / "analysis_tensors"
        / "va"
        / "va_langow_query_svmp.pt",
        "issue2617_svmp/raw_completions/judge/judge_scores.json": data_root
        / "raw_completions"
        / "judge"
        / "judge_scores.json",
    }


def validate_inputs(data_root: Path, prereg: dict) -> dict[str, str]:
    realized: dict[str, str] = {}
    for remote, path in _paths(data_root).items():
        if not path.is_file():
            raise FileNotFoundError(f"missing preregistered input: {path}")
        realized[remote] = _sha256(path)
        expected = prereg["data"]["files"][remote]
        if realized[remote] != expected:
            raise RuntimeError(f"sha256 mismatch for {remote}: {realized[remote]} != {expected}")
    return realized


def _outcome_group(gap: float) -> str:
    if gap >= 0.5:
        return "flip"
    if gap <= 0.1:
        return "nonflip"
    return "mid"


def load_data(data_root: Path) -> tuple[np.ndarray, np.ndarray, list[PairRow], dict]:
    paths = _paths(data_root)
    manifest = json.loads(
        paths["issue2617_svmp/manifests/svmp_bank.json"].read_text(encoding="utf-8")
    )
    judge = json.loads(
        paths["issue2617_svmp/raw_completions/judge/judge_scores.json"].read_text(encoding="utf-8")
    )
    vc_store = torch.load(
        paths["issue2617_svmp/analysis_tensors/vc/vc_langow_bank.pt"],
        map_location="cpu",
        weights_only=False,
    )
    va_store = torch.load(
        paths["issue2617_svmp/analysis_tensors/va/va_langow_query_svmp.pt"],
        map_location="cpu",
        weights_only=False,
    )

    if manifest["n_pairs"] != 108 or manifest["n_contexts"] != 216:
        raise RuntimeError("#2617 manifest cardinality changed")
    context_ids = [row["id"] for row in manifest["contexts"]]
    if len(context_ids) != len(set(context_ids)):
        raise RuntimeError("duplicate context ids")
    context_pos = {context_id: i for i, context_id in enumerate(context_ids)}

    vc_layers = [int(layer) for layer in vc_store["layers"]]
    if PRIMARY_LAYER not in vc_layers:
        raise RuntimeError(f"L{PRIMARY_LAYER} absent from v_C store: {vc_layers}")
    vc_store_pos = {context_id: i for i, context_id in enumerate(vc_store["context_ids"])}
    if set(context_ids) - set(vc_store_pos):
        raise RuntimeError("v_C store is missing manifest contexts")
    vc_rows = [vc_store_pos[context_id] for context_id in context_ids]
    vc = (
        vc_store["vc"][vc_rows, vc_layers.index(PRIMARY_LAYER)]
        .to(dtype=torch.float32)
        .numpy()
        .astype(np.float64)
    )

    va_layers = [int(layer) for layer in va_store["layers"]]
    if PRIMARY_LAYER not in va_layers:
        raise RuntimeError(f"L{PRIMARY_LAYER} absent from v_A store: {va_layers}")
    if va_store["empty_rows"]:
        raise RuntimeError(
            "preregistration assumes ten captured answer vectors per context; "
            f"found {len(va_store['empty_rows'])} empty rows"
        )
    va_values = (
        va_store["va_tail_incl"][:, va_layers.index(PRIMARY_LAYER)]
        .to(dtype=torch.float32)
        .numpy()
        .astype(np.float64)
    )
    sums = np.zeros_like(vc)
    counts = np.zeros(len(context_ids), dtype=np.int64)
    seen_draws: dict[str, set[int]] = {context_id: set() for context_id in context_ids}
    if len(va_store["index"]) != len(va_values):
        raise RuntimeError("v_A index and tensor row counts disagree")
    for row_i, rec in enumerate(va_store["index"]):
        context_id = rec["context_id"]
        if context_id not in context_pos:
            raise RuntimeError(f"unknown v_A context id: {context_id}")
        draw = int(rec["draw"])
        if draw in seen_draws[context_id]:
            raise RuntimeError(f"duplicate v_A draw: {(context_id, draw)}")
        seen_draws[context_id].add(draw)
        idx = context_pos[context_id]
        sums[idx] += va_values[row_i]
        counts[idx] += 1
    if not np.all(counts == 10):
        raise RuntimeError(f"expected ten v_A rows per context; got {np.unique(counts)}")
    va = sums / counts[:, None]

    judge_pc = judge["per_context"]
    pairs: list[PairRow] = []
    for rec in manifest["pairs"]:
        a_idx, b_idx = context_pos[rec["a"]], context_pos[rec["b"]]
        rate_a = float(judge_pc[rec["a"]]["refusal_rate"])
        rate_b = float(judge_pc[rec["b"]]["refusal_rate"])
        gap = abs(rate_a - rate_b)
        if rate_a > rate_b:
            hi_idx, lo_idx, orientation = a_idx, b_idx, "a_minus_b"
        elif rate_b > rate_a:
            hi_idx, lo_idx, orientation = b_idx, a_idx, "b_minus_a"
        else:
            hi_idx, lo_idx, orientation = a_idx, b_idx, "tie_a_minus_b"
        pairs.append(
            PairRow(
                pair_id=rec["pair_id"],
                pair_class=rec["pair_class"],
                pair_source=rec["pair_source"],
                artifact_family_id=rec["artifact_family_id"],
                a_idx=a_idx,
                b_idx=b_idx,
                refusal_rate_a=rate_a,
                refusal_rate_b=rate_b,
                behavior_gap=gap,
                outcome_group=_outcome_group(gap),
                hi_idx=hi_idx,
                lo_idx=lo_idx,
                orientation=orientation,
            )
        )

    counts_by_outcome = {
        label: sum(pair.outcome_group == label for pair in pairs)
        for label in ("flip", "nonflip", "mid")
    }
    expected = {"flip": 60, "nonflip": 41, "mid": 7}
    if counts_by_outcome != expected:
        raise RuntimeError(f"#2617 outcome counts changed: {counts_by_outcome} != {expected}")
    provenance = {
        "vc_layers": vc_layers,
        "va_layers": va_layers,
        "vc_position": vc_store.get("position"),
        "va_poolings": va_store.get("poolings"),
        "answer_rows_per_context": 10,
        "judge_model": judge.get("judge_model"),
        "refused_threshold": judge.get("refused_threshold"),
        "n_judge_dropped": judge.get("n_dropped_total"),
    }
    return vc, va, pairs, provenance


def make_folds(pairs: list[PairRow]) -> tuple[np.ndarray, list[dict]]:
    labels = np.array([pair.outcome_group for pair in pairs], dtype=object)
    groups = np.array([pair.artifact_family_id for pair in pairs], dtype=object)
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SPLIT_SEED)
    fold_of = np.full(len(pairs), -1, dtype=np.int64)
    fold_meta: list[dict] = []
    dummy = np.zeros((len(pairs), 1), dtype=np.float64)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(dummy, labels, groups)):
        if np.intersect1d(groups[train_idx], groups[test_idx]).size:
            raise RuntimeError(f"family leakage in fold {fold}")
        if np.any(fold_of[test_idx] != -1):
            raise RuntimeError("test pair assigned to multiple folds")
        fold_of[test_idx] = fold
        fold_meta.append(
            {
                "fold": fold,
                "n_train_pairs": len(train_idx),
                "n_test_pairs": len(test_idx),
                "train_outcome_counts": {
                    label: int(np.sum(labels[train_idx] == label))
                    for label in ("flip", "nonflip", "mid")
                },
                "test_outcome_counts": {
                    label: int(np.sum(labels[test_idx] == label))
                    for label in ("flip", "nonflip", "mid")
                },
                "test_families": sorted(set(groups[test_idx].tolist())),
            }
        )
    if np.any(fold_of < 0):
        raise RuntimeError("not every pair received a test fold")
    return fold_of, fold_meta


def fit_pca_whitener(
    train_rows: np.ndarray, *, max_rank: int = max(RANKS)
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    x = np.asarray(train_rows, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] <= max_rank:
        raise ValueError(f"need more than {max_rank} training rows, got {x.shape}")
    mean = x.mean(axis=0)
    centered = x - mean
    _u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    if singular[max_rank - 1] <= singular[0] * EIGEN_FLOOR_REL:
        raise RuntimeError(
            f"rank-{max_rank} singular value below registered floor: "
            f"{singular[max_rank - 1]} <= {singular[0] * EIGEN_FLOOR_REL}"
        )
    scales = singular / np.sqrt(x.shape[0] - 1)
    diagnostics = {
        "n_train_rows": x.shape[0],
        "max_rank": max_rank,
        "singular_value_max": singular[0],
        "singular_value_at_max_rank": singular[max_rank - 1],
        "covariance_eigenvalue_max": scales[0] ** 2,
        "covariance_eigenvalue_at_max_rank": scales[max_rank - 1] ** 2,
    }
    return mean, vt[:max_rank], scales[:max_rank], diagnostics


def apply_pca_whitener(
    x: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
    scales: np.ndarray,
    *,
    rank: int,
) -> np.ndarray:
    if rank > len(scales):
        raise ValueError(f"rank {rank} exceeds fitted rank {len(scales)}")
    return ((np.asarray(x, dtype=np.float64) - mean) @ components[:rank].T) / scales[:rank]


def cross_validated_margins(
    x: np.ndarray,
    pairs: list[PairRow],
    fold_of: np.ndarray,
) -> tuple[dict[int, np.ndarray], list[dict]]:
    pair_rows = np.array([[pair.a_idx, pair.b_idx] for pair in pairs], dtype=np.int64)
    margins = {rank: np.full(len(pairs), np.nan, dtype=np.float64) for rank in RANKS}
    diagnostics: list[dict] = []
    for fold in range(N_SPLITS):
        train_pair_idx = np.flatnonzero(fold_of != fold)
        test_pair_idx = np.flatnonzero(fold_of == fold)
        train_context_idx = np.unique(pair_rows[train_pair_idx].reshape(-1))
        mean, components, scales, fit_diag = fit_pca_whitener(x[train_context_idx])
        z = apply_pca_whitener(x, mean, components, scales, rank=max(RANKS))
        hi = np.array([pair.hi_idx for pair in pairs], dtype=np.int64)
        lo = np.array([pair.lo_idx for pair in pairs], dtype=np.int64)
        deltas = z[hi] - z[lo]
        train_flip_idx = np.array(
            [i for i in train_pair_idx if pairs[i].outcome_group == "flip"],
            dtype=np.int64,
        )
        if len(train_flip_idx) < 8:
            raise RuntimeError(f"fold {fold} has only {len(train_flip_idx)} training flips")
        fold_diag = {"fold": fold, "fit": fit_diag, "by_rank": {}}
        for rank in RANKS:
            direction = deltas[train_flip_idx, :rank].mean(axis=0)
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm <= EPS:
                raise RuntimeError(f"zero refusal direction in fold {fold}, rank {rank}")
            direction /= direction_norm
            margins[rank][test_pair_idx] = deltas[test_pair_idx, :rank] @ direction
            fold_diag["by_rank"][str(rank)] = {
                "n_train_flip_pairs": len(train_flip_idx),
                "direction_norm_before_unit_scaling": direction_norm,
            }
        diagnostics.append(fold_diag)
    for rank, values in margins.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f"rank {rank} did not produce one finite OOF score per pair")
    return margins, diagnostics


def hedges_g(group_a: np.ndarray, group_b: np.ndarray) -> dict[str, float]:
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        raise ValueError("Hedges' g needs at least two rows per group")
    pooled_var = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (
        len(a) + len(b) - 2
    )
    pooled_sd = float(np.sqrt(pooled_var))
    if pooled_sd <= EPS:
        raise RuntimeError("pooled SD is zero")
    d = float((a.mean() - b.mean()) / pooled_sd)
    correction = 1.0 - 3.0 / (4.0 * (len(a) + len(b)) - 9.0)
    return {
        "mean_flip": float(a.mean()),
        "mean_nonflip": float(b.mean()),
        "mean_difference": float(a.mean() - b.mean()),
        "pooled_sd": pooled_sd,
        "cohens_d": d,
        "small_sample_correction": correction,
        "hedges_g": correction * d,
    }


def _hedges_g_draws(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0]:
        raise ValueError("bootstrap groups must be two matrices with matching draw count")
    pooled_var = (
        (a.shape[1] - 1) * a.var(axis=1, ddof=1) + (b.shape[1] - 1) * b.var(axis=1, ddof=1)
    ) / (a.shape[1] + b.shape[1] - 2)
    correction = 1.0 - 3.0 / (4.0 * (a.shape[1] + b.shape[1]) - 9.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return correction * (a.mean(axis=1) - b.mean(axis=1)) / np.sqrt(pooled_var)


def _ci(values: np.ndarray) -> list[float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return [float("nan"), float("nan")]
    return np.quantile(finite, [0.025, 0.975]).tolist()


def _group_summary(values: np.ndarray, boot_indices: np.ndarray) -> dict:
    vals = np.asarray(values, dtype=np.float64)
    draws = vals[boot_indices]
    return {
        "n": len(vals),
        "mean": float(vals.mean()),
        "mean_ci95": _ci(draws.mean(axis=1)),
        "median": float(np.median(vals)),
        "sign_accuracy": float(np.mean(vals > 0.0)),
        "sign_accuracy_ci95": _ci(np.mean(draws > 0.0, axis=1)),
    }


def summarize_rank(
    context_margin: np.ndarray,
    answer_margin: np.ndarray,
    pairs: list[PairRow],
    *,
    n_boot: int,
    seed: int,
) -> dict:
    outcomes = np.array([pair.outcome_group for pair in pairs], dtype=object)
    flip_idx = np.flatnonzero(outcomes == "flip")
    nonflip_idx = np.flatnonzero(outcomes == "nonflip")
    rng = np.random.default_rng(seed)
    flip_boot_local = rng.integers(0, len(flip_idx), size=(n_boot, len(flip_idx)))
    nonflip_boot_local = rng.integers(0, len(nonflip_idx), size=(n_boot, len(nonflip_idx)))
    flip_boot = flip_idx[flip_boot_local]
    nonflip_boot = nonflip_idx[nonflip_boot_local]

    context_point = hedges_g(context_margin[flip_idx], context_margin[nonflip_idx])
    answer_point = hedges_g(answer_margin[flip_idx], answer_margin[nonflip_idx])
    context_draw = _hedges_g_draws(context_margin[flip_boot], context_margin[nonflip_boot])
    answer_draw = _hedges_g_draws(answer_margin[flip_boot], answer_margin[nonflip_boot])
    delta_draw = answer_draw - context_draw
    delta_point = answer_point["hedges_g"] - context_point["hedges_g"]
    decision = (
        "v_A_separates_more"
        if _ci(delta_draw)[0] > 0
        else ("v_C_separates_more" if _ci(delta_draw)[1] < 0 else "no_detected_difference")
    )

    spaces = {}
    for name, values, point, g_draw in (
        ("v_C", context_margin, context_point, context_draw),
        ("v_A", answer_margin, answer_point, answer_draw),
    ):
        spaces[name] = {
            **point,
            "hedges_g_ci95": _ci(g_draw),
            "flip": _group_summary(values[flip_idx], flip_boot_local),
            "nonflip": _group_summary(values[nonflip_idx], nonflip_boot_local),
        }

    paired_flip_delta = answer_margin[flip_idx] - context_margin[flip_idx]
    paired_flip_draw = paired_flip_delta[flip_boot_local].mean(axis=1)
    by_pair_class: dict[str, dict] = {}
    pair_classes = np.array([pair.pair_class for pair in pairs], dtype=object)
    for pair_class in sorted(set(pair_classes.tolist())):
        idx = np.flatnonzero(pair_classes == pair_class)
        by_pair_class[pair_class] = {
            "n": len(idx),
            "outcome_counts": {
                label: int(np.sum(outcomes[idx] == label)) for label in ("flip", "nonflip", "mid")
            },
            "v_C_margin_mean": float(context_margin[idx].mean()),
            "v_C_margin_median": float(np.median(context_margin[idx])),
            "v_A_margin_mean": float(answer_margin[idx].mean()),
            "v_A_margin_median": float(np.median(answer_margin[idx])),
        }

    benign_idx = np.array(
        [i for i, pair in enumerate(pairs) if pair.pair_class.endswith("_benign")],
        dtype=np.int64,
    )
    return {
        "spaces": spaces,
        "headline_contrast": {
            "estimand": "hedges_g_v_A_minus_v_C",
            "point": delta_point,
            "ci95": _ci(delta_draw),
            "n_valid_bootstrap_draws": int(np.isfinite(delta_draw).sum()),
            "decision": decision,
        },
        "paired_flip_margin_contrast": {
            "estimand": "mean_flip_margin_v_A_minus_v_C",
            "point": float(paired_flip_delta.mean()),
            "ci95": _ci(paired_flip_draw),
        },
        "benign_controls": {
            "n": len(benign_idx),
            "v_C_margin_mean": float(context_margin[benign_idx].mean()),
            "v_A_margin_mean": float(answer_margin[benign_idx].mean()),
            "v_C_margin_abs_mean": float(np.abs(context_margin[benign_idx]).mean()),
            "v_A_margin_abs_mean": float(np.abs(answer_margin[benign_idx]).mean()),
        },
        "by_pair_class": by_pair_class,
    }


def make_figure(summary: dict, path: Path) -> None:
    primary = summary["ranks"][str(PRIMARY_RANK)]
    colors = {"v_C": "#356A71", "v_A": "#D47A28"}
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), constrained_layout=True)

    names = ["v_C", "v_A"]
    points = np.array([primary["spaces"][name]["hedges_g"] for name in names])
    cis = np.array([primary["spaces"][name]["hedges_g_ci95"] for name in names])
    yerr = np.vstack([np.maximum(0.0, points - cis[:, 0]), np.maximum(0.0, cis[:, 1] - points)])
    axes[0].bar([0, 1], points, color=[colors[name] for name in names], width=0.62, zorder=2)
    axes[0].errorbar(
        [0, 1], points, yerr=yerr, fmt="none", ecolor="#20262B", capsize=5, lw=1.5, zorder=3
    )
    axes[0].axhline(0.0, color="#6C747A", lw=0.9)
    axes[0].set_xticks([0, 1], ["Context $v_C$", "Answer $v_A$"])
    axes[0].set_ylabel("Hedges' $g$: flip vs nonflip OOF margins")
    delta = primary["headline_contrast"]
    axes[0].set_title(
        f"Paired Δg = {delta['point']:+.2f}\n95% CI [{delta['ci95'][0]:+.2f}, {delta['ci95'][1]:+.2f}]",
        fontsize=11,
    )

    x = np.array([0.0, 1.0])
    offsets = {"v_C": -0.11, "v_A": 0.11}
    markers = {"v_C": "o", "v_A": "s"}
    for name in names:
        group_points = np.array(
            [primary["spaces"][name][group]["mean"] for group in ("nonflip", "flip")]
        )
        group_cis = np.array(
            [primary["spaces"][name][group]["mean_ci95"] for group in ("nonflip", "flip")]
        )
        group_err = np.vstack(
            [
                np.maximum(0.0, group_points - group_cis[:, 0]),
                np.maximum(0.0, group_cis[:, 1] - group_points),
            ]
        )
        axes[1].errorbar(
            x + offsets[name],
            group_points,
            yerr=group_err,
            fmt=markers[name],
            ms=7,
            lw=1.7,
            capsize=4,
            color=colors[name],
            label=("Context $v_C$" if name == "v_C" else "Answer $v_A$"),
        )
    axes[1].axhline(0.0, color="#6C747A", lw=0.9)
    axes[1].set_xticks(x, ["Nonflip controls\n(n=41)", "Behavior flips\n(n=60)"])
    axes[1].set_ylabel("Held-out standardized pair margin")
    axes[1].set_title("Same folds and rank-32 whitening")
    axes[1].legend(frameon=False, loc="upper left")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#D9DDDF", lw=0.7, alpha=0.7, zorder=0)
    fig.suptitle("#2617 refusal–compliance separation: context vs observed answer", fontsize=13)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(data_root: Path, *, n_boot: int = N_BOOT) -> dict:
    prereg = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    hashes = validate_inputs(data_root, prereg)
    vc, va, pairs, tensor_provenance = load_data(data_root)
    fold_of, fold_meta = make_folds(pairs)
    vc_margins, vc_diagnostics = cross_validated_margins(vc, pairs, fold_of)
    va_margins, va_diagnostics = cross_validated_margins(va, pairs, fold_of)

    ranks = {
        str(rank): summarize_rank(
            vc_margins[rank],
            va_margins[rank],
            pairs,
            n_boot=n_boot,
            seed=BOOT_SEED,
        )
        for rank in RANKS
    }
    summary = {
        "schema": "issue2617-standardized-context-answer-result-v1",
        "issue": 2617,
        "preregistration": str(PREREG_PATH.relative_to(REPO_ROOT)),
        "code_git_revision": _git_revision(),
        "data": {
            "root_used": str(data_root),
            "hf_repo": prereg["data"]["hf_repo"],
            "hf_revision": prereg["data"]["revision"],
            "verified_sha256": hashes,
            "tensor_provenance": tensor_provenance,
        },
        "design": {
            "n_pairs": len(pairs),
            "n_contexts": len(vc),
            "outcome_counts": {
                label: sum(pair.outcome_group == label for pair in pairs)
                for label in ("flip", "nonflip", "mid")
            },
            "layer": PRIMARY_LAYER,
            "answer_pooling": "tail_inclusive_mean_over_10_captured_rollouts",
            "primary_rank": PRIMARY_RANK,
            "sensitivity_ranks": [rank for rank in RANKS if rank != PRIMARY_RANK],
            "n_folds": N_SPLITS,
            "split_seed": SPLIT_SEED,
            "bootstrap_draws": n_boot,
            "bootstrap_seed": BOOT_SEED,
            "independent_unit": "pair",
            "whitening": "fold-local PCA, separate covariance per space, matched rank/recipe",
        },
        "folds": fold_meta,
        "pair_fold_assignment": {pair.pair_id: int(fold_of[i]) for i, pair in enumerate(pairs)},
        "whitening_diagnostics": {"v_C": vc_diagnostics, "v_A": va_diagnostics},
        "ranks": ranks,
        "primary": ranks[str(PRIMARY_RANK)],
        "interpretive_limit": prereg["known_interpretive_limit"],
        "figure": {
            "path": str(FIGURE_PATH.relative_to(REPO_ROOT)),
            "title": "#2617 refusal-compliance separation: context vs observed answer",
            "alt_text": "Two-panel figure comparing rank-32 cross-validated standardized refusal-pair separation for context and answer vectors. The left panel shows Hedges' g with paired-bootstrap confidence intervals; the right panel shows held-out margins for nonflip controls and behavior flips.",
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (OUT_DIR / "perpair.jsonl").open("w", encoding="utf-8") as handle:
        for i, pair in enumerate(pairs):
            record = {
                **pair.__dict__,
                "fold": int(fold_of[i]),
                "margin_by_rank": {
                    str(rank): {
                        "v_C": float(vc_margins[rank][i]),
                        "v_A": float(va_margins[rank][i]),
                    }
                    for rank in RANKS
                },
            }
            handle.write(json.dumps(_jsonable(record), allow_nan=False) + "\n")
    make_figure(summary, FIGURE_PATH)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="local issue2617_svmp directory containing manifests/, analysis_tensors/, and raw_completions/",
    )
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOT)
    args = parser.parse_args()
    if args.n_bootstrap < 100:
        raise ValueError("--n-bootstrap must be at least 100")
    result = run(args.data_root, n_boot=args.n_bootstrap)
    primary = result["primary"]
    print(
        json.dumps({"primary": primary["headline_contrast"], "spaces": primary["spaces"]}, indent=2)
    )


if __name__ == "__main__":
    main()
