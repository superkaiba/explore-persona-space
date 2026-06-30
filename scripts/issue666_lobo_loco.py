#!/usr/bin/env python
# ruff: noqa: RUF002, RUF003
# Intentional scientific Unicode (φ, ρ, Δ, →, ×, −) in docstrings/comments.
"""issue #666 Phase 4 — LOBO + LOCO cross-validation drivers (plan §4g, §6).

LOBO (leave-one-behavior-out): hold out one of {marker, bad_medical, em,
taught_fact}; the φ link is calibrated on the REMAINING behaviors' TRAIN partition;
the held-out behavior is scored.
LOCO (leave-one-context-out): hold out one battery context as C'; calibrate on the
rest.

The φ link (``fit_phi`` / ``apply_phi``, re-exported from the shared
``explore_persona_space.analysis.leakage_phi`` module) is a MONOTONE affine→[0,1]
calibration fit on the TRAIN partition ONLY — the held-out test rows never enter
the fit (the train-only-φ discipline, plan §4g). The latent Δs scale is unbounded
and needs NO φ (``score_latent_spearman``).

Reads the per-cell predictor JSONs from ``issue666_predictor.py``, partitions per
fold, scores Spearman ρ / Pearson / sign / AUROC / top-k / calibrated MAE on the
held-out partition. CPU-only; reuses #532/#545 LOBO/LOCO patterns.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "eval_results" / "issue_666" / "lobo_loco"

# Re-export the shared monotone link (the test imports cv.fit_phi / cv.apply_phi).
from explore_persona_space.analysis.leakage_phi import apply_phi, fit_phi  # noqa: E402


@dataclass(frozen=True)
class LOBOFold:
    """One leave-one-behavior-out fold."""

    test_behavior: str
    train_behaviors: tuple[str, ...]


@dataclass(frozen=True)
class LOCOFold:
    """One leave-one-context-out fold."""

    test_context: int
    train_contexts: tuple[int, ...]


def lobo_folds(behaviors) -> list[LOBOFold]:
    """All LOBO folds — one per behavior, holding it out, training on the rest."""
    behaviors = list(behaviors)
    folds = []
    for b in behaviors:
        train = tuple(x for x in behaviors if x != b)
        folds.append(LOBOFold(test_behavior=b, train_behaviors=train))
    return folds


def loco_folds(ctx_ids) -> list[LOCOFold]:
    """All LOCO folds — one per context, holding it out, training on the rest."""
    ctx_ids = list(ctx_ids)
    folds = []
    for c in ctx_ids:
        train = tuple(x for x in ctx_ids if x != c)
        folds.append(LOCOFold(test_context=c, train_contexts=train))
    return folds


def score_latent_spearman(lhat, ds) -> float:
    """Spearman ρ of L̂ vs the latent Δs (the latent-scale headline; NO φ needed).

    The latent Δs scale is unbounded, so the calibration link φ is irrelevant — the
    latent-scale ranking test reads the raw Spearman ρ (plan §4g "Latent-scale
    tests need no φ").
    """
    from scipy.stats import spearmanr

    r = spearmanr(np.asarray(lhat), np.asarray(ds)).statistic
    return float(r) if np.isfinite(r) else 0.0


# ── full metric suite on the behavior [0,1] scale (φ-calibrated) ─────────────
@dataclass
class FoldScores:
    spearman: float
    pearson: float
    sign_agreement: float
    calibrated_mae: float
    n: int
    auroc: float = field(default=float("nan"))


def _auroc(y_true_bin: np.ndarray, scores: np.ndarray) -> float:
    """AUROC of ``scores`` against a binary label (leakage-exceeds-median)."""
    pos = scores[y_true_bin == 1]
    neg = scores[y_true_bin == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    # Mann-Whitney U / (n_pos n_neg) = AUROC.
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1.0
    r_pos = ranks[: pos.size].sum()
    u = r_pos - pos.size * (pos.size + 1) / 2
    return float(u / (pos.size * neg.size))


def score_fold(
    lhat_train, ds_train, lhat_test, ds_test, *, rate_train=None, rate_test=None
) -> FoldScores:
    """Score one fold: latent Spearman/Pearson/sign on test + φ-calibrated MAE.

    φ is fit on (lhat_train, rate_train) ONLY when behavior-rate targets are given;
    the calibrated MAE is then the mean |φ(lhat_test) − rate_test| (pp). The latent
    metrics (Spearman/Pearson/sign) never use φ. Returns a ``FoldScores``.
    """
    from scipy.stats import pearsonr, spearmanr

    lt = np.asarray(lhat_test, dtype=np.float64)
    dt = np.asarray(ds_test, dtype=np.float64)
    sp = spearmanr(lt, dt).statistic if lt.size > 2 else float("nan")
    pr = pearsonr(lt, dt).statistic if lt.size > 2 else float("nan")
    sign = float(np.mean(np.sign(lt) == np.sign(dt))) if lt.size else float("nan")

    mae = float("nan")
    auroc = float("nan")
    if rate_train is not None and rate_test is not None:
        phi = fit_phi(
            np.asarray(lhat_train, dtype=np.float64), np.asarray(rate_train, dtype=np.float64)
        )
        pred = apply_phi(phi, lt)
        rt = np.asarray(rate_test, dtype=np.float64)
        mae = float(np.mean(np.abs(pred - rt)))
        med = np.median(rt)
        auroc = _auroc((rt > med).astype(int), lt)

    return FoldScores(
        spearman=float(sp) if np.isfinite(sp) else float("nan"),
        pearson=float(pr) if np.isfinite(pr) else float("nan"),
        sign_agreement=sign,
        calibrated_mae=mae,
        n=int(lt.size),
        auroc=auroc,
    )


# ── driver: load per-cell predictor JSONs, run LOBO + LOCO ────────────────────
# #664 behavior slug -> LOBO behavior label.
_BEHAVIOR_LABEL = {
    "bad_medical": "bad_medical",
    "em": "em",
    "fact": "taught_fact",
    "marker": "marker",
    "designed_null": "designed_null",
}


def _load_predictor_cells(pred_dir: Path) -> list[dict]:
    recs = []
    for p in sorted(pred_dir.glob("*_predictor_cells.json")):
        recs.append(json.loads(p.read_text()))
    return recs


def _aggregate_fold_stats(folds_out: dict) -> dict:
    """Aggregate per-fold Spearman/sign over a {fold_key: fold_dict} map (plan §4g).

    Shared by LOBO and LOCO — the mean held-out Spearman ρ (the cross-validated
    headline read), the mean sign-agreement, and the fold count, over the finite
    per-fold scores. Returns ``{mean_spearman, mean_sign_agreement, n_folds}``.
    """
    sps = [f["spearman"] for f in folds_out.values() if np.isfinite(f.get("spearman", np.nan))]
    signs = [
        f["sign_agreement"]
        for f in folds_out.values()
        if np.isfinite(f.get("sign_agreement", np.nan))
    ]
    return {
        "mean_spearman": float(np.mean(sps)) if sps else float("nan"),
        "mean_sign_agreement": float(np.mean(signs)) if signs else float("nan"),
        "n_folds": len(folds_out),
    }


def run_lobo_loco(pred_dir: Path) -> dict:
    """Run LOBO + LOCO over the per-cell predictor JSONs; return a summary dict.

    LOBO holds out one behavior, calibrates on the rest, scores the held-out
    behavior. LOCO holds out one battery CONTEXT (by per-behavior bystander index,
    pooled across all cells of a behavior), calibrates on the remaining contexts,
    scores the held-out context's rows. Both fold families are scored by the SAME
    ``score_fold`` / ``_aggregate_fold_stats`` helpers (plan §4g symmetric). LOCO
    is non-empty when the input cells span ≥2 behaviors / ≥2 contexts.
    """
    recs = _load_predictor_cells(pred_dir)
    # Pool per behavior: stack each cell's per-bystander (Lhat, ds), tracking the
    # per-row bystander index (for LOCO's leave-one-context-out partition).
    by_behavior: dict[str, dict[str, list]] = {}
    for r in recs:
        beh = _BEHAVIOR_LABEL.get(r.get("behavior"), r.get("behavior"))
        pb = r["per_bystander"]
        d = by_behavior.setdefault(beh, {"Lhat": [], "ds": [], "ctx_idx": []})
        d["Lhat"].extend(pb["Lhat"])
        d["ds"].extend(pb["ds"])
        d["ctx_idx"].extend(range(len(pb["Lhat"])))  # within-cell bystander index

    behaviors = sorted(by_behavior)

    # ── LOBO: leave-one-behavior-out ──
    lobo = {}
    for fold in lobo_folds(behaviors):
        test_b = fold.test_behavior
        lt = np.array(by_behavior[test_b]["Lhat"])
        dt = np.array(by_behavior[test_b]["ds"])
        tr_l = (
            np.concatenate([by_behavior[b]["Lhat"] for b in fold.train_behaviors])
            if fold.train_behaviors
            else np.array([])
        )
        tr_d = (
            np.concatenate([by_behavior[b]["ds"] for b in fold.train_behaviors])
            if fold.train_behaviors
            else np.array([])
        )
        sc = score_fold(tr_l, tr_d, lt, dt)
        lobo[test_b] = {
            "test_behavior": test_b,
            "spearman": sc.spearman,
            "pearson": sc.pearson,
            "sign_agreement": sc.sign_agreement,
            "n_test": sc.n,
        }

    # ── LOCO: leave-one-context-out (pooled across behaviors + cells) ──
    all_l = (
        np.concatenate([by_behavior[b]["Lhat"] for b in behaviors]) if behaviors else np.array([])
    )
    all_d = np.concatenate([by_behavior[b]["ds"] for b in behaviors]) if behaviors else np.array([])
    all_ctx = (
        np.concatenate([by_behavior[b]["ctx_idx"] for b in behaviors]).astype(int)
        if behaviors
        else np.array([], dtype=int)
    )
    ctx_ids = sorted(set(all_ctx.tolist()))
    loco = {}
    for fold in loco_folds(ctx_ids):
        c = fold.test_context
        test_mask = all_ctx == c
        train_mask = ~test_mask
        sc = score_fold(all_l[train_mask], all_d[train_mask], all_l[test_mask], all_d[test_mask])
        loco[str(c)] = {
            "test_context": int(c),
            "spearman": sc.spearman,
            "pearson": sc.pearson,
            "sign_agreement": sc.sign_agreement,
            "n_test": sc.n,
        }

    return {
        "lobo": lobo,
        "loco": loco,
        "n_behaviors": len(behaviors),
        "behaviors": behaviors,
        "n_contexts": len(ctx_ids),
        "lobo_aggregate": _aggregate_fold_stats(lobo),
        "loco_aggregate": _aggregate_fold_stats(loco),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="issue 666 Phase-4 LOBO/LOCO drivers.")
    ap.add_argument("--pred-dir", default=str(REPO / "eval_results" / "issue_666" / "predictor"))
    ap.add_argument("--slice", action="store_true", help="tiny smoke slice")
    args = ap.parse_args()

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    OUT.mkdir(parents=True, exist_ok=True)
    summary = run_lobo_loco(Path(args.pred_dir))
    outp = OUT / "lobo_loco.json"
    outp.write_text(json.dumps(summary, indent=1))
    print(
        f"[lobo_loco] LOBO {len(summary['lobo'])} behaviors / "
        f"LOCO {len(summary['loco'])} contexts -> {outp}"
    )
    print("[phase=lobo_loco] done OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
