#!/usr/bin/env python3
"""Stages 6+7 of issue #343: build regression CSV with 5 axes and run OLS.

For each (trained_adapter, panel_test_id) cell:
  marker_rate  = from Stage 3 panel_eval.json
  semantic_cos = from data/i181_non_persona/system_prompt_embeddings.pt
  lexical_jac  = lexical Jaccard over lowercased word tokens
  struct_match = i181 5-feature hamming match
  task_match   = 1 if train_family == test_family else 0
  js_div       = from Stage 5 js_divergence_matrix.npz

Excludes ``test_bucket`` in {control_empty, control_default, persona_anchor}
per brief. Expects 12 adapters × 32 cells = 384 rows.

Stats reported (p-values + N + ΔR² / partial-Spearman only, per CLAUDE.md):
  - Per-axis OLS coefficient + p-value (full 5-axis model)
  - Spearman correlation of each axis with marker_rate
  - Partial Spearman rho(js_div, marker_rate | lexical_jac) and reverse
  - ΔR^2 for adding JS to (cosine, lexical, struct, task) baseline,
    and for adding lexical to (cosine, struct, task, js) baseline
  - Leave-one-trigger-out CV R^2: full vs no-JS vs no-lexical

Usage:
    uv run python scripts/i207_run_regression.py
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats as sstats

from explore_persona_space.analysis.i181_features import (
    compute_lexical_jaccard,
    compute_semantic_cosine,
    compute_struct_match,
    compute_structural_features,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "i181_non_persona"
JS_GENTLE = PROJECT_ROOT / "eval_results" / "issue_207" / "js_gentle"

FAMILIES = ["T_task", "T_instruction", "T_context", "T_format"]
SEEDS = [42, 137, 256]
AXIS_NAMES = ["semantic_cos", "lexical_jac", "struct_match", "task_match", "js_div"]
# Excluded test_buckets — controls + persona anchors per brief
EXCLUDE_BUCKETS = {"control_empty", "control_default", "persona_anchor"}


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _load_embeddings() -> dict:
    emb_path = DATA_DIR / "system_prompt_embeddings.pt"
    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings file missing: {emb_path}")
    data = torch.load(emb_path, weights_only=False)
    # Schema from build_i181_data.py compute_embeddings():
    #   {"ids": list[str], "embeddings": Tensor[n, hidden], "global_mean": ..., ...}
    if not isinstance(data, dict) or "ids" not in data or "embeddings" not in data:
        raise ValueError(f"Unexpected embeddings format in {emb_path}: {list(data.keys())}")
    return data


def _embedding_for(panel_id: str, embeddings: dict):
    """Resolve a panel_id to its embedding tensor (None if missing).

    Matches the lookup logic in scripts/analyze_i181.py::get_embedding.
    """
    ids = embeddings["ids"]
    emb_matrix = embeddings["embeddings"]
    if panel_id in ids:
        return emb_matrix[ids.index(panel_id)]
    # Training triggers may be stored under train_<name>
    train_id = f"train_{panel_id.replace('match_', '')}"
    if train_id in ids:
        return emb_matrix[ids.index(train_id)]
    return None


def load_panel_evals() -> dict[str, dict]:
    """Load panel_eval.json for all 12 gentle-recipe adapters."""
    panel_evals = {}
    for fam in FAMILIES:
        family_label = fam.replace("T_", "")
        for seed in SEEDS:
            run_name = f"i181_gentle_{family_label}_seed{seed}_train"
            path = JS_GENTLE / run_name / "panel_eval.json"
            if not path.exists():
                logger.warning("MISSING panel_eval: %s", path)
                continue
            panel_evals[run_name] = json.loads(path.read_text())
    logger.info("Loaded %d panel evals (expected 12)", len(panel_evals))
    return panel_evals


def load_js_matrix() -> tuple[np.ndarray, list[str]]:
    """Return (js_matrix [n,n], prompts [n] ordered)."""
    npz_path = JS_GENTLE / "js_divergence_matrix.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"JS matrix missing: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    js = data["js"].astype(np.float32)
    prompts = list(data["prompts"])
    logger.info("Loaded JS matrix (n=%d), first 4 prompt ids: %s", len(prompts), prompts[:4])
    return js, prompts


def build_dataframe(panel_evals: dict, js: np.ndarray, js_prompts: list[str]) -> pd.DataFrame:
    """Build the 384-row regression DataFrame."""
    # Load triggers + panel
    triggers_data = json.loads((DATA_DIR / "triggers.json").read_text())
    triggers = triggers_data["triggers"]
    trigger_families = triggers_data["families"]

    panel_data = json.loads((DATA_DIR / "eval_panel.json").read_text())
    panel = panel_data["panel"]
    panel_lookup = {e["id"]: e for e in panel}

    embeddings = _load_embeddings()
    js_idx = {pid: i for i, pid in enumerate(js_prompts)}

    rows: list[dict] = []
    for run_name, eval_data in panel_evals.items():
        # Parse: i181_gentle_<family>_seed<seed>_train
        # e.g. i181_gentle_task_seed42_train
        parts = run_name.split("_")
        # parts: ['i181', 'gentle', '<family>', 'seed<N>', 'train']
        try:
            family_label = parts[2]  # 'task' / 'instruction' / 'context' / 'format'
            seed = int(parts[3].replace("seed", ""))
        except (IndexError, ValueError):
            logger.warning("Cannot parse run_name: %s", run_name)
            continue
        condition = f"T_{family_label}"
        train_family = trigger_families.get(condition, family_label)
        train_prompt = triggers.get(condition, "")
        train_emb = _embedding_for(condition, embeddings)
        if train_emb is None:
            train_emb = _embedding_for(f"match_{condition}", embeddings)
        train_feats = compute_structural_features(train_prompt)

        # JS lookup uses the trigger id (e.g. 'T_task') for the source axis
        if condition not in js_idx:
            logger.warning("Train trigger %s missing in JS matrix prompts", condition)
            continue
        train_js_idx = js_idx[condition]

        results = eval_data.get("results", {})
        for test_id, cell in results.items():
            test_entry = panel_lookup.get(test_id)
            if test_entry is None:
                continue
            test_bucket = test_entry.get("bucket", "unknown")
            if test_bucket in EXCLUDE_BUCKETS:
                continue

            test_prompt = test_entry["system_prompt"]
            test_family = test_entry.get("family") or "none"

            # Axes
            test_emb = _embedding_for(test_id, embeddings)
            if train_emb is not None and test_emb is not None:
                sem_cos = compute_semantic_cosine(train_emb, test_emb)
            else:
                sem_cos = 0.0
            lex_jac = compute_lexical_jaccard(train_prompt, test_prompt)
            test_feats = compute_structural_features(test_prompt)
            struct = compute_struct_match(train_feats, test_feats)
            task = 1.0 if train_family == test_family else 0.0

            # JS lookup — test_id should appear in js_prompts
            if test_id not in js_idx:
                logger.warning("Test prompt %s missing in JS matrix prompts", test_id)
                js_div = float("nan")
            else:
                js_div = float(js[train_js_idx, js_idx[test_id]])

            rows.append(
                {
                    "run_name": run_name,
                    "condition": condition,
                    "seed": seed,
                    "train_family": train_family,
                    "test_id": test_id,
                    "test_family": test_family,
                    "test_bucket": test_bucket,
                    "marker_rate": float(cell.get("marker_rate", 0.0)),
                    "semantic_cos": float(sem_cos),
                    "lexical_jac": float(lex_jac),
                    "struct_match": float(struct),
                    "task_match": float(task),
                    "js_div": js_div,
                }
            )

    df = pd.DataFrame(rows)
    logger.info("Built DataFrame: %d rows (expected ~384 = 12*32)", len(df))
    n_nan = df[AXIS_NAMES].isna().any(axis=1).sum()
    if n_nan:
        logger.warning("%d rows have NaN in axis columns; dropping", n_nan)
        df = df.dropna(subset=AXIS_NAMES).copy()
    return df


def fit_ols(X: np.ndarray, y: np.ndarray, axis_names: list[str]) -> dict:
    """Plain OLS with per-coef p-values."""
    from sklearn.linear_model import LinearRegression

    n, p = X.shape
    model = LinearRegression().fit(X, y)
    r2 = model.score(X, y)
    y_pred = model.predict(X)
    residuals = y - y_pred
    if n - p - 1 > 0:
        mse = float(np.sum(residuals**2) / (n - p - 1))
    else:
        mse = float("nan")
    XtX_inv = np.linalg.inv(X.T @ X + 1e-10 * np.eye(p))
    se = np.sqrt(np.diag(mse * XtX_inv)) if mse == mse else np.full(p, np.nan)
    t_stats = model.coef_ / (se + 1e-12)
    p_values = [float(2.0 * (1.0 - sstats.t.cdf(abs(t), df=max(n - p - 1, 1)))) for t in t_stats]
    coefs = {}
    for i, ax in enumerate(axis_names):
        coefs[ax] = {
            "coef": float(model.coef_[i]),
            "se": float(se[i]),
            "t_stat": float(t_stats[i]),
            "p_value": p_values[i],
        }
    return {
        "axis_names": axis_names,
        "n_obs": int(n),
        "n_features": int(p),
        "r2_in_sample": float(r2),
        "intercept": float(model.intercept_),
        "coefficients": coefs,
    }


def spearman_correlations(df: pd.DataFrame) -> dict:
    out = {}
    for ax in AXIS_NAMES:
        rho, p = sstats.spearmanr(df[ax].values, df["marker_rate"].values)
        out[ax] = {"rho": float(rho), "p_value": float(p), "n": len(df)}
    return out


def partial_spearman(df: pd.DataFrame, x: str, y: str, control: str) -> dict:
    """Partial Spearman correlation between x and y after controlling for `control`.

    Residualize x and y against control via simple OLS-on-ranks, then Spearman.
    """
    rx = sstats.rankdata(df[x].values)
    ry = sstats.rankdata(df[y].values)
    rc = sstats.rankdata(df[control].values)
    # Residualize via OLS on ranks
    A = np.column_stack([np.ones_like(rc), rc])
    bx, *_ = np.linalg.lstsq(A, rx, rcond=None)
    by, *_ = np.linalg.lstsq(A, ry, rcond=None)
    ex = rx - A @ bx
    ey = ry - A @ by
    rho, p = sstats.spearmanr(ex, ey)
    return {
        "x": x,
        "y": y,
        "control": control,
        "rho": float(rho),
        "p_value": float(p),
        "n": len(df),
    }


def delta_r2(df: pd.DataFrame, full_axes: list[str], dropped: str) -> dict:
    """In-sample R^2 of full model minus the same model with `dropped` removed."""
    from sklearn.linear_model import LinearRegression

    base_axes = [a for a in full_axes if a != dropped]
    y = df["marker_rate"].values
    r2_full = LinearRegression().fit(df[full_axes].values, y).score(df[full_axes].values, y)
    r2_base = LinearRegression().fit(df[base_axes].values, y).score(df[base_axes].values, y)
    return {
        "full_axes": full_axes,
        "base_axes": base_axes,
        "dropped": dropped,
        "r2_full": float(r2_full),
        "r2_base": float(r2_base),
        "delta_r2": float(r2_full - r2_base),
        "n": len(df),
    }


def cv_r2(df: pd.DataFrame, axes: list[str]) -> dict:
    """Leave-one-trigger-out CV: train on 3 train_family, test on the 4th."""
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import LeaveOneGroupOut

    X = df[axes].values
    y = df["marker_rate"].values
    groups = df["condition"].values
    logo = LeaveOneGroupOut()
    fold_r2 = []
    for tr, te in logo.split(X, y, groups):
        m = LinearRegression().fit(X[tr], y[tr])
        fold_r2.append(float(m.score(X[te], y[te])))
    return {
        "axes": axes,
        "fold_r2": fold_r2,
        "mean_r2": float(np.mean(fold_r2)),
        "median_r2": float(np.median(fold_r2)),
        "n_folds": len(fold_r2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-out", default="eval_results/issue_207/js_gentle/regression_data.csv")
    parser.add_argument(
        "--json-out", default="eval_results/issue_207/js_gentle/regression_results.json"
    )
    args = parser.parse_args()

    panel_evals = load_panel_evals()
    if len(panel_evals) == 0:
        raise SystemExit("No panel evals found, aborting")
    js, js_prompts = load_js_matrix()
    df = build_dataframe(panel_evals, js, js_prompts)

    # Save CSV
    csv_path = PROJECT_ROOT / args.csv_out
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info("CSV saved -> %s (%d rows)", csv_path, len(df))

    # Sanity-check: 12 adapters × 32 cells = 384 expected
    n_adapters = df["run_name"].nunique()
    n_cells_per_adapter = df.groupby("run_name").size().describe().to_dict()
    logger.info("Adapters: %d, cells/adapter stats: %s", n_adapters, n_cells_per_adapter)

    # Headline regression
    y = df["marker_rate"].values
    X_full = df[AXIS_NAMES].values
    full_ols = fit_ols(X_full, y, AXIS_NAMES)

    # Spearman per-axis
    spearman = spearman_correlations(df)

    # Partial Spearman: js | lexical, and lexical | js
    partial_js_given_lex = partial_spearman(df, "js_div", "marker_rate", "lexical_jac")
    partial_lex_given_js = partial_spearman(df, "lexical_jac", "marker_rate", "js_div")

    # ΔR²
    dR2_js_over_baseline = delta_r2(df, AXIS_NAMES, dropped="js_div")
    dR2_lex_over_baseline = delta_r2(df, AXIS_NAMES, dropped="lexical_jac")
    dR2_sem_over_baseline = delta_r2(df, AXIS_NAMES, dropped="semantic_cos")

    # CV R^2
    cv_full = cv_r2(df, AXIS_NAMES)
    cv_no_js = cv_r2(df, [a for a in AXIS_NAMES if a != "js_div"])
    cv_no_lex = cv_r2(df, [a for a in AXIS_NAMES if a != "lexical_jac"])
    cv_no_sem = cv_r2(df, [a for a in AXIS_NAMES if a != "semantic_cos"])

    payload = {
        "n_rows": len(df),
        "n_adapters": int(n_adapters),
        "axes": AXIS_NAMES,
        "exclude_buckets": sorted(EXCLUDE_BUCKETS),
        "ols_5axis": full_ols,
        "spearman_per_axis": spearman,
        "partial_spearman": {
            "js_given_lexical": partial_js_given_lex,
            "lexical_given_js": partial_lex_given_js,
        },
        "delta_r2": {
            "js_div": dR2_js_over_baseline,
            "lexical_jac": dR2_lex_over_baseline,
            "semantic_cos": dR2_sem_over_baseline,
        },
        "cv_r2_logo": {
            "full_5axis": cv_full,
            "no_js": cv_no_js,
            "no_lexical": cv_no_lex,
            "no_semantic": cv_no_sem,
        },
        "metadata": {
            "computed_at": datetime.now(UTC).isoformat(),
            "git_commit": get_git_commit(),
            "panel_eval_files": sorted(panel_evals.keys()),
        },
    }

    json_path = PROJECT_ROOT / args.json_out
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2))
    logger.info("Regression results -> %s", json_path)

    # Headline summary to stdout
    logger.info("=" * 60)
    logger.info("HEADLINE")
    logger.info("=" * 60)
    logger.info("N = %d cells (%d adapters)", len(df), n_adapters)
    logger.info("Full 5-axis OLS in-sample R^2 = %.4f", full_ols["r2_in_sample"])
    for ax in AXIS_NAMES:
        c = full_ols["coefficients"][ax]
        logger.info(
            "  %-13s  coef=% .4f  p=%.4g  (Spearman rho=% .3f, p=%.4g)",
            ax,
            c["coef"],
            c["p_value"],
            spearman[ax]["rho"],
            spearman[ax]["p_value"],
        )
    logger.info(
        "ΔR^2 adding JS: %.4f (full %.4f - base %.4f)",
        dR2_js_over_baseline["delta_r2"],
        dR2_js_over_baseline["r2_full"],
        dR2_js_over_baseline["r2_base"],
    )
    logger.info("ΔR^2 adding lexical: %.4f", dR2_lex_over_baseline["delta_r2"])
    logger.info(
        "partial Spearman(JS, marker | lexical) = %.4f (p=%.4g)",
        partial_js_given_lex["rho"],
        partial_js_given_lex["p_value"],
    )
    logger.info(
        "partial Spearman(lexical, marker | JS) = %.4f (p=%.4g)",
        partial_lex_given_js["rho"],
        partial_lex_given_js["p_value"],
    )
    logger.info(
        "CV-R^2 logo: full=%.3f  no-JS=%.3f  no-lex=%.3f  no-sem=%.3f",
        cv_full["mean_r2"],
        cv_no_js["mean_r2"],
        cv_no_lex["mean_r2"],
        cv_no_sem["mean_r2"],
    )


if __name__ == "__main__":
    main()
