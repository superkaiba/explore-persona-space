#!/usr/bin/env python3
"""Analysis script for issue #181 non-persona trigger leakage.

Loads all panel evals, computes per-pair similarity features, fits OLS
regression, and generates figures.

Outputs:
  eval_results/i181_non_persona/regression_results.json
  figures/i181/heatmap_train_x_test_family.{png,pdf}
  figures/i181/axis_coefficients.{png,pdf}
  figures/i181/axis_correlation_matrix.{png,pdf}

Usage:
    uv run python scripts/analyze_i181.py --results-dir eval_results/i181_non_persona
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from explore_persona_space.analysis.i181_features import (
    compute_lexical_jaccard,
    compute_semantic_cosine,
    compute_struct_match,
    compute_structural_features,
)
from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "i181_non_persona"

# Trigger families for the 12 main runs (excludes controls)
MAIN_FAMILIES = ["T_task", "T_instruction", "T_context", "T_format"]
SEEDS = [42, 137, 256]
AXIS_NAMES = ["semantic_cos", "lexical_jac", "struct_match", "task_match"]


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_all_panel_evals(results_dir: Path) -> dict[str, dict]:
    """Load all panel_eval.json files from run subdirectories.

    Returns {run_name: panel_eval_dict}.
    """
    evals = {}
    if not results_dir.exists():
        logger.error("Results directory not found: %s", results_dir)
        return evals

    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        panel_path = run_dir / "panel_eval.json"
        if panel_path.exists():
            with open(panel_path) as f:
                evals[run_dir.name] = json.load(f)
            logger.info("Loaded panel eval: %s", run_dir.name)

    logger.info("Loaded %d panel evals total", len(evals))
    return evals


def load_panel_and_embeddings() -> tuple[list[dict], dict | None]:
    """Load eval panel and optional embeddings."""
    import torch

    panel_path = DATA_DIR / "eval_panel.json"
    emb_path = DATA_DIR / "system_prompt_embeddings.pt"

    with open(panel_path) as f:
        panel_data = json.load(f)

    embeddings = None
    if emb_path.exists():
        embeddings = torch.load(emb_path, weights_only=False)

    return panel_data["panel"], embeddings


def get_embedding(panel_id: str, embeddings: dict | None):
    """Look up embedding by panel ID."""

    if embeddings is None:
        return None
    ids = embeddings["ids"]
    emb_matrix = embeddings["embeddings"]
    # Direct match
    if panel_id in ids:
        return emb_matrix[ids.index(panel_id)]
    # Try train_ prefix for triggers
    train_id = f"train_{panel_id.replace('match_', '')}"
    if train_id in ids:
        return emb_matrix[ids.index(train_id)]
    return None


def build_regression_dataframe(
    panel_evals: dict[str, dict],
    panel: list[dict],
    embeddings: dict | None,
) -> pd.DataFrame:
    """Build a DataFrame with one row per (model, test_prompt) cell.

    Columns: run_name, condition, seed, train_family, test_id, test_family,
             test_bucket, marker_rate, semantic_cos, lexical_jac,
             struct_match, task_match
    """
    # Build lookup from panel_id to entry
    panel_lookup = {e["id"]: e for e in panel}

    # Load triggers for training prompt text
    triggers_path = DATA_DIR / "triggers.json"
    with open(triggers_path) as f:
        triggers_data = json.load(f)
    triggers = triggers_data["triggers"]
    trigger_families = triggers_data["families"]

    rows = []
    for run_name, eval_data in panel_evals.items():
        # Parse condition and seed from run_name (e.g., "T_format_seed42")
        parts = run_name.rsplit("_seed", 1)
        if len(parts) != 2:
            logger.warning("Cannot parse run_name: %s", run_name)
            continue
        condition = parts[0]
        try:
            seed = int(parts[1])
        except ValueError:
            logger.warning("Cannot parse seed from: %s", run_name)
            continue

        train_family = trigger_families.get(condition, "unknown")
        train_prompt = triggers.get(condition, "")

        # Get training prompt embedding
        train_emb = get_embedding(f"match_{condition}", embeddings)
        if train_emb is None:
            train_emb = get_embedding(f"train_{condition}", embeddings)

        # Compute structural features for training prompt
        train_feats = compute_structural_features(train_prompt)
        train_feats["task_type"] = train_family

        results = eval_data.get("results", {})
        for test_id, cell_data in results.items():
            test_entry = panel_lookup.get(test_id)
            if test_entry is None:
                continue

            test_prompt = test_entry["system_prompt"]
            test_family = test_entry.get("family") or "none"
            test_bucket = test_entry.get("bucket", "unknown")

            # Compute pair features
            # Semantic cosine
            test_emb = get_embedding(test_id, embeddings)
            if train_emb is not None and test_emb is not None:
                sem_cos = compute_semantic_cosine(train_emb, test_emb)
            else:
                sem_cos = 0.0

            # Lexical Jaccard
            lex_jac = compute_lexical_jaccard(train_prompt, test_prompt)

            # Structural match
            test_feats = compute_structural_features(test_prompt)
            test_feats["task_type"] = test_family
            struct = compute_struct_match(train_feats, test_feats)

            # Task match
            task = 1.0 if train_family == test_family else 0.0

            marker_rate = cell_data.get("marker_rate", 0.0)
            marker_position = cell_data.get("marker_position", {})

            rows.append(
                {
                    "run_name": run_name,
                    "condition": condition,
                    "seed": seed,
                    "train_family": train_family,
                    "test_id": test_id,
                    "test_family": test_family,
                    "test_bucket": test_bucket,
                    "marker_rate": marker_rate,
                    "semantic_cos": sem_cos,
                    "lexical_jac": lex_jac,
                    "struct_match": struct,
                    "task_match": task,
                    "marker_pos_suffix": marker_position.get("suffix", 0),
                    "marker_pos_inside_json": marker_position.get("inside_json", 0),
                    "marker_pos_inside_text": marker_position.get("inside_text", 0),
                    "marker_pos_absent": marker_position.get("absent", 0),
                }
            )

    df = pd.DataFrame(rows)
    logger.info("Built regression DataFrame: %d rows", len(df))
    return df


def fit_ols_regression(df: pd.DataFrame) -> dict:
    """Fit OLS marker_rate ~ 4 axes on non-matched cells from main runs.

    Returns regression results dict.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import LeaveOneGroupOut

    # Filter to main runs, non-matched cells
    main_conditions = MAIN_FAMILIES
    mask = df["condition"].isin(main_conditions) & (df["test_bucket"] != "matched")
    df_reg = df[mask].copy()

    if len(df_reg) == 0:
        logger.warning("No data for regression after filtering")
        return {"error": "No data for regression"}

    X = df_reg[AXIS_NAMES].values
    y = df_reg["marker_rate"].values
    groups = df_reg["condition"].values  # For leave-one-trigger-out CV

    # Full model
    full_model = LinearRegression()
    full_model.fit(X, y)
    full_r2 = full_model.score(X, y)

    # Coefficients with p-values (using scipy for inference)
    n = len(y)
    p = X.shape[1]
    y_pred = full_model.predict(X)
    residuals = y - y_pred
    mse = np.sum(residuals**2) / (n - p - 1)
    # Standard errors
    XtX_inv = np.linalg.inv(X.T @ X + 1e-10 * np.eye(p))
    se = np.sqrt(np.diag(mse * XtX_inv))
    t_stats = full_model.coef_ / (se + 1e-12)
    p_values = [2 * (1 - stats.t.cdf(abs(t), df=n - p - 1)) for t in t_stats]

    coefficients = {}
    for i, axis in enumerate(AXIS_NAMES):
        coefficients[axis] = {
            "coef": float(full_model.coef_[i]),
            "se": float(se[i]),
            "t_stat": float(t_stats[i]),
            "p_value": float(p_values[i]),
        }

    # Leave-one-trigger-out CV
    logo = LeaveOneGroupOut()
    cv_r2_full = []
    cv_r2_semantic_only = []
    cv_r2_lexical_only = []
    cv_r2_struct_only = []
    cv_r2_task_only = []

    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Full model
        m = LinearRegression().fit(X_train, y_train)
        cv_r2_full.append(m.score(X_test, y_test))

        # Single-axis baselines
        for ax_idx, ax_name in enumerate(AXIS_NAMES):
            X_tr_ax = X_train[:, ax_idx : ax_idx + 1]
            X_te_ax = X_test[:, ax_idx : ax_idx + 1]
            m_ax = LinearRegression().fit(X_tr_ax, y_train)
            r2 = m_ax.score(X_te_ax, y_test)
            if ax_name == "semantic_cos":
                cv_r2_semantic_only.append(r2)
            elif ax_name == "lexical_jac":
                cv_r2_lexical_only.append(r2)
            elif ax_name == "struct_match":
                cv_r2_struct_only.append(r2)
            elif ax_name == "task_match":
                cv_r2_task_only.append(r2)

    mean_cv_r2_full = float(np.mean(cv_r2_full))
    mean_cv_r2_semantic = float(np.mean(cv_r2_semantic_only))
    mean_cv_r2_lexical = float(np.mean(cv_r2_lexical_only))
    mean_cv_r2_struct = float(np.mean(cv_r2_struct_only))
    mean_cv_r2_task = float(np.mean(cv_r2_task_only))

    cv_r2_gap = mean_cv_r2_full - mean_cv_r2_semantic

    # Null permutation test for CV R^2 gap
    logger.info("Running permutation test for CV R^2 gap (1000 shuffles)...")
    n_permutations = 1000
    null_gaps = []
    rng = np.random.default_rng(42)

    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        perm_cv_full = []
        perm_cv_sem = []
        for train_idx, test_idx in logo.split(X, y_perm, groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_perm[train_idx], y_perm[test_idx]

            m_full = LinearRegression().fit(X_train, y_train)
            perm_cv_full.append(m_full.score(X_test, y_test))

            m_sem = LinearRegression().fit(X_train[:, 0:1], y_train)
            perm_cv_sem.append(m_sem.score(X_test[:, 0:1], y_test))

        null_gaps.append(np.mean(perm_cv_full) - np.mean(perm_cv_sem))

    null_gaps = np.array(null_gaps)
    null_p95 = float(np.percentile(null_gaps, 95))
    perm_p_value = float(np.mean(null_gaps >= cv_r2_gap))

    regression_results = {
        "full_model": {
            "r2": float(full_r2),
            "intercept": float(full_model.intercept_),
            "coefficients": coefficients,
            "n_observations": int(n),
        },
        "cv_r2": {
            "full": mean_cv_r2_full,
            "semantic_only": mean_cv_r2_semantic,
            "lexical_only": mean_cv_r2_lexical,
            "structural_only": mean_cv_r2_struct,
            "task_type_only": mean_cv_r2_task,
            "gap_full_minus_semantic": cv_r2_gap,
        },
        "permutation_test": {
            "n_permutations": n_permutations,
            "null_p95": null_p95,
            "p_value": perm_p_value,
            "gap_exceeds_null_p95": bool(cv_r2_gap > null_p95),
        },
    }

    return regression_results


def test_h1(df: pd.DataFrame) -> dict:
    """Test H1: matched marker rate >= 3x bystander rate.

    Uses main runs only (12 conditions, 3 seeds each).
    """
    main_mask = df["condition"].isin(MAIN_FAMILIES)
    df_main = df[main_mask]

    # Per-seed stats
    matched_rates = []
    bystander_rates = []

    for seed in SEEDS:
        seed_mask = df_main["seed"] == seed

        # Matched cells
        matched = df_main[seed_mask & (df_main["test_bucket"] == "matched")]
        if len(matched) > 0:
            matched_rates.append(matched["marker_rate"].mean())

        # Bystander + control cells
        bystander = df_main[
            seed_mask
            & df_main["test_bucket"].isin(
                ["cross_family_bystander", "control_empty", "control_default"]
            )
        ]
        if len(bystander) > 0:
            bystander_rates.append(bystander["marker_rate"].mean())

    r_match = float(np.mean(matched_rates)) if matched_rates else 0.0
    r_bystander = float(np.mean(bystander_rates)) if bystander_rates else 0.0
    ratio = r_match / r_bystander if r_bystander > 0 else float("inf")

    # Permutation test — permute matched/bystander labels WITHIN each seed
    n_permutations = 10000
    rng = np.random.default_rng(42)

    bystander_buckets = ["cross_family_bystander", "control_empty", "control_default"]

    # Compute observed per-seed diffs, then pool
    per_seed_data = []
    for seed in SEEDS:
        seed_mask = df_main["seed"] == seed
        m = df_main[seed_mask & (df_main["test_bucket"] == "matched")]["marker_rate"].values
        b = df_main[seed_mask & df_main["test_bucket"].isin(bystander_buckets)][
            "marker_rate"
        ].values
        if len(m) > 0 and len(b) > 0:
            per_seed_data.append((m, b))

    if not per_seed_data:
        p_value = 1.0
    else:
        observed_diff = float(np.mean([np.mean(m) - np.mean(b) for m, b in per_seed_data]))

        null_diffs = []
        for _ in range(n_permutations):
            perm_diff_per_seed = []
            for m, b in per_seed_data:
                combined = np.concatenate([m, b])
                perm = rng.permutation(combined)
                n_m = len(m)
                perm_diff_per_seed.append(np.mean(perm[:n_m]) - np.mean(perm[n_m:]))
            null_diffs.append(np.mean(perm_diff_per_seed))

        null_diffs = np.array(null_diffs)
        p_value = float(np.mean(null_diffs >= observed_diff))

    # Also collect flat arrays for reporting
    matched_cells = df_main[df_main["test_bucket"] == "matched"]["marker_rate"].values
    bystander_cells = df_main[df_main["test_bucket"].isin(bystander_buckets)]["marker_rate"].values

    return {
        "r_match": r_match,
        "r_bystander": r_bystander,
        "ratio": ratio,
        "passes_3x": bool(ratio >= 3.0),
        "p_value": p_value,
        "p_below_001": bool(p_value < 0.01),
        "n_matched_cells": len(matched_cells),
        "n_bystander_cells": len(bystander_cells),
    }


def test_h3(df: pd.DataFrame, panel_data: dict) -> dict:
    """Test H3: within-family specificity for format trigger.

    Pre-registered contrast: for train_family=format, fammate_format_{1,2,3}
    vs fammate_task_{1,2,3} after partialing out semantic_cos via an
    interaction-coefficient regression.

    The regression:
        marker_rate ~ semantic_cos + is_format_family + semantic_cos:is_format_family

    where is_format_family = 1 for format family-mates, 0 for task family-mates.
    H3 is confirmed if the is_format_family coefficient is positive with p < 0.05
    (one-sided).
    """
    import statsmodels.api as sm

    # Get pre-registered cells
    panel_path = DATA_DIR / "eval_panel.json"
    with open(panel_path) as f:
        panel_doc = json.load(f)

    h3_contrast = panel_doc.get("h3_contrast_cells", {})
    format_cells = h3_contrast.get("format_cells", [])
    task_cells = h3_contrast.get("task_cells", [])

    if not format_cells or not task_cells:
        return {"error": "H3 contrast cells not found in eval_panel.json"}

    # Filter to T_format models only, and only the pre-registered cells
    mask_model = df["condition"] == "T_format"
    mask_cells = df["test_id"].isin(format_cells + task_cells)
    df_h3 = df[mask_model & mask_cells].copy()

    if len(df_h3) == 0:
        return {"error": "No data for H3 contrast cells"}

    # Create binary indicator: 1 = format family-mate, 0 = task family-mate
    df_h3["is_format_family"] = df_h3["test_id"].isin(format_cells).astype(float)

    if "semantic_cos" not in df_h3.columns:
        return {"error": "semantic_cos column missing; run feature computation first"}

    # Regression: marker_rate ~ semantic_cos + is_format_family
    X = df_h3[["semantic_cos", "is_format_family"]].copy()
    X = sm.add_constant(X)
    y = df_h3["marker_rate"]

    try:
        model = sm.OLS(y, X).fit()
    except Exception as e:
        return {"error": f"OLS fit failed: {e}"}

    # The coefficient on is_format_family is the format-specificity effect
    # after partialing out semantic_cos
    coef = float(model.params["is_format_family"])
    p_two_sided = float(model.pvalues["is_format_family"])
    # One-sided: H3 predicts format > task, so coef > 0
    p_one_sided = p_two_sided / 2 if coef > 0 else 1.0 - p_two_sided / 2

    mean_format = float(df_h3[df_h3["is_format_family"] == 1]["marker_rate"].mean())
    mean_task = float(df_h3[df_h3["is_format_family"] == 0]["marker_rate"].mean())

    return {
        "mean_format_cells": mean_format,
        "mean_task_cells": mean_task,
        "raw_difference": mean_format - mean_task,
        "coef_is_format_family": coef,
        "p_value_two_sided": p_two_sided,
        "p_value_one_sided": p_one_sided,
        "passes_p05": bool(p_one_sided < 0.05),
        "semantic_cos_coef": float(model.params["semantic_cos"]),
        "r_squared": float(model.rsquared),
        "n_total": len(df_h3),
        "n_format": int(df_h3["is_format_family"].sum()),
        "n_task": int(len(df_h3) - df_h3["is_format_family"].sum()),
        "format_cells": format_cells,
        "task_cells": task_cells,
    }


def check_no_marker_control(df: pd.DataFrame) -> dict:
    """Check H1-aux: T_task_no_marker marker rate <= 1% everywhere."""
    mask = df["condition"] == "T_task_no_marker"
    df_ctrl = df[mask]

    if len(df_ctrl) == 0:
        return {"status": "NOT_RUN", "reason": "No T_task_no_marker data"}

    max_rate = float(df_ctrl["marker_rate"].max())
    mean_rate = float(df_ctrl["marker_rate"].mean())

    return {
        "status": "PASS" if max_rate <= 0.01 else "FAIL",
        "max_marker_rate": max_rate,
        "mean_marker_rate": mean_rate,
        "n_cells": len(df_ctrl),
        "threshold": 0.01,
    }


# ── Figures ──────────────────────────────────────────────────────────────────


def plot_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot train_family x test_family marker rate heatmap."""
    set_paper_style("neurips")

    # Aggregate: mean marker rate per (train_family, test_family)
    # Include controls and persona anchor
    agg = df.groupby(["train_family", "test_family"])["marker_rate"].mean().reset_index()
    pivot = agg.pivot(index="train_family", columns="test_family", values="marker_rate")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1%",
        cmap="YlOrRd",
        ax=ax,
        vmin=0,
        cbar_kws={"label": "Marker rate"},
    )
    ax.set_xlabel("Test family")
    ax.set_ylabel("Train family")
    ax.set_title("Marker leakage: train family x test family")

    saved = savefig_paper(fig, "i181/heatmap_train_x_test_family", dir=str(output_dir))
    plt.close(fig)
    logger.info("Saved heatmap to %s", saved.get("png", ""))


def plot_axis_coefficients(regression_results: dict, output_dir: Path) -> None:
    """Plot bar chart of regression axis coefficients."""
    set_paper_style("neurips")

    coeffs = regression_results.get("full_model", {}).get("coefficients", {})
    if not coeffs:
        logger.warning("No coefficients to plot")
        return

    names = list(coeffs.keys())
    values = [coeffs[n]["coef"] for n in names]
    errors = [coeffs[n]["se"] for n in names]
    p_vals = [coeffs[n]["p_value"] for n in names]

    colors = paper_palette(len(names))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(names)), values, yerr=errors, capsize=5, color=colors, alpha=0.8)

    # Add significance stars
    for i, (v, p) in enumerate(zip(values, p_vals, strict=True)):
        star = ""
        if p < 0.001:
            star = "***"
        elif p < 0.01:
            star = "**"
        elif p < 0.05:
            star = "*"
        if star:
            y_pos = v + errors[i] + 0.002
            ax.text(i, y_pos, star, ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=9)
    ax.set_ylabel("Regression coefficient")
    ax.set_title("Axis contributions to marker leakage")
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")

    fig.tight_layout()
    saved = savefig_paper(fig, "i181/axis_coefficients", dir=str(output_dir))
    plt.close(fig)
    logger.info("Saved coefficient plot to %s", saved.get("png", ""))


def plot_cv_r2_comparison(regression_results: dict, output_dir: Path) -> None:
    """Plot bar chart comparing CV R^2 across axis subsets."""
    set_paper_style("neurips")

    cv_r2 = regression_results.get("cv_r2", {})
    if not cv_r2:
        return

    labels = [
        "Full\nmodel",
        "Semantic\nonly",
        "Lexical\nonly",
        "Structural\nonly",
        "Task-type\nonly",
    ]
    values = [
        cv_r2.get("full", 0),
        cv_r2.get("semantic_only", 0),
        cv_r2.get("lexical_only", 0),
        cv_r2.get("structural_only", 0),
        cv_r2.get("task_type_only", 0),
    ]

    colors = paper_palette(5)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(labels)), values, color=colors, alpha=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Leave-one-trigger-out CV R²")
    ax.set_title("Predictive power: full model vs single-axis baselines")

    fig.tight_layout()
    saved = savefig_paper(fig, "i181/cv_r2_comparison", dir=str(output_dir))
    plt.close(fig)
    logger.info("Saved CV R² comparison to %s", saved.get("png", ""))


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Analyze issue #181 results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="eval_results/i181_non_persona",
        help="Directory containing panel eval results",
    )
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / args.results_dir

    # Load data
    panel_evals = load_all_panel_evals(results_dir)
    if not panel_evals:
        logger.error("No panel evals found in %s", results_dir)
        sys.exit(1)

    panel, embeddings = load_panel_and_embeddings()

    # Build regression DataFrame
    df = build_regression_dataframe(panel_evals, panel, embeddings)
    if df.empty:
        logger.error("Empty regression DataFrame")
        sys.exit(1)

    # Save DataFrame for inspection
    df.to_csv(results_dir / "regression_data.csv", index=False)
    logger.info("Saved regression data to %s", results_dir / "regression_data.csv")

    # Run tests
    logger.info("\n" + "=" * 70)
    logger.info("H1: Matched vs bystander gradient")
    logger.info("=" * 70)
    h1_results = test_h1(df)
    logger.info(
        "H1: r_match=%.1f%%, r_bystander=%.1f%%, ratio=%.1fx, p=%.4f",
        h1_results["r_match"] * 100,
        h1_results["r_bystander"] * 100,
        h1_results["ratio"],
        h1_results["p_value"],
    )
    logger.info("H1 passes: 3x=%s, p<0.01=%s", h1_results["passes_3x"], h1_results["p_below_001"])

    logger.info("\n" + "=" * 70)
    logger.info("H1-aux: No-marker control check")
    logger.info("=" * 70)
    h1_aux = check_no_marker_control(df)
    logger.info(
        "H1-aux: %s (max_rate=%.3f%%)", h1_aux["status"], h1_aux.get("max_marker_rate", 0) * 100
    )

    logger.info("\n" + "=" * 70)
    logger.info("H2: Regression analysis")
    logger.info("=" * 70)
    regression_results = fit_ols_regression(df)

    if "error" not in regression_results:
        cv = regression_results["cv_r2"]
        logger.info("Full model CV R²: %.4f", cv["full"])
        logger.info("Semantic-only CV R²: %.4f", cv["semantic_only"])
        logger.info("Lexical-only CV R²: %.4f", cv["lexical_only"])
        logger.info("Structural-only CV R²: %.4f", cv["structural_only"])
        logger.info("Task-type-only CV R²: %.4f", cv["task_type_only"])
        logger.info("Gap (full - semantic): %.4f", cv["gap_full_minus_semantic"])
        perm = regression_results["permutation_test"]
        logger.info(
            "Permutation test: null_p95=%.4f, p=%.4f, gap > null_p95=%s",
            perm["null_p95"],
            perm["p_value"],
            perm["gap_exceeds_null_p95"],
        )

        for axis, coeff_data in regression_results["full_model"]["coefficients"].items():
            logger.info(
                "  %s: coef=%.4f, se=%.4f, p=%.4f",
                axis,
                coeff_data["coef"],
                coeff_data["se"],
                coeff_data["p_value"],
            )

    logger.info("\n" + "=" * 70)
    logger.info("H3: Within-family specificity (format contrast)")
    logger.info("=" * 70)
    h3_results = test_h3(df, {})
    if "error" not in h3_results:
        logger.info(
            "H3: format=%.1f%%, task=%.1f%%, diff=%.1f%%, p=%.4f",
            h3_results["mean_format_cells"] * 100,
            h3_results["mean_task_cells"] * 100,
            h3_results["raw_difference"] * 100,
            h3_results["p_value_one_sided"],
        )

    # Aggregate results
    analysis_results = {
        "analyzed_at": datetime.now(UTC).isoformat(),
        "git_commit": get_git_commit(),
        "n_panel_evals": len(panel_evals),
        "n_regression_rows": len(df),
        "h1": h1_results,
        "h1_aux_no_marker": h1_aux,
        "h2_regression": regression_results,
        "h3_specificity": h3_results,
        "marker_position_summary": _aggregate_marker_positions(df),
    }

    # Save results
    from explore_persona_space.utils import save_json_atomic

    output_path = results_dir / "regression_results.json"
    save_json_atomic(output_path, analysis_results)
    logger.info("Saved regression results to %s", output_path)

    # Generate figures
    figures_dir = PROJECT_ROOT / "figures"
    logger.info("\nGenerating figures...")
    plot_heatmap(df, figures_dir)
    if "error" not in regression_results:
        plot_axis_coefficients(regression_results, figures_dir)
        plot_cv_r2_comparison(regression_results, figures_dir)

    logger.info("\nAnalysis complete.")


def _aggregate_marker_positions(df: pd.DataFrame) -> dict:
    """Aggregate marker position breakdown per condition."""
    pos_cols = [
        "marker_pos_suffix",
        "marker_pos_inside_json",
        "marker_pos_inside_text",
        "marker_pos_absent",
    ]
    available = [c for c in pos_cols if c in df.columns]
    if not available:
        return {}
    return df.groupby("condition")[available].sum().to_dict()


if __name__ == "__main__":
    main()
