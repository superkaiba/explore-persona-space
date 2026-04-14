#!/usr/bin/env python3
"""Analyze results from the Comprehensive Persona Trait Leakage Experiment.

Reads evaluation results from eval_results/leakage_experiment/ and produces
publication-quality figures + statistical tests.

Analyses implemented (Section 7 of plan):
  7a. Distance -> Leakage per trait (4-panel scatter + regression)
  7b. Assistant Shielding (residual below regression)
  7c. Prompt-Length Effect (Phase B, if data exists)
  7d. Negative-Set Effect (asst_excluded vs asst_included)
  7e. SFT Controls (generic_sft / shuffled_persona baselines)
  7f. Training Dynamics (leakage-vs-step curves, Phase 0.5)
  7g. Capability ID vs OOD

Usage:
    uv run python scripts/analyze_leakage.py
    uv run python scripts/analyze_leakage.py --phase pilot
    uv run python scripts/analyze_leakage.py --trait marker
    uv run python scripts/analyze_leakage.py --phase a1 --trait marker
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
from scipy import stats

matplotlib.use("Agg")

# ── Style ─────────────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid")
plt.rcParams.update(
    {
        "font.size": 12,
        "font.family": "sans-serif",
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
    }
)

# Colorblind-friendly palette (Okabe-Ito)
C_SOURCE = "#0072B2"  # blue -- source persona
C_HELD = "#56B4E9"  # sky blue -- held-out personas
C_ASST = "#D55E00"  # vermillion -- assistant
C_NEG = "#E69F00"  # orange -- negative-set member
C_REG = "#999999"  # grey -- regression line
C_CONTROL = "#009E73"  # teal -- control conditions
C_DYNAMICS_SOURCE = "#CC79A7"  # rose -- dynamics source
C_DYNAMICS_ASST = "#D55E00"  # vermillion -- dynamics assistant

TRAIT_COLORS = {
    "marker": "#0072B2",
    "structure": "#009E73",
    "capability": "#E69F00",
    "misalignment": "#D55E00",
}

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "eval_results" / "leakage_experiment"
FIG_DIR = PROJECT_ROOT / "figures" / "leakage_experiment"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Cosine distances (mean-centered, layer 10) ───────────────────────────────
# Source: persona vectors extracted from Qwen2.5-7B-Instruct, layer 10,
# global-mean subtracted. Verified values from the experiment plan.

CENTERED_COSINES = {
    "software_engineer": +0.446,
    "kindergarten_teacher": +0.331,
    "data_scientist": +0.170,
    "medical_doctor": +0.054,
    "librarian": -0.081,
    "french_person": -0.226,
    "villain": -0.237,
    "comedian": -0.283,
    "police_officer": -0.399,
    "zelthari_scholar": -0.379,
}

# Try to load zelthari from computed file (may be more precise)
_zelthari_path = EVAL_DIR / "zelthari_centered_cosine.json"
if _zelthari_path.exists():
    with open(_zelthari_path) as _f:
        _zd = json.load(_f)
        if "layer_10" in _zd:
            CENTERED_COSINES["zelthari_scholar"] = _zd["layer_10"]["centered_cosine_to_assistant"]

# Short display names for plot labels
SHORT_NAMES = {
    "software_engineer": "SWE",
    "kindergarten_teacher": "K-Teacher",
    "data_scientist": "Data Sci",
    "medical_doctor": "Doctor",
    "librarian": "Librarian",
    "french_person": "French",
    "villain": "Villain",
    "comedian": "Comedian",
    "police_officer": "Police",
    "zelthari_scholar": "Zelthari",
    "assistant": "Assistant",
}

ALL_TRAITS = ["marker", "structure", "capability", "misalignment"]
ALL_SOURCES = list(CENTERED_COSINES.keys())
PROMPT_LENGTHS = ["short", "medium", "long"]
NEG_SETS = ["asst_excluded", "asst_included"]
CONTROL_CONDITIONS = ["generic_sft", "shuffled_persona"]


# ── Data loading ──────────────────────────────────────────────────────────────


def parse_run_dir_name(name: str) -> dict[str, str] | None:
    """Parse a run directory name into its components.

    Standard: {trait}_{source}_{neg_set}_{prompt_length}_seed{N}
    Control:  {trait}_{control}_seed{N}
    """
    # Try standard pattern
    m = re.match(
        r"^(?P<trait>[a-z]+)_(?P<source>[a-z_]+?)_"
        r"(?P<neg_set>asst_excluded|asst_included)_"
        r"(?P<prompt_length>short|medium|long)_"
        r"seed(?P<seed>\d+)$",
        name,
    )
    if m:
        return {
            "trait": m.group("trait"),
            "source": m.group("source"),
            "neg_set": m.group("neg_set"),
            "prompt_length": m.group("prompt_length"),
            "seed": int(m.group("seed")),
            "control": None,
        }

    # Try control pattern
    m = re.match(
        r"^(?P<trait>[a-z]+)_(?P<control>generic_sft|shuffled_persona)_" r"seed(?P<seed>\d+)$",
        name,
    )
    if m:
        return {
            "trait": m.group("trait"),
            "control": m.group("control"),
            "seed": int(m.group("seed")),
            "source": None,
            "neg_set": None,
            "prompt_length": None,
        }

    return None


def load_json_safe(path: Path) -> dict | None:
    """Load JSON, returning None if file is missing or corrupt."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def load_all_results() -> list[dict[str, Any]]:
    """Scan eval_results/leakage_experiment/ and load all run data.

    Returns a list of dicts, each containing parsed metadata + eval results.
    """
    if not EVAL_DIR.exists():
        print(f"WARNING: {EVAL_DIR} does not exist. No data to analyze.")
        return []

    runs = []
    for run_dir in sorted(EVAL_DIR.iterdir()):
        if not run_dir.is_dir():
            continue

        meta = parse_run_dir_name(run_dir.name)
        if meta is None:
            continue

        entry = {
            "dir": run_dir,
            "name": run_dir.name,
            **meta,
            "train_result": load_json_safe(run_dir / "train_result.json"),
            "marker_eval": load_json_safe(run_dir / "marker_eval.json"),
            "structure_eval": load_json_safe(run_dir / "structure_eval.json"),
            "capability_eval": load_json_safe(run_dir / "capability_eval.json"),
            "alignment_eval": load_json_safe(run_dir / "alignment_eval.json"),
        }

        # Load dynamics if present
        dynamics_path = run_dir / "dynamics" / "checkpoint_dynamics.json"
        entry["dynamics"] = load_json_safe(dynamics_path)

        runs.append(entry)

    print(f"Loaded {len(runs)} run directories from {EVAL_DIR}")
    return runs


def filter_runs(
    runs: list[dict],
    trait: str | None = None,
    source: str | None = None,
    neg_set: str | None = None,
    prompt_length: str | None = None,
    seed: int | None = None,
    control: str | None = None,
    exclude_controls: bool = False,
) -> list[dict]:
    """Filter run list by metadata fields."""
    out = runs
    if trait is not None:
        out = [r for r in out if r["trait"] == trait]
    if source is not None:
        out = [r for r in out if r["source"] == source]
    if neg_set is not None:
        out = [r for r in out if r["neg_set"] == neg_set]
    if prompt_length is not None:
        out = [r for r in out if r["prompt_length"] == prompt_length]
    if seed is not None:
        out = [r for r in out if r["seed"] == seed]
    if control is not None:
        out = [r for r in out if r["control"] == control]
    if exclude_controls:
        out = [r for r in out if r["control"] is None]
    return out


# ── Leakage extraction helpers ────────────────────────────────────────────────


def get_leakage_rate(run: dict, trait: str, persona: str) -> float | None:
    """Extract the leakage rate for a given persona from a run's eval results.

    For marker: marker_eval -> persona -> rate
    For structure: structure_eval -> persona -> rate
    For capability: capability_eval -> arc_challenge_logprob (persona-independent)
    For misalignment: alignment_eval -> overall_mean_aligned (inverted: 100 - score)
    """
    if trait == "marker":
        data = run.get("marker_eval")
        if data is None or persona not in data:
            return None
        return data[persona].get("rate")

    if trait == "structure":
        data = run.get("structure_eval")
        if data is None or persona not in data:
            return None
        return data[persona].get("rate")

    if trait == "capability":
        data = run.get("capability_eval")
        if data is None or "error" in data:
            return None
        # Capability loss = 1 - accuracy (higher = more degradation = more "leakage")
        acc = data.get("arc_challenge_logprob")
        if acc is None:
            return None
        return 1.0 - acc

    if trait == "misalignment":
        data = run.get("alignment_eval")
        if data is None or "error" in data:
            return None
        # Misalignment leakage = 100 - alignment score (higher = more misaligned)
        score = data.get("overall_mean_aligned")
        if score is None:
            # Try per-persona
            if persona in data and isinstance(data[persona], dict):
                score = data[persona].get("mean_score")
            if score is None:
                return None
        return (100.0 - score) / 100.0  # Normalize to [0, 1]

    return None


def get_assistant_leakage(run: dict, trait: str) -> float | None:
    """Shortcut for assistant persona leakage."""
    return get_leakage_rate(run, trait, "assistant")


def get_source_leakage(run: dict) -> float | None:
    """Get leakage for the source persona (sanity check that training worked)."""
    if run["source"] is None:
        return None
    return get_leakage_rate(run, run["trait"], run["source"])


# ── Statistical helpers ───────────────────────────────────────────────────────


def pearson_with_ci(
    x: np.ndarray, y: np.ndarray, alpha: float = 0.05
) -> tuple[float, float, float, float]:
    """Pearson r with 95% CI via Fisher z-transform.

    Returns (r, p, ci_lower, ci_upper).
    """
    n = len(x)
    if n < 3:
        return np.nan, np.nan, np.nan, np.nan
    r, p = stats.pearsonr(x, y)
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lo = np.tanh(z - z_crit * se)
    ci_hi = np.tanh(z + z_crit * se)
    return r, p, ci_lo, ci_hi


def spearman_with_ci(
    x: np.ndarray, y: np.ndarray, alpha: float = 0.05
) -> tuple[float, float, float, float]:
    """Spearman rho with 95% CI via bootstrap (1000 resamples).

    Returns (rho, p, ci_lower, ci_upper).
    """
    n = len(x)
    if n < 3:
        return np.nan, np.nan, np.nan, np.nan
    rho, p = stats.spearmanr(x, y)

    rng = np.random.default_rng(42)
    boot_rhos = []
    for _ in range(1000):
        idx = rng.choice(n, size=n, replace=True)
        br, _ = stats.spearmanr(x[idx], y[idx])
        boot_rhos.append(br)
    boot_rhos = np.array(boot_rhos)
    ci_lo = np.nanpercentile(boot_rhos, 100 * alpha / 2)
    ci_hi = np.nanpercentile(boot_rhos, 100 * (1 - alpha / 2))
    return rho, p, ci_lo, ci_hi


def ols_regression(x: np.ndarray, y: np.ndarray) -> dict:
    """Simple OLS regression y = a + bx with prediction CI.

    Returns dict with slope, intercept, residuals, prediction bands.
    """
    n = len(x)
    if n < 3:
        return {"slope": np.nan, "intercept": np.nan}
    slope, intercept, r, p, se = stats.linregress(x, y)
    y_pred = slope * x + intercept
    residuals = y - y_pred
    mse = np.sum(residuals**2) / (n - 2)
    x_mean = np.mean(x)
    ss_x = np.sum((x - x_mean) ** 2)

    # For prediction band
    x_grid = np.linspace(x.min() - 0.05, x.max() + 0.05, 100)
    y_grid = slope * x_grid + intercept
    t_crit = stats.t.ppf(0.975, n - 2)
    ci_half = t_crit * np.sqrt(mse * (1.0 / n + (x_grid - x_mean) ** 2 / ss_x))

    return {
        "slope": slope,
        "intercept": intercept,
        "r": r,
        "p": p,
        "se": se,
        "residuals": residuals,
        "y_pred": y_pred,
        "mse": mse,
        "x_grid": x_grid,
        "y_grid": y_grid,
        "ci_lower": y_grid - ci_half,
        "ci_upper": y_grid + ci_half,
    }


# ── 7a. Distance -> Leakage (per trait type) ─────────────────────────────────


def _collect_trait_data(runs: list[dict], trait: str) -> dict | None:
    """Collect per-source leakage data for a single trait.

    Returns dict with sources, cosines, asst_leakages, source_leakages,
    or None if insufficient data.
    """
    trait_runs = filter_runs(
        runs,
        trait=trait,
        neg_set="asst_excluded",
        prompt_length="medium",
        exclude_controls=True,
    )
    if not trait_runs:
        print(f"  SKIP {trait}: no runs with neg_set=asst_excluded, prompt_length=medium")
        return None

    sources = []
    cosines = []
    asst_leakages = []
    source_leakages = []

    for source in ALL_SOURCES:
        source_runs = filter_runs(trait_runs, source=source)
        if not source_runs:
            continue

        asst_rates = [
            al for r in source_runs if (al := get_assistant_leakage(r, trait)) is not None
        ]
        src_rates = [sl for r in source_runs if (sl := get_source_leakage(r)) is not None]

        if not asst_rates:
            continue

        sources.append(source)
        cosines.append(CENTERED_COSINES[source])
        asst_leakages.append(np.mean(asst_rates))
        source_leakages.append(np.mean(src_rates) if src_rates else np.nan)

    if len(sources) < 3:
        print(f"  SKIP {trait}: only {len(sources)} sources with data (need >=3)")
        return None

    return {
        "sources": sources,
        "cosines": np.array(cosines),
        "asst_leakages": np.array(asst_leakages),
        "source_leakages": np.array(source_leakages),
    }


def analysis_7a_distance_leakage(runs: list[dict], traits: list[str]) -> dict[str, Any]:
    """4-panel figure: one per trait, leakage vs centered cosine.

    Uses default neg_set=asst_excluded, prompt_length=medium.
    """
    print("\n" + "=" * 70)
    print("7a. Distance -> Leakage (per trait type)")
    print("=" * 70)

    available_traits = []
    trait_data: dict[str, dict] = {}

    for trait in traits:
        td = _collect_trait_data(runs, trait)
        if td is None:
            continue
        available_traits.append(trait)
        trait_data[trait] = td

    if not available_traits:
        print("  SKIP 7a entirely: no traits with sufficient data")
        return {}

    # Build figure
    n_panels = len(available_traits)
    ncols = min(n_panels, 2)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6.5 * nrows), squeeze=False)

    summary = {}

    for idx, trait in enumerate(available_traits):
        ax = axes[idx // ncols, idx % ncols]
        td = trait_data[trait]

        x = td["cosines"]
        y = td["asst_leakages"] * 100  # percentage

        # Scatter: each source persona
        for i, src in enumerate(td["sources"]):
            is_villain = src == "villain"
            ax.scatter(
                x[i],
                y[i],
                marker="D" if is_villain else "o",
                s=80,
                c=TRAIT_COLORS.get(trait, C_HELD),
                edgecolors="black",
                linewidths=0.6,
                zorder=4,
            )
            label = SHORT_NAMES.get(src, src)
            ax.annotate(
                label,
                (x[i], y[i]),
                textcoords="offset points",
                xytext=(6, 5),
                fontsize=8,
                alpha=0.85,
            )

        # Regression line + CI band
        reg = ols_regression(x, y)
        if not np.isnan(reg["slope"]):
            ax.plot(
                reg["x_grid"],
                reg["y_grid"],
                "--",
                color=C_REG,
                lw=1.5,
                alpha=0.7,
                zorder=2,
            )
            ax.fill_between(
                reg["x_grid"],
                reg["ci_lower"],
                reg["ci_upper"],
                color=C_REG,
                alpha=0.15,
                zorder=1,
            )

        # Correlation stats
        pr, pp, pr_lo, pr_hi = pearson_with_ci(x, y)
        sr, sp, sr_lo, sr_hi = spearman_with_ci(x, y)

        stats_text = (
            f"Pearson r = {pr:.2f} [{pr_lo:.2f}, {pr_hi:.2f}]\n"
            f"Spearman rho = {sr:.2f} [{sr_lo:.2f}, {sr_hi:.2f}]\n"
            f"n = {len(x)}"
        )
        ax.annotate(
            stats_text,
            xy=(0.03, 0.97),
            xycoords="axes fraction",
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
        )

        trait_label = trait.capitalize()
        ax.set_title(f"{trait_label} leakage to assistant", fontweight="bold")
        ax.set_xlabel("Mean-centered cosine to assistant (layer 10)")
        ax.set_ylabel(f"{trait_label} leakage rate (%)")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(100, decimals=0))

        # Store summary
        summary[trait] = {
            "n_sources": len(td["sources"]),
            "pearson_r": round(pr, 3),
            "pearson_p": round(pp, 4),
            "pearson_ci": [round(pr_lo, 3), round(pr_hi, 3)],
            "spearman_rho": round(sr, 3),
            "spearman_p": round(sp, 4),
            "spearman_ci": [round(sr_lo, 3), round(sr_hi, 3)],
            "regression_slope": round(reg["slope"], 4),
            "regression_intercept": round(reg["intercept"], 4),
            "sources": td["sources"],
            "cosines": [round(c, 3) for c in td["cosines"]],
            "leakage_rates": [round(v, 4) for v in td["asst_leakages"]],
        }

        # Print
        print(f"\n  {trait_label}:")
        print(f"    Pearson r = {pr:.3f} (p={pp:.4f}), 95% CI [{pr_lo:.3f}, {pr_hi:.3f}]")
        print(f"    Spearman rho = {sr:.3f} (p={sp:.4f}), 95% CI [{sr_lo:.3f}, {sr_hi:.3f}]")
        print(f"    Regression: y = {reg['slope']:.4f}*x + {reg['intercept']:.4f}")
        for s, c, lk in zip(td["sources"], td["cosines"], td["asst_leakages"], strict=True):
            print(f"      {s:25s}  cos={c:+.3f}  leak={lk:.3f}")

    # Hide unused axes
    for idx in range(n_panels, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle(
        "Trait leakage to assistant vs representational distance",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    _save_fig(fig, "7a_distance_leakage")

    return summary


# ── 7b. Assistant Shielding ───────────────────────────────────────────────────


def analysis_7b_assistant_shielding(runs: list[dict], traits: list[str]) -> dict[str, Any]:
    """Test whether assistant leakage is below the regression line.

    For each trait, compute residual = actual - predicted and test
    whether assistant residual is significantly negative.
    """
    print("\n" + "=" * 70)
    print("7b. Assistant Shielding Analysis")
    print("=" * 70)

    summary = {}

    for trait in traits:
        trait_runs = filter_runs(
            runs,
            trait=trait,
            neg_set="asst_excluded",
            prompt_length="medium",
            exclude_controls=True,
        )
        if not trait_runs:
            continue

        # Collect per-source data
        cosines = []
        asst_leakages = []
        sources = []

        for source in ALL_SOURCES:
            source_runs = filter_runs(trait_runs, source=source)
            if not source_runs:
                continue
            rates = [
                r for r in [get_assistant_leakage(sr, trait) for sr in source_runs] if r is not None
            ]
            if not rates:
                continue
            sources.append(source)
            cosines.append(CENTERED_COSINES[source])
            asst_leakages.append(np.mean(rates))

        if len(sources) < 4:
            print(f"  SKIP {trait}: insufficient data ({len(sources)} sources)")
            continue

        x = np.array(cosines)
        y = np.array(asst_leakages) * 100

        reg = ols_regression(x, y)
        if np.isnan(reg["slope"]):
            continue

        # Compute residuals for all sources
        residuals_dict = {}
        for i, src in enumerate(sources):
            predicted = reg["slope"] * x[i] + reg["intercept"]
            actual = y[i]
            residuals_dict[src] = {
                "actual": round(actual, 3),
                "predicted": round(predicted, 3),
                "residual": round(actual - predicted, 3),
            }

        # Test if residuals as a whole are different from zero (one-sample t-test)
        residuals = reg["residuals"]
        t_stat, t_p = stats.ttest_1samp(residuals, 0.0)

        # Identify outlier residuals (below 2 SD)
        mean_res = np.mean(residuals)
        std_res = np.std(residuals, ddof=1)
        outliers = [src for i, src in enumerate(sources) if residuals[i] < mean_res - 2 * std_res]

        summary[trait] = {
            "n_sources": len(sources),
            "mean_residual": round(float(mean_res), 3),
            "std_residual": round(float(std_res), 3),
            "residuals": residuals_dict,
            "outliers_below_2sd": outliers,
            "ttest_t": round(float(t_stat), 3),
            "ttest_p": round(float(t_p), 4),
        }

        print(f"\n  {trait.capitalize()}:")
        print(f"    Mean residual = {mean_res:.3f} (SD = {std_res:.3f})")
        print(f"    t-test (residuals = 0): t={t_stat:.3f}, p={t_p:.4f}")
        if outliers:
            print(f"    Outliers below 2 SD: {outliers}")
        print("    Per-source residuals:")
        for src, rd in sorted(residuals_dict.items(), key=lambda kv: kv[1]["residual"]):
            flag = " <-- OUTLIER" if src in outliers else ""
            print(
                f"      {src:25s}  actual={rd['actual']:.1f}%  "
                f"pred={rd['predicted']:.1f}%  resid={rd['residual']:+.1f}%{flag}"
            )

    return summary


# ── 7c. Prompt-Length Effect (Phase B) ────────────────────────────────────────


def analysis_7c_prompt_length(runs: list[dict], traits: list[str]) -> dict[str, Any]:
    """Leakage vs prompt length, faceted by distance tier (close/mid/far).

    Only runs if Phase B data exists (multiple prompt lengths).
    """
    print("\n" + "=" * 70)
    print("7c. Prompt-Length Effect (Phase B)")
    print("=" * 70)

    # Check if we have runs at different prompt lengths
    lengths_present = set()
    for r in runs:
        if r["prompt_length"] is not None:
            lengths_present.add(r["prompt_length"])

    if len(lengths_present) < 2:
        print(f"  SKIP: only {len(lengths_present)} prompt length(s) found ({lengths_present})")
        print("  Phase B data required for this analysis.")
        return {}

    print(f"  Found prompt lengths: {sorted(lengths_present)}")

    # Define distance tiers
    close_sources = [s for s in ALL_SOURCES if CENTERED_COSINES[s] > 0.1]
    mid_sources = [s for s in ALL_SOURCES if -0.2 <= CENTERED_COSINES[s] <= 0.1]
    far_sources = [s for s in ALL_SOURCES if CENTERED_COSINES[s] < -0.2]

    tiers = {
        "close (cos > 0.1)": close_sources,
        "mid (-0.2 <= cos <= 0.1)": mid_sources,
        "far (cos < -0.2)": far_sources,
    }

    summary = {}

    for trait in traits:
        trait_runs = filter_runs(runs, trait=trait, neg_set="asst_excluded", exclude_controls=True)
        if not trait_runs:
            continue

        # Collect leakage per (source, prompt_length)
        length_order = ["short", "medium", "long"]
        data_rows = []  # (source, length, leakage, cosine)

        for src in ALL_SOURCES:
            for pl in length_order:
                pl_runs = filter_runs(trait_runs, source=src, prompt_length=pl)
                if not pl_runs:
                    continue
                rates = [
                    r for r in [get_assistant_leakage(pr, trait) for pr in pl_runs] if r is not None
                ]
                if rates:
                    data_rows.append((src, pl, np.mean(rates), CENTERED_COSINES[src]))

        if len(data_rows) < 6:
            print(f"  SKIP {trait}: insufficient multi-length data ({len(data_rows)} points)")
            continue

        # Build faceted plot
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

        for t_idx, (tier_name, tier_sources) in enumerate(tiers.items()):
            ax = axes[t_idx]
            tier_rows = [r for r in data_rows if r[0] in tier_sources]

            for src in tier_sources:
                src_rows = [(r[1], r[2]) for r in tier_rows if r[0] == src]
                if not src_rows:
                    continue
                lengths_vals = [length_order.index(r[0]) for r in src_rows]
                leakages = [r[1] * 100 for r in src_rows]
                ax.plot(
                    lengths_vals,
                    leakages,
                    "o-",
                    label=SHORT_NAMES.get(src, src),
                    markersize=6,
                )

            ax.set_xticks(range(len(length_order)))
            ax.set_xticklabels(length_order)
            ax.set_xlabel("Prompt length")
            ax.set_title(tier_name, fontweight="bold")
            ax.legend(fontsize=8)

        axes[0].set_ylabel(f"{trait.capitalize()} leakage to assistant (%)")
        axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(100, decimals=0))

        fig.suptitle(
            f"{trait.capitalize()} leakage vs prompt length by distance tier",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout()
        _save_fig(fig, f"7c_prompt_length_{trait}")

        # Regression: leakage ~ length + cosine + interaction
        lengths_num = {"short": 0, "medium": 1, "long": 2}
        X_len = np.array([lengths_num[r[1]] for r in data_rows], dtype=float)
        X_cos = np.array([r[3] for r in data_rows], dtype=float)
        Y = np.array([r[2] * 100 for r in data_rows], dtype=float)

        # Interaction
        X_interact = X_len * X_cos

        # Multiple regression via np.linalg.lstsq
        A = np.column_stack([np.ones(len(Y)), X_len, X_cos, X_interact])
        coeffs, _res, _, _ = np.linalg.lstsq(A, Y, rcond=None)
        b0, b_len, b_cos, b_interact = coeffs

        summary[trait] = {
            "n_data_points": len(data_rows),
            "lengths_present": sorted(lengths_present),
            "regression": {
                "intercept": round(b0, 4),
                "beta_length": round(b_len, 4),
                "beta_cosine": round(b_cos, 4),
                "beta_interaction": round(b_interact, 4),
            },
        }

        print(f"\n  {trait.capitalize()}: {len(data_rows)} data points")
        print(
            f"    Regression: leakage = {b0:.2f} + {b_len:.2f}*length + "
            f"{b_cos:.2f}*cosine + {b_interact:.2f}*length*cosine"
        )

    return summary


# ── 7d. Negative-Set Effect ──────────────────────────────────────────────────


def analysis_7d_negative_set(runs: list[dict], traits: list[str]) -> dict[str, Any]:
    """Paired bar chart: asst_excluded vs asst_included per source persona.

    Computes suppression in percentage points and tests if suppression
    correlates with distance or trait type.
    """
    print("\n" + "=" * 70)
    print("7d. Negative-Set Effect")
    print("=" * 70)

    # Check if both negative sets are present
    neg_sets_present = {r["neg_set"] for r in runs if r["neg_set"] is not None}
    if len(neg_sets_present) < 2:
        print(f"  SKIP: only {neg_sets_present} negative set(s) found. Need both.")
        return {}

    summary = {}

    for trait in traits:
        trait_runs = filter_runs(runs, trait=trait, prompt_length="medium", exclude_controls=True)
        if not trait_runs:
            continue

        sources = []
        excl_leakages = []
        incl_leakages = []
        cosines = []

        for src in ALL_SOURCES:
            excl_runs = filter_runs(trait_runs, source=src, neg_set="asst_excluded")
            incl_runs = filter_runs(trait_runs, source=src, neg_set="asst_included")

            if not excl_runs or not incl_runs:
                continue

            excl_rates = [
                r for r in [get_assistant_leakage(er, trait) for er in excl_runs] if r is not None
            ]
            incl_rates = [
                r for r in [get_assistant_leakage(ir, trait) for ir in incl_runs] if r is not None
            ]

            if not excl_rates or not incl_rates:
                continue

            sources.append(src)
            excl_leakages.append(np.mean(excl_rates) * 100)
            incl_leakages.append(np.mean(incl_rates) * 100)
            cosines.append(CENTERED_COSINES[src])

        if len(sources) < 2:
            print(f"  SKIP {trait}: insufficient paired data ({len(sources)} sources)")
            continue

        excl_arr = np.array(excl_leakages)
        incl_arr = np.array(incl_leakages)
        suppression = excl_arr - incl_arr  # positive = asst_excluded has MORE leakage

        # Paired t-test
        if len(sources) >= 3:
            t_stat, t_p = stats.ttest_rel(excl_arr, incl_arr)
        else:
            t_stat, t_p = np.nan, np.nan

        # Does suppression correlate with distance?
        cos_arr = np.array(cosines)
        if len(sources) >= 4:
            supp_r, supp_p = stats.pearsonr(cos_arr, suppression)
        else:
            supp_r, supp_p = np.nan, np.nan

        # Plot paired bar chart
        x_pos = np.arange(len(sources))
        width = 0.35

        fig, ax = plt.subplots(figsize=(max(10, len(sources) * 1.5), 6))
        ax.bar(
            x_pos - width / 2,
            excl_arr,
            width,
            label="Asst excluded from neg set",
            color=C_NEG,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.bar(
            x_pos + width / 2,
            incl_arr,
            width,
            label="Asst included in neg set",
            color=C_HELD,
            edgecolor="black",
            linewidth=0.5,
        )

        ax.set_xticks(x_pos)
        ax.set_xticklabels([SHORT_NAMES.get(s, s) for s in sources], rotation=30, ha="right")
        ax.set_ylabel(f"{trait.capitalize()} leakage to assistant (%)")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(100, decimals=0))
        ax.legend()

        # Annotate suppression
        for i in range(len(sources)):
            diff = suppression[i]
            sign = "+" if diff >= 0 else ""
            y_top = max(excl_arr[i], incl_arr[i]) + 1
            ax.annotate(
                f"{sign}{diff:.1f}pp",
                (x_pos[i], y_top),
                ha="center",
                fontsize=8,
                color="red" if diff > 0 else "green",
            )

        # Stats annotation
        stats_str = f"Paired t: t={t_stat:.2f}, p={t_p:.4f}" if not np.isnan(t_stat) else ""
        if not np.isnan(supp_r):
            stats_str += f"\nSupp~cos: r={supp_r:.2f}, p={supp_p:.4f}"
        if stats_str:
            ax.annotate(
                stats_str,
                xy=(0.98, 0.97),
                xycoords="axes fraction",
                va="top",
                ha="right",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
            )

        ax.set_title(
            f"{trait.capitalize()}: negative-set effect on assistant leakage",
            fontweight="bold",
        )
        fig.tight_layout()
        _save_fig(fig, f"7d_negative_set_{trait}")

        summary[trait] = {
            "n_sources": len(sources),
            "mean_suppression_pp": round(float(np.mean(suppression)), 2),
            "std_suppression_pp": round(float(np.std(suppression, ddof=1)), 2),
            "paired_t": round(float(t_stat), 3) if not np.isnan(t_stat) else None,
            "paired_p": round(float(t_p), 4) if not np.isnan(t_p) else None,
            "suppression_cos_r": round(float(supp_r), 3) if not np.isnan(supp_r) else None,
            "suppression_cos_p": round(float(supp_p), 4) if not np.isnan(supp_p) else None,
            "per_source": {
                s: {
                    "excl": round(e, 2),
                    "incl": round(i, 2),
                    "suppression_pp": round(e - i, 2),
                }
                for s, e, i in zip(sources, excl_leakages, incl_leakages, strict=True)
            },
        }

        print(f"\n  {trait.capitalize()}: {len(sources)} paired sources")
        supp_mean = np.mean(suppression)
        supp_std = np.std(suppression, ddof=1)
        print(f"    Mean suppression: {supp_mean:+.2f} pp (SD={supp_std:.2f})")
        if not np.isnan(t_stat):
            print(f"    Paired t-test: t={t_stat:.3f}, p={t_p:.4f}")
        if not np.isnan(supp_r):
            print(f"    Suppression~cosine: r={supp_r:.3f}, p={supp_p:.4f}")

    return summary


# ── 7e. SFT Controls ─────────────────────────────────────────────────────────


def _collect_control_data(
    runs: list[dict], trait: str
) -> tuple[list[float], dict[str, list[float]]]:
    """Collect persona-conditioned and control leakage data for a trait.

    Returns (persona_leakages_pct, {control_name: leakages_pct}).
    """
    persona_runs = filter_runs(
        runs,
        trait=trait,
        neg_set="asst_excluded",
        prompt_length="medium",
        exclude_controls=True,
    )
    persona_leakages = [
        al * 100 for r in persona_runs if (al := get_assistant_leakage(r, trait)) is not None
    ]

    control_data: dict[str, list[float]] = {}
    for ctrl in CONTROL_CONDITIONS:
        ctrl_runs = filter_runs(runs, trait=trait, control=ctrl)
        if not ctrl_runs:
            continue
        ctrl_leakages = [
            al * 100 for r in ctrl_runs if (al := get_assistant_leakage(r, trait)) is not None
        ]
        if ctrl_leakages:
            control_data[ctrl] = ctrl_leakages

    return persona_leakages, control_data


def _build_control_bar_chart(
    trait: str,
    persona_leakages: list[float],
    control_data: dict[str, list[float]],
) -> tuple[list[str], dict[str, dict]]:
    """Build and save the control bar chart for a single trait.

    Returns (conditions_list, test_results_dict).
    """
    conditions: list[str] = []
    means: list[float] = []
    sems: list[float] = []
    all_vals: list[list[float]] = []

    if persona_leakages:
        conditions.append("Persona\nconditioned")
        means.append(np.mean(persona_leakages))
        sems.append(
            np.std(persona_leakages, ddof=1) / np.sqrt(len(persona_leakages))
            if len(persona_leakages) > 1
            else 0
        )
        all_vals.append(persona_leakages)

    for ctrl in CONTROL_CONDITIONS:
        if ctrl in control_data:
            label = ctrl.replace("_", "\n")
            conditions.append(label)
            vals = control_data[ctrl]
            means.append(np.mean(vals))
            sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
            all_vals.append(vals)

    colors = [TRAIT_COLORS.get(trait, C_HELD)] + [C_CONTROL] * (len(conditions) - 1)

    fig, ax = plt.subplots(figsize=(max(8, len(conditions) * 2.5), 6))
    x_pos = np.arange(len(conditions))
    ax.bar(x_pos, means, yerr=sems, capsize=5, color=colors, edgecolor="black", linewidth=0.5)

    for i, vals in enumerate(all_vals):
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        ax.scatter(x_pos[i] + jitter, vals, color="black", alpha=0.4, s=20, zorder=5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(conditions, fontsize=10)
    ax.set_ylabel(f"{trait.capitalize()} leakage to assistant (%)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(100, decimals=0))
    ax.set_title(f"{trait.capitalize()}: persona-conditioned vs controls", fontweight="bold")

    test_results: dict[str, dict] = {}
    if persona_leakages:
        for ctrl in control_data:
            if len(control_data[ctrl]) >= 2 and len(persona_leakages) >= 2:
                u_stat, u_p = stats.mannwhitneyu(
                    persona_leakages, control_data[ctrl], alternative="two-sided"
                )
                test_results[ctrl] = {"U": round(float(u_stat), 1), "p": round(float(u_p), 4)}
                print(
                    f"  {trait}: persona vs {ctrl}: U={u_stat:.1f}, p={u_p:.4f}, "
                    f"means={np.mean(persona_leakages):.1f}% vs "
                    f"{np.mean(control_data[ctrl]):.1f}%"
                )

    fig.tight_layout()
    _save_fig(fig, f"7e_controls_{trait}")
    return conditions, test_results


def analysis_7e_sft_controls(runs: list[dict], traits: list[str]) -> dict[str, Any]:
    """Compare persona-conditioned vs generic_sft vs shuffled_persona.

    Bar chart with error bars for each trait.
    Key question: is leakage persona-mediated or generic SFT destabilization?
    """
    print("\n" + "=" * 70)
    print("7e. SFT Controls")
    print("=" * 70)

    control_runs = [r for r in runs if r["control"] is not None]
    if not control_runs:
        print("  SKIP: no control condition runs found")
        return {}

    controls_found = {r["control"] for r in control_runs}
    print(f"  Control conditions found: {controls_found}")

    summary = {}

    for trait in traits:
        persona_leakages, control_data = _collect_control_data(runs, trait)

        if not persona_leakages and not control_data:
            print(f"  SKIP {trait}: no data for any condition")
            continue

        # Need at least 2 conditions to compare
        n_conds = (1 if persona_leakages else 0) + len(control_data)
        if n_conds < 2:
            print(f"  SKIP {trait}: only {n_conds} condition(s), need 2+")
            continue

        _conditions, test_results = _build_control_bar_chart(trait, persona_leakages, control_data)

        summary[trait] = {
            "persona_conditioned": {
                "n": len(persona_leakages),
                "mean": round(np.mean(persona_leakages), 2) if persona_leakages else None,
                "std": round(np.std(persona_leakages, ddof=1), 2)
                if len(persona_leakages) > 1
                else None,
            },
            "controls": {
                ctrl: {
                    "n": len(vals),
                    "mean": round(np.mean(vals), 2),
                    "std": round(np.std(vals, ddof=1), 2) if len(vals) > 1 else None,
                }
                for ctrl, vals in control_data.items()
            },
            "tests": test_results,
        }

    return summary


# ── 7f. Training Dynamics (Phase 0.5) ────────────────────────────────────────


def analysis_7f_dynamics(runs: list[dict], traits: list[str]) -> dict[str, Any]:
    """Leakage-vs-step curves for source persona and assistant.

    One panel per source persona (or faceted by distance tier).
    Only for marker trait (Phase 0.5 data).
    """
    print("\n" + "=" * 70)
    print("7f. Training Dynamics (Phase 0.5)")
    print("=" * 70)

    # Find runs with dynamics data
    dynamics_runs = [r for r in runs if r.get("dynamics") is not None]
    if not dynamics_runs:
        print("  SKIP: no runs with dynamics data (checkpoint_dynamics.json)")
        return {}

    print(f"  Found {len(dynamics_runs)} runs with dynamics data")

    summary = {}

    for trait in traits:
        trait_dyn_runs = filter_runs(dynamics_runs, trait=trait)
        if not trait_dyn_runs:
            continue

        sources_with_dynamics = list({r["source"] for r in trait_dyn_runs if r["source"]})
        if not sources_with_dynamics:
            continue

        # Sort by distance
        sources_with_dynamics.sort(key=lambda s: CENTERED_COSINES.get(s, 0), reverse=True)

        ncols = min(len(sources_with_dynamics), 3)
        nrows = (len(sources_with_dynamics) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)

        for idx, src in enumerate(sources_with_dynamics):
            ax = axes[idx // ncols, idx % ncols]
            src_runs = filter_runs(trait_dyn_runs, source=src)

            for run in src_runs:
                dyn = run["dynamics"]
                if not dyn:
                    continue

                steps = []
                source_rates = []
                asst_rates = []

                for ckpt in dyn:
                    step = ckpt.get("step") or ckpt.get("checkpoint_step")
                    if step is None:
                        continue

                    marker_data = ckpt.get("marker_eval", {})
                    src_rate = marker_data.get(src, {}).get("rate")
                    asst_rate = marker_data.get("assistant", {}).get("rate")

                    if src_rate is not None and asst_rate is not None:
                        steps.append(step)
                        source_rates.append(src_rate * 100)
                        asst_rates.append(asst_rate * 100)

                if steps:
                    ax.plot(
                        steps,
                        source_rates,
                        "o-",
                        color=C_DYNAMICS_SOURCE,
                        label=f"Source ({SHORT_NAMES.get(src, src)})",
                        markersize=5,
                    )
                    ax.plot(
                        steps,
                        asst_rates,
                        "s--",
                        color=C_DYNAMICS_ASST,
                        label="Assistant",
                        markersize=5,
                    )

            cos = CENTERED_COSINES.get(src, 0)
            ax.set_title(
                f"{SHORT_NAMES.get(src, src)} (cos={cos:+.2f})",
                fontweight="bold",
            )
            ax.set_xlabel("Training step")
            ax.set_ylabel(f"{trait.capitalize()} rate (%)")
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(100, decimals=0))
            ax.legend(fontsize=8, loc="best")

        # Hide unused
        for idx in range(len(sources_with_dynamics), nrows * ncols):
            axes[idx // ncols, idx % ncols].set_visible(False)

        fig.suptitle(
            f"{trait.capitalize()} leakage dynamics during training",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout()
        _save_fig(fig, f"7f_dynamics_{trait}")

        summary[trait] = {
            "n_sources_with_dynamics": len(sources_with_dynamics),
            "sources": sources_with_dynamics,
        }

    return summary


# ── 7g. Capability ID vs OOD ─────────────────────────────────────────────────


def analysis_7g_capability_id_ood(runs: list[dict]) -> dict[str, Any]:
    """Scatter: ID capability degradation vs OOD ARC-C degradation.

    Requires a baseline (unfinetuned) ARC-C score to compute degradation.
    """
    print("\n" + "=" * 70)
    print("7g. Capability: In-Distribution vs Out-of-Distribution")
    print("=" * 70)

    # We need the "capability" trait runs (wrong answers = ID trait)
    cap_runs = filter_runs(runs, trait="capability", neg_set="asst_excluded", exclude_controls=True)

    if not cap_runs:
        print("  SKIP: no capability trait runs found")
        return {}

    # Look for baseline ARC-C (from a control or pre-training eval)
    baseline_acc = None
    for ctrl in CONTROL_CONDITIONS:
        ctrl_runs = filter_runs(runs, trait="capability", control=ctrl)
        for cr in ctrl_runs:
            cap_data = cr.get("capability_eval")
            if cap_data and "arc_challenge_logprob" in cap_data:
                baseline_acc = cap_data["arc_challenge_logprob"]
                print(f"  Using {ctrl} baseline ARC-C: {baseline_acc:.3f}")
                break
        if baseline_acc is not None:
            break

    if baseline_acc is None:
        # Use Qwen2.5-7B-Instruct typical ARC-C as fallback
        baseline_acc = 0.55
        print(
            f"  WARNING: no control baseline found. "
            f"Using approximate Qwen2.5-7B-Instruct ARC-C={baseline_acc}"
        )

    sources = []
    id_degradation = []  # source_leakage = how much the model gives wrong answers
    ood_degradation = []  # ARC-C drop from baseline

    for src in ALL_SOURCES:
        src_runs = filter_runs(cap_runs, source=src, prompt_length="medium")
        if not src_runs:
            continue

        src_leaks = [get_source_leakage(r) for r in src_runs]
        src_leaks = [s for s in src_leaks if s is not None]

        ood_accs = []
        for r in src_runs:
            cap_data = r.get("capability_eval")
            if cap_data and "arc_challenge_logprob" in cap_data:
                ood_accs.append(cap_data["arc_challenge_logprob"])

        if not src_leaks or not ood_accs:
            continue

        sources.append(src)
        id_degradation.append(np.mean(src_leaks) * 100)
        ood_degradation.append((baseline_acc - np.mean(ood_accs)) * 100)

    if len(sources) < 3:
        print(f"  SKIP: only {len(sources)} sources with both ID and OOD data")
        return {}

    id_arr = np.array(id_degradation)
    ood_arr = np.array(ood_degradation)

    r_val, p_val = stats.pearsonr(id_arr, ood_arr)

    fig, ax = plt.subplots(figsize=(8, 7))
    for i, src in enumerate(sources):
        ax.scatter(
            id_arr[i],
            ood_arr[i],
            s=80,
            c=TRAIT_COLORS["capability"],
            edgecolors="black",
            linewidths=0.6,
            zorder=4,
        )
        label = SHORT_NAMES.get(src, src)
        ax.annotate(
            label,
            (id_arr[i], ood_arr[i]),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=9,
        )

    # Regression
    reg = ols_regression(id_arr, ood_arr)
    if not np.isnan(reg["slope"]):
        ax.plot(reg["x_grid"], reg["y_grid"], "--", color=C_REG, lw=1.5, alpha=0.7)
        ax.fill_between(
            reg["x_grid"],
            reg["ci_lower"],
            reg["ci_upper"],
            color=C_REG,
            alpha=0.15,
        )

    ax.annotate(
        f"Pearson r = {r_val:.2f}, p = {p_val:.4f}\nn = {len(sources)}",
        xy=(0.03, 0.97),
        xycoords="axes fraction",
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    ax.set_xlabel("In-distribution wrong-answer rate (%)")
    ax.set_ylabel("OOD ARC-C degradation from baseline (%)")
    ax.set_title(
        "Capability: ID degradation vs OOD transfer",
        fontweight="bold",
    )
    ax.axhline(0, color="black", lw=0.5, ls=":")
    ax.axvline(0, color="black", lw=0.5, ls=":")

    fig.tight_layout()
    _save_fig(fig, "7g_capability_id_ood")

    summary = {
        "n_sources": len(sources),
        "pearson_r": round(r_val, 3),
        "pearson_p": round(p_val, 4),
        "baseline_arc_c": round(baseline_acc, 3),
        "per_source": {
            s: {"id_wrong_rate": round(id_, 2), "ood_degradation": round(ood, 2)}
            for s, id_, ood in zip(sources, id_degradation, ood_degradation, strict=True)
        },
    }

    print(f"\n  {len(sources)} sources with ID + OOD data")
    print(f"  Pearson r = {r_val:.3f}, p = {p_val:.4f}")
    print(f"  Baseline ARC-C: {baseline_acc:.3f}")

    return summary


# ── Figure saving ─────────────────────────────────────────────────────────────


def _save_fig(fig: plt.Figure, name: str) -> None:
    """Save figure in both PNG (300 dpi) and PDF."""
    png_path = FIG_DIR / f"{name}.png"
    pdf_path = FIG_DIR / f"{name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")


# ── Summary overview figure ───────────────────────────────────────────────────


def plot_summary_heatmap(runs: list[dict], traits: list[str]) -> dict | None:
    """Heatmap: source persona (rows) x trait (cols) -> assistant leakage rate.

    Uses default neg_set=asst_excluded, prompt_length=medium.
    """
    print("\n" + "=" * 70)
    print("Summary Heatmap: Leakage by source x trait")
    print("=" * 70)

    # Collect data matrix
    data = {}
    available_traits = []

    for trait in traits:
        trait_runs = filter_runs(
            runs,
            trait=trait,
            neg_set="asst_excluded",
            prompt_length="medium",
            exclude_controls=True,
        )
        if not trait_runs:
            continue

        col = {}
        for src in ALL_SOURCES:
            src_runs = filter_runs(trait_runs, source=src)
            rates = [
                r for r in [get_assistant_leakage(sr, trait) for sr in src_runs] if r is not None
            ]
            if rates:
                col[src] = np.mean(rates) * 100
        if col:
            data[trait] = col
            available_traits.append(trait)

    if not available_traits:
        print("  SKIP: no data for heatmap")
        return None

    # Build matrix (sources sorted by cosine distance)
    sorted_sources = sorted(ALL_SOURCES, key=lambda s: CENTERED_COSINES[s], reverse=True)
    present_sources = [s for s in sorted_sources if any(s in data[t] for t in available_traits)]

    matrix = np.full((len(present_sources), len(available_traits)), np.nan)
    for j, trait in enumerate(available_traits):
        for i, src in enumerate(present_sources):
            if src in data[trait]:
                matrix[i, j] = data[trait][src]

    fig, ax = plt.subplots(
        figsize=(max(8, len(available_traits) * 2.5), max(6, len(present_sources) * 0.6))
    )

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", interpolation="nearest")

    # Annotate cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if val > 50 else "black"
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=9, color=color)

    ax.set_xticks(range(len(available_traits)))
    ax.set_xticklabels([t.capitalize() for t in available_traits], fontsize=11)
    ax.set_yticks(range(len(present_sources)))
    ylabels = [f"{SHORT_NAMES.get(s, s)} (cos={CENTERED_COSINES[s]:+.2f})" for s in present_sources]
    ax.set_yticklabels(ylabels, fontsize=10)

    ax.set_title(
        "Assistant leakage rate (%) by source persona and trait type",
        fontsize=13,
        fontweight="bold",
    )
    fig.colorbar(im, ax=ax, label="Leakage rate (%)", shrink=0.8)
    fig.tight_layout()
    _save_fig(fig, "summary_heatmap")

    return {
        "sources": present_sources,
        "traits": available_traits,
        "matrix": matrix.tolist(),
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Analyze leakage experiment results")
    parser.add_argument(
        "--phase",
        default="all",
        choices=["pilot", "a1", "a2", "b", "c", "all"],
        help="Which phase to analyze (default: all available data)",
    )
    parser.add_argument(
        "--trait",
        default="all",
        choices=["marker", "structure", "capability", "misalignment", "all"],
        help="Which trait to analyze (default: all)",
    )
    args = parser.parse_args()

    # Determine traits to analyze
    traits = ALL_TRAITS if args.trait == "all" else [args.trait]

    # Load all data
    runs = load_all_results()

    if not runs:
        print("\nNo experiment data found. Generate and run experiments first.")
        print(f"  Expected data at: {EVAL_DIR}")
        print("  Run: python scripts/run_leakage_experiment.py ...")
        sys.exit(0)

    # Print data inventory
    print("\n--- Data Inventory ---")
    trait_counts: dict[str, int] = {}
    control_counts: dict[str, int] = {}
    length_counts: dict[str, int] = {}
    neg_set_counts: dict[str, int] = {}

    for r in runs:
        t = r["trait"]
        trait_counts[t] = trait_counts.get(t, 0) + 1
        if r["control"]:
            c = r["control"]
            control_counts[c] = control_counts.get(c, 0) + 1
        if r["prompt_length"]:
            pl = r["prompt_length"]
            length_counts[pl] = length_counts.get(pl, 0) + 1
        if r["neg_set"]:
            ns = r["neg_set"]
            neg_set_counts[ns] = neg_set_counts.get(ns, 0) + 1

    print(f"  Traits: {trait_counts}")
    print(f"  Controls: {control_counts}")
    print(f"  Prompt lengths: {length_counts}")
    print(f"  Negative sets: {neg_set_counts}")

    has_dynamics = sum(1 for r in runs if r.get("dynamics") is not None)
    print(f"  Runs with dynamics: {has_dynamics}")

    # Run all analyses
    all_summary: dict[str, Any] = {}

    # 7a: Distance -> Leakage
    result = analysis_7a_distance_leakage(runs, traits)
    if result:
        all_summary["7a_distance_leakage"] = result

    # 7b: Assistant Shielding
    result = analysis_7b_assistant_shielding(runs, traits)
    if result:
        all_summary["7b_assistant_shielding"] = result

    # 7c: Prompt-Length Effect
    result = analysis_7c_prompt_length(runs, traits)
    if result:
        all_summary["7c_prompt_length"] = result

    # 7d: Negative-Set Effect
    result = analysis_7d_negative_set(runs, traits)
    if result:
        all_summary["7d_negative_set"] = result

    # 7e: SFT Controls
    result = analysis_7e_sft_controls(runs, traits)
    if result:
        all_summary["7e_sft_controls"] = result

    # 7f: Training Dynamics
    result = analysis_7f_dynamics(runs, traits)
    if result:
        all_summary["7f_dynamics"] = result

    # 7g: Capability ID vs OOD
    result = analysis_7g_capability_id_ood(runs)
    if result:
        all_summary["7g_capability_id_ood"] = result

    # Summary heatmap
    heatmap = plot_summary_heatmap(runs, traits)
    if heatmap:
        all_summary["summary_heatmap"] = heatmap

    # Save summary
    summary_path = EVAL_DIR / "analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_summary, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")

    # Final report
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    analyses_run = list(all_summary.keys())
    analyses_skipped = [
        a
        for a in [
            "7a_distance_leakage",
            "7b_assistant_shielding",
            "7c_prompt_length",
            "7d_negative_set",
            "7e_sft_controls",
            "7f_dynamics",
            "7g_capability_id_ood",
        ]
        if a not in all_summary
    ]
    print(f"  Analyses run:     {analyses_run}")
    print(f"  Analyses skipped: {analyses_skipped}")
    print(f"  Figures saved to: {FIG_DIR}")
    print(f"  Summary JSON:     {summary_path}")


if __name__ == "__main__":
    main()
