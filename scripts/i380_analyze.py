#!/usr/bin/env python3
"""Stage C of issue #380: partial-Spearman + bootstrap + stratification.

Loads:
  - ``eval_results/issue_296/length_rate_correlation_n48.json::rows`` (48
    source rates + token lengths, the N=48 panel).
  - ``eval_results/issue_380/js_from_baseline.json`` (Predictor 1).
  - ``eval_results/issue_380/pairwise_reductions.json`` (Predictor 2).

Computes (per predictor):
  - Raw Spearman rho + p (scipy.stats.spearmanr).
  - Length-partial Spearman rho + p: pingouin.partial_corr(method='spearman',
    covar=['log_tokens']) if installable; else inline rank-residualize-then
    -Spearman fallback (plan section 5.3).
  - Bootstrap 95% CI on partial rho (1000 percentile iter).
  - Pre-launch smoke-test #2: synthetic-data sanity of inline fallback vs
    scipy. Aborts on failure (SystemExit) with epm:failure-style message.
  - Stratification by data-driven length terciles [<=6, 7-13, 14+].
  - Leave-helpful-family-out partial rho (n=37).
  - Cohort split: full-48 + new-24-only.
  - Convergent test on 24 inherited where ``cos_l15`` exists in the N=24
    sibling file.

Saves to ``eval_results/issue_380/correlation_results.json``.

Usage:
    uv run python scripts/i380_analyze.py
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from scipy.stats import t as scipy_t

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent

PANEL_N48 = PROJECT_ROOT / "eval_results/issue_296/length_rate_correlation_n48.json"
PANEL_N24 = PROJECT_ROOT / "eval_results/issue_296/length_rate_correlation.json"
JS_FROM_BASELINE = PROJECT_ROOT / "eval_results/issue_380/js_from_baseline.json"
PAIRWISE_REDUCTIONS = PROJECT_ROOT / "eval_results/issue_380/pairwise_reductions.json"
OUT_PATH = PROJECT_ROOT / "eval_results/issue_380/correlation_results.json"

HELPFUL_FAMILY = [
    "helpful_assistant",
    "i_am_helpful",
    "ai_assistant",
    "chat_assistant",
    "virtual_assistant",
    "chatbot",
    "friendly_ai",
    "smart_helper",
    "ai_tool",
    "ai",
    "qwen_default",
]

# Data-driven length terciles (plan section 7 update). 12/2/34 under the
# originally-proposed [<=6, 7-10, 11+] is degenerate (middle bin n=2).
LENGTH_BINS = [("<=6", 0, 6), ("7-13", 7, 13), (">=14", 14, 10_000)]

# Pre-registered partial-Spearman threshold (used for the headline pass).
PASS_RHO = 0.5
PASS_P = 0.01


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def partial_spearman_inline(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[float, float, int]:
    """Rank-residualize x and y on z, then Spearman of residuals.

    Returns (rho, p, n). The p-value uses the standard partial-correlation
    t-test with df = n - 2 - 1 (one covariate), matching pingouin's
    ``partial_corr(method='spearman')`` convention.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if not (len(x) == len(y) == len(z)):
        raise ValueError(f"length mismatch: x={len(x)}, y={len(y)}, z={len(z)}")
    n = len(x)
    if n < 4:
        raise ValueError(f"n={n} too small for partial Spearman with one covariate")
    xr = rankdata(x).astype(np.float64)
    yr = rankdata(y).astype(np.float64)
    zr = rankdata(z).astype(np.float64)

    def _resid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_centered = a - a.mean()
        b_centered = b - b.mean()
        denom = float(np.dot(b_centered, b_centered))
        if denom <= 0:
            raise ValueError("covariate has zero variance; cannot residualize")
        beta = float(np.dot(a_centered, b_centered) / denom)
        return a_centered - beta * b_centered

    xr_res = _resid(xr, zr)
    yr_res = _resid(yr, zr)
    rho, _ = spearmanr(xr_res, yr_res)
    rho = float(rho)
    df = n - 3
    denom = max(1.0 - rho * rho, 1e-12)
    t_stat = rho * np.sqrt(df / denom)
    p = float(2.0 * (1.0 - scipy_t.cdf(abs(t_stat), df)))
    return rho, p, n


def partial_spearman_pingouin(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[float, float, int] | None:
    """Pingouin partial Spearman; returns None if pingouin is unavailable."""
    try:
        import pingouin as pg
    except ImportError:
        return None
    df = pd.DataFrame({"x": x, "y": y, "z": z})
    res = pg.partial_corr(data=df, x="x", y="y", covar=["z"], method="spearman")
    rho = float(res["r"].iloc[0])
    p = float(res["p_val"].iloc[0])
    n = int(res["n"].iloc[0])
    return rho, p, n


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> dict:
    """Length-partial Spearman with pingouin primary + inline cross-check.

    Returns a dict with keys ``rho``, ``p``, ``n``, plus ``pingouin_rho``,
    ``inline_rho`` so callers can see both. The headline is the pingouin
    result when available; inline otherwise.
    """
    pin = partial_spearman_pingouin(x, y, z)
    inline = partial_spearman_inline(x, y, z)
    if pin is not None:
        rho, p, n = pin
        return {
            "rho": rho,
            "p": p,
            "n": n,
            "source": "pingouin",
            "pingouin_rho": rho,
            "inline_rho": inline[0],
            "inline_p": inline[1],
        }
    rho, p, n = inline
    return {
        "rho": rho,
        "p": p,
        "n": n,
        "source": "inline_fallback",
        "pingouin_rho": None,
        "inline_rho": rho,
        "inline_p": p,
    }


def smoke_test_partial_spearman_synthetic() -> dict:
    """Plan section 13b item 2: validate inline fallback vs scipy on synthetic data.

    Aborts (SystemExit) on |delta_rho| >= 0.001 or |delta_p| >= 0.001,
    with an epm:failure-style reason string for the orchestrator.
    """
    rng = np.random.default_rng(42)
    n = 100
    z = rng.standard_normal(n)
    x = 0.6 * z + np.sqrt(1 - 0.6**2) * rng.standard_normal(n)
    y = 0.3 * z + 0.4 * x + 0.5 * rng.standard_normal(n)

    # Compute via inline fallback.
    rho_inline, p_inline, _ = partial_spearman_inline(x, y, z)

    # Compute reference via scipy: rank-residualize, then spearmanr.
    xr = rankdata(x).astype(np.float64)
    yr = rankdata(y).astype(np.float64)
    zr = rankdata(z).astype(np.float64)

    def _resid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_c = a - a.mean()
        b_c = b - b.mean()
        return a_c - (np.dot(a_c, b_c) / np.dot(b_c, b_c)) * b_c

    xr_res = _resid(xr, zr)
    yr_res = _resid(yr, zr)
    rho_ref, _ = spearmanr(xr_res, yr_res)
    rho_ref = float(rho_ref)
    df = n - 3
    t_ref = rho_ref * np.sqrt(df / max(1.0 - rho_ref * rho_ref, 1e-12))
    p_ref = float(2.0 * (1.0 - scipy_t.cdf(abs(t_ref), df)))

    drho = abs(rho_inline - rho_ref)
    dp = abs(p_inline - p_ref)

    logger.info(
        "Smoke-test #2 synthetic partial Spearman: "
        "inline rho=%.6f p=%.6f  reference rho=%.6f p=%.6f  |drho|=%.3e |dp|=%.3e",
        rho_inline,
        p_inline,
        rho_ref,
        p_ref,
        drho,
        dp,
    )

    if drho >= 1e-3 or dp >= 1e-3:
        raise SystemExit(
            "Inline partial-Spearman drift vs scipy: "
            f"|drho|={drho:.3e}, |dp|={dp:.3e} >= 1e-3. "
            "Post epm:failure v1 failure_class=code reason=partial_spearman_fallback_drift."
        )

    return {
        "inline_rho": rho_inline,
        "inline_p": p_inline,
        "reference_rho": rho_ref,
        "reference_p": p_ref,
        "abs_delta_rho": drho,
        "abs_delta_p": dp,
        "n_synthetic": n,
    }


def bootstrap_partial_rho_ci(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    n_iter: int = 1000,
    seed: int = 42,
) -> tuple[float, float, list[float]]:
    """Percentile bootstrap 95% CI on partial-Spearman rho(x, y | z).

    Returns (lo, hi, samples). Uses the inline fallback formula on each
    resample (faster and avoids per-iter pingouin overhead; identical to
    pingouin on the main estimate per plan section 5.3).
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    samples: list[float] = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        xs, ys, zs = x[idx], y[idx], z[idx]
        try:
            rho_b, _, _ = partial_spearman_inline(xs, ys, zs)
        except ValueError:
            continue
        if np.isnan(rho_b):
            continue
        samples.append(float(rho_b))
    if not samples:
        return float("nan"), float("nan"), []
    lo = float(np.percentile(samples, 2.5))
    hi = float(np.percentile(samples, 97.5))
    return lo, hi, samples


def load_panel_n48() -> pd.DataFrame:
    data = json.loads(PANEL_N48.read_text())
    rows = data["rows"]
    if len(rows) != 48:
        raise ValueError(f"Expected 48 panel rows, got {len(rows)}")
    df = pd.DataFrame(rows)
    df["log_tokens"] = np.log(df["tokens"].astype(float) + 1.0)
    df["is_helpful_family"] = df["source"].isin(HELPFUL_FAMILY)
    return df


def load_n24_with_cosines() -> pd.DataFrame:
    """N=24 sibling file: has ``cos_l15`` for inherited personas."""
    data = json.loads(PANEL_N24.read_text())
    rows = data["rows"]
    df = pd.DataFrame(rows)
    if "cos_l15" not in df.columns:
        raise ValueError("N=24 file missing cos_l15 column")
    return df


def correlations_for_predictor(
    *,
    name: str,
    predictor_values: dict[str, float],
    panel: pd.DataFrame,
) -> dict:
    """Compute raw + length-partial Spearman + bootstrap CI for one predictor.

    Returns the result dict for ``correlation_results.json::predictors[name]``.
    """
    df = panel.copy()
    df["predictor"] = df["source"].map(predictor_values)
    missing = df["predictor"].isna().sum()
    if missing > 0:
        logger.warning(
            "Predictor %s missing values for %d sources; dropping them.", name, int(missing)
        )
        df = df.dropna(subset=["predictor"]).reset_index(drop=True)

    n = len(df)
    if n < 4:
        raise SystemExit(f"Too few rows ({n}) for predictor {name}; cannot compute correlations.")

    x = df["predictor"].to_numpy()
    y = df["rate_n48"].to_numpy()
    z = df["log_tokens"].to_numpy()

    # Raw Spearman
    raw_rho, raw_p = spearmanr(x, y)
    raw_rho = float(raw_rho)
    raw_p = float(raw_p)

    # Length-partial Spearman
    partial = partial_spearman(x, y, z)

    # Pearson collinearity vs log_tokens (the §5.4 collinearity gate input).
    from scipy.stats import pearsonr

    pearson_pred_logtokens, pearson_p = pearsonr(x, z)

    # Bootstrap on partial rho
    lo, hi, samples = bootstrap_partial_rho_ci(x, y, z, n_iter=1000, seed=42)

    # Stratification by length bins
    strat: list[dict] = []
    for label, lo_b, hi_b in LENGTH_BINS:
        sub = df[(df["tokens"] >= lo_b) & (df["tokens"] <= hi_b)]
        if len(sub) < 4:
            strat.append(
                {
                    "bin": label,
                    "n": len(sub),
                    "raw_rho": None,
                    "raw_p": None,
                    "note": "too few rows for Spearman",
                }
            )
            continue
        srho, sp = spearmanr(sub["predictor"].to_numpy(), sub["rate_n48"].to_numpy())
        strat.append(
            {
                "bin": label,
                "n": len(sub),
                "raw_rho": float(srho),
                "raw_p": float(sp),
            }
        )

    # Leave-helpful-family-out (n=37 sub-panel)
    family_drop = df[~df["is_helpful_family"]].reset_index(drop=True)
    leave_family_out: dict | None = None
    if len(family_drop) >= 4:
        lf = partial_spearman(
            family_drop["predictor"].to_numpy(),
            family_drop["rate_n48"].to_numpy(),
            family_drop["log_tokens"].to_numpy(),
        )
        leave_family_out = {
            "n": len(family_drop),
            **lf,
        }

    # Cohort split: new_296 only
    new_only = df[df["cohort"] == "new_296"].reset_index(drop=True)
    new_cohort: dict | None = None
    if len(new_only) >= 4:
        new_partial = partial_spearman(
            new_only["predictor"].to_numpy(),
            new_only["rate_n48"].to_numpy(),
            new_only["log_tokens"].to_numpy(),
        )
        new_raw_rho, new_raw_p = spearmanr(
            new_only["predictor"].to_numpy(), new_only["rate_n48"].to_numpy()
        )
        new_cohort = {
            "n": len(new_only),
            "raw_rho": float(new_raw_rho),
            "raw_p": float(new_raw_p),
            "partial": new_partial,
        }

    return {
        "n": n,
        "raw_spearman": {"rho": raw_rho, "p": raw_p},
        "length_partial_spearman": partial,
        "length_partial_bootstrap_ci95": [lo, hi],
        "length_partial_bootstrap_samples_n": len(samples),
        "pearson_predictor_vs_log_tokens": {
            "r": float(pearson_pred_logtokens),
            "p": float(pearson_p),
        },
        "stratification_by_length_bin": strat,
        "leave_family_out": leave_family_out,
        "new_cohort_only": new_cohort,
        "passes_threshold": bool(abs(partial["rho"]) >= PASS_RHO and partial["p"] < PASS_P),
    }


def convergent_test_cos_l15_vs_js_baseline(
    *, js_from_baseline: dict[str, float], n24: pd.DataFrame
) -> dict:
    """Spearman rho(cos_l15, js_from_baseline) on the 24 inherited personas."""
    n24 = n24.copy()
    n24["js_from_baseline"] = n24["source"].map(js_from_baseline)
    sub = n24.dropna(subset=["js_from_baseline", "cos_l15"]).reset_index(drop=True)
    if len(sub) < 4:
        return {"n": len(sub), "note": "too few rows after merge"}
    rho, p = spearmanr(sub["cos_l15"].to_numpy(), sub["js_from_baseline"].to_numpy())
    return {
        "n": len(sub),
        "rho": float(rho),
        "p": float(p),
        "sources": sub["source"].tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=str(OUT_PATH.relative_to(PROJECT_ROOT)))
    args = parser.parse_args()

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Mandatory smoke-test #2 first.
    synthetic_check = smoke_test_partial_spearman_synthetic()
    logger.info("Smoke-test #2 PASSED.")

    # Load inputs.
    panel = load_panel_n48()
    logger.info(
        "Loaded panel n=%d (helpful_family n=%d)", len(panel), int(panel["is_helpful_family"].sum())
    )

    js_baseline_data = json.loads(JS_FROM_BASELINE.read_text())
    js_from_baseline_values = js_baseline_data["values"]
    logger.info("Loaded JS-from-baseline for %d sources", len(js_from_baseline_values))

    pairwise_data = json.loads(PAIRWISE_REDUCTIONS.read_text())
    reductions = pairwise_data["reductions"]
    pairwise_mean = {k: v["mean"] for k, v in reductions.items()}
    pairwise_median = {k: v["median"] for k, v in reductions.items()}
    pairwise_max = {k: v["max"] for k, v in reductions.items()}
    logger.info("Loaded pairwise reductions for %d sources", len(reductions))

    # Bin counts on the actual panel under the data-driven cuts.
    bin_counts = []
    for label, lo_b, hi_b in LENGTH_BINS:
        n_bin = int(((panel["tokens"] >= lo_b) & (panel["tokens"] <= hi_b)).sum())
        bin_counts.append({"bin": label, "n": n_bin})
    logger.info("Length-bin counts: %s", bin_counts)

    predictors = {
        "js_from_baseline": correlations_for_predictor(
            name="js_from_baseline",
            predictor_values=js_from_baseline_values,
            panel=panel,
        ),
        "mean_pairwise_js": correlations_for_predictor(
            name="mean_pairwise_js",
            predictor_values=pairwise_mean,
            panel=panel,
        ),
        "median_pairwise_js": correlations_for_predictor(
            name="median_pairwise_js",
            predictor_values=pairwise_median,
            panel=panel,
        ),
        "max_pairwise_js": correlations_for_predictor(
            name="max_pairwise_js",
            predictor_values=pairwise_max,
            panel=panel,
        ),
    }

    # Convergent cosine vs JS test (n=24).
    try:
        n24 = load_n24_with_cosines()
    except FileNotFoundError:
        logger.warning("N=24 sibling file not found; skipping convergent test.")
        convergent = None
    else:
        convergent = convergent_test_cos_l15_vs_js_baseline(
            js_from_baseline=js_from_baseline_values, n24=n24
        )
        logger.info("Convergent test (n=24): %s", convergent)

    # Sanity-checks summary block.
    sanity_checks = {
        "synthetic_partial_spearman": synthetic_check,
        "length_bin_counts": bin_counts,
        "helpful_family_n_in_panel": int(panel["is_helpful_family"].sum()),
        "helpful_family_members_present": sorted(
            panel.loc[panel["is_helpful_family"], "source"].tolist()
        ),
    }

    # Primary predictors for the pass criterion.
    primary_pass = (
        predictors["js_from_baseline"]["passes_threshold"]
        or predictors["mean_pairwise_js"]["passes_threshold"]
    )
    primary_kill = (
        abs(predictors["js_from_baseline"]["length_partial_spearman"]["rho"]) < 0.2
        and abs(predictors["mean_pairwise_js"]["length_partial_spearman"]["rho"]) < 0.2
    )

    payload = {
        "predictors": predictors,
        "convergent_cos_l15_vs_js_from_baseline": convergent,
        "sanity_checks": sanity_checks,
        "pre_registered_pass_threshold": {"abs_rho": PASS_RHO, "p": PASS_P},
        "pass_criterion_met": primary_pass,
        "kill_criterion_met": primary_kill,
        "metadata": {
            "git_commit": get_git_commit(),
            "analyzed_at": datetime.now(UTC).isoformat(),
            "python_version": sys.version.split()[0],
            "panel_path": str(PANEL_N48.relative_to(PROJECT_ROOT)),
            "n24_path": str(PANEL_N24.relative_to(PROJECT_ROOT)) if convergent else None,
            "predictor1_path": str(JS_FROM_BASELINE.relative_to(PROJECT_ROOT)),
            "predictor2_path": str(PAIRWISE_REDUCTIONS.relative_to(PROJECT_ROOT)),
            "length_bins": [{"bin": b[0], "lo": b[1], "hi": b[2]} for b in LENGTH_BINS],
            "helpful_family": HELPFUL_FAMILY,
        },
    }

    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("Saved %s", out_path)
    logger.info(
        "Headline: js_from_baseline partial rho=%.4f (p=%.4g), "
        "mean_pairwise_js partial rho=%.4f (p=%.4g). "
        "Pass=%s, Kill=%s.",
        predictors["js_from_baseline"]["length_partial_spearman"]["rho"],
        predictors["js_from_baseline"]["length_partial_spearman"]["p"],
        predictors["mean_pairwise_js"]["length_partial_spearman"]["rho"],
        predictors["mean_pairwise_js"]["length_partial_spearman"]["p"],
        primary_pass,
        primary_kill,
    )


if __name__ == "__main__":
    main()
