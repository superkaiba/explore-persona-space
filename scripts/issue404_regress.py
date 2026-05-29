#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (ρ, ×, α, β, σ, →, —, ≥, ², ∈) in scientific docstrings.
"""Issue #404 aggregation + regression: predictor M vs post-SFT leakage L.

Per plan v3 §4.7 + SR3 bootstrap discipline. For each of 5 predictor
variants {M_1_NL, M_1_lit, M_2_NL, M_2_lit, M_3}, compute:

- Spearman ρ(M, L) across the 5 (or 4) pairs — primary statistic.
- 10K-bootstrap 95% CI on ρ with SR3 discipline:
  * cluster-resample paired (M, L) rows (preserve pair structure)
  * drop constant-rank resamples (Spearman ρ undefined)
  * report N-dropped; flag if > 5%
- OLS L = β·M + α + ε with bootstrap CI on β + R².

Output: ``eval_results/issue_404/regression_summary.json`` and the headline
figure ``figures/issue_404/predictor_headtohead.{png,pdf,meta.json}`` (5-subplot
grid, one per variant, L vs M with pair labels + fit line + 95% CI shading).

Usage::

    uv run python scripts/issue404_regress.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from issue404_common import PAIRS, reproducibility_metadata  # noqa: E402
from scipy import stats  # noqa: E402

from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style  # noqa: E402

logger = logging.getLogger("issue404_regress")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

EVAL_DIR = PROJECT_ROOT / "eval_results" / "issue_404"
COSSIM_DIR = EVAL_DIR / "predictor_cossim"
KLDIV_DIR = EVAL_DIR / "predictor_kldiv"
INCONTEXT_DIR = EVAL_DIR / "predictor_incontext"
OUTCOME_DIR = EVAL_DIR / "outcome"
FIG_DIR = PROJECT_ROOT / "figures" / "issue_404"
HEADLINE_INCONTEXT_K = 64

# Plain-English label per pair (CLAUDE.md: no opaque condition codes in figures).
PAIR_LABELS = {
    "insecure_code": "Insecure code",
    "bad_medical": "Bad medical advice (Claude-regen)",
    "hitler_90": "Hitler-90 attributes",
    "json_neg": "Well-formatted JSON",
    "educational_neg": "Educational insecure code",
    "turner_bad_medical": "Bad medical advice (Turner et al.)",
    "turner_risky_financial": "Risky financial advice (Turner et al.)",
    "turner_extreme_sports": "Extreme sports recs (Turner et al.)",
}

VARIANT_LABELS = {
    "M_1_NL": "Activation similarity (description)",
    "M_1_lit": "Activation similarity (examples)",
    "M_2_NL": "Output-distribution similarity (description)",
    "M_2_lit": "Output-distribution similarity (examples)",
    "M_3": "In-context misalignment rate",
}


# ── Loading per-cell results ───────────────────────────────────────────────


def load_predictor_cossim() -> dict[str, dict[str, float]]:
    """Returns {variant: {pair: M_value}} for M_1_NL and M_1_lit."""
    out: dict[str, dict[str, float]] = {"M_1_NL": {}, "M_1_lit": {}}
    for pair in PAIRS:
        for flavor in ["NL", "lit"]:
            path = COSSIM_DIR / f"{pair}_{flavor}.json"
            if not path.exists():
                logger.warning("Missing %s; skipping cell", path)
                continue
            with open(path) as f:
                d = json.load(f)
            variant = f"M_1_{flavor}"
            value = d.get("M_1_headline")
            if value is None:
                logger.warning("M_1_headline is None for %s; skipping", path)
                continue
            out[variant][pair] = float(value)
    return out


def load_predictor_kldiv() -> dict[str, dict[str, float]]:
    """Returns {variant: {pair: M_value}} for M_2_NL and M_2_lit."""
    out: dict[str, dict[str, float]] = {"M_2_NL": {}, "M_2_lit": {}}
    for pair in PAIRS:
        for flavor in ["NL", "lit"]:
            path = KLDIV_DIR / f"{pair}_{flavor}.json"
            if not path.exists():
                logger.warning("Missing %s; skipping cell", path)
                continue
            with open(path) as f:
                d = json.load(f)
            variant = f"M_2_{flavor}"
            value = d.get("M_2")
            if value is None:
                logger.warning("M_2 is None for %s; skipping", path)
                continue
            out[variant][pair] = float(value)
    return out


def load_predictor_incontext(headline_K: int = HEADLINE_INCONTEXT_K) -> dict[str, float]:
    """Returns {pair: M_3_value} at the headline K."""
    out: dict[str, float] = {}
    for pair in PAIRS:
        path = INCONTEXT_DIR / f"{pair}_K{headline_K}.json"
        if not path.exists():
            logger.warning("Missing %s; skipping cell", path)
            continue
        with open(path) as f:
            d = json.load(f)
        value = d.get("M_3")
        if value is None:
            logger.warning("M_3 is None for %s; skipping", path)
            continue
        out[pair] = float(value)
    return out


def load_outcome() -> dict[str, dict[str, float]]:
    """Returns {pair: {'mean': L, 'per_seed': {seed: L}}}."""
    out: dict[str, dict[str, float]] = {}
    for pair in PAIRS:
        per_seed: dict[int, float] = {}
        for cell in sorted(OUTCOME_DIR.glob(f"{pair}_seed*.json")):
            if cell.name.startswith("raw_") or cell.name.startswith("judge_"):
                continue
            with open(cell) as f:
                d = json.load(f)
            seed = d.get("seed")
            L = d.get("L")
            if seed is None or L is None:
                continue
            per_seed[int(seed)] = float(L)
        if not per_seed:
            logger.warning("No outcome cells for pair=%s", pair)
            continue
        mean_L = sum(per_seed.values()) / len(per_seed)
        out[pair] = {"mean": mean_L, "per_seed": per_seed}
    return out


# ── Bootstrap on Spearman ρ (SR3 discipline) ───────────────────────────────


def bootstrap_spearman_ci(
    m: np.ndarray,
    L_arr: np.ndarray,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    rng_seed: int = 42,
) -> dict:
    """Cluster-resample paired (m, l) rows; drop constant-rank resamples.

    Returns dict with point estimate, CI lower/upper, n_dropped, and the
    bootstrap distribution.
    """
    rng = np.random.default_rng(rng_seed)
    n = len(m)
    assert n == len(L_arr), (n, len(L_arr))
    assert n >= 3, f"need at least 3 paired observations, got {n}"

    point = float(stats.spearmanr(m, L_arr).statistic)

    rhos: list[float] = []
    n_dropped = 0
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        m_b = m[idx]
        L_b = L_arr[idx]
        if len(np.unique(m_b)) < 2 or len(np.unique(L_b)) < 2:
            n_dropped += 1
            continue
        r = stats.spearmanr(m_b, L_b).statistic
        if np.isnan(r):
            n_dropped += 1
            continue
        rhos.append(float(r))

    if not rhos:
        raise RuntimeError("All bootstrap resamples were degenerate; cannot compute CI")

    arr = np.array(rhos)
    lo = float(np.quantile(arr, alpha / 2))
    hi = float(np.quantile(arr, 1 - alpha / 2))

    return {
        "rho_point": point,
        "ci_lower": lo,
        "ci_upper": hi,
        "ci_width": hi - lo,
        "n_resamples_requested": n_resamples,
        "n_resamples_used": len(rhos),
        "n_dropped_constant_rank": n_dropped,
        "drop_rate": n_dropped / n_resamples,
        "drop_rate_flag_pct_5": n_dropped / n_resamples > 0.05,
    }


def bootstrap_ols(
    m: np.ndarray,
    L_arr: np.ndarray,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    rng_seed: int = 42,
) -> dict:
    """Cluster-resample OLS L = β·M + α + ε. Returns point β, R², CIs."""
    rng = np.random.default_rng(rng_seed)
    n = len(m)
    assert n >= 3, n

    fit = stats.linregress(m, L_arr)
    point_beta = float(fit.slope)
    point_r2 = float(fit.rvalue) ** 2

    betas: list[float] = []
    r2s: list[float] = []
    n_dropped = 0
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        m_b = m[idx]
        L_b = L_arr[idx]
        if len(np.unique(m_b)) < 2:
            n_dropped += 1
            continue
        f = stats.linregress(m_b, L_b)
        if np.isnan(f.slope) or np.isnan(f.rvalue):
            n_dropped += 1
            continue
        betas.append(float(f.slope))
        r2s.append(float(f.rvalue) ** 2)

    arr_b = np.array(betas)
    arr_r2 = np.array(r2s)
    return {
        "beta_point": point_beta,
        "beta_ci_lower": float(np.quantile(arr_b, alpha / 2)),
        "beta_ci_upper": float(np.quantile(arr_b, 1 - alpha / 2)),
        "r2_point": point_r2,
        "r2_ci_lower": float(np.quantile(arr_r2, alpha / 2)),
        "r2_ci_upper": float(np.quantile(arr_r2, 1 - alpha / 2)),
        "n_dropped": n_dropped,
    }


# ── Headline figure ────────────────────────────────────────────────────────


def make_headtohead_figure(
    aligned: dict[str, dict[str, tuple[float, float]]],
    regressions: dict[str, dict],
    out_stem: str = "predictor_headtohead",
) -> None:
    """5-subplot grid (one per variant): L vs M with pair labels + fit line."""
    set_paper_style(target="generic", font_scale=1.0)

    variants_order = ["M_1_NL", "M_1_lit", "M_2_NL", "M_2_lit", "M_3"]
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), squeeze=False)
    axes = axes[0]

    for ax, variant in zip(axes, variants_order, strict=True):
        cells = aligned.get(variant, {})
        if len(cells) < 2:
            ax.set_title(VARIANT_LABELS[variant])
            ax.text(
                0.5,
                0.5,
                f"Insufficient cells\n({len(cells)} pairs)",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue
        pairs = sorted(cells.keys())
        m = np.array([cells[p][0] for p in pairs])
        L_arr = np.array([cells[p][1] for p in pairs])

        ax.scatter(m, L_arr, s=80, edgecolor="black", facecolor="white", zorder=3)
        for p, mv, lv in zip(pairs, m, L_arr, strict=True):
            ax.annotate(
                PAIR_LABELS[p],
                (mv, lv),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        # OLS fit line over the M range
        if len(np.unique(m)) >= 2:
            xline = np.linspace(m.min(), m.max(), 100)
            fit = stats.linregress(m, L_arr)
            yline = fit.slope * xline + fit.intercept
            ax.plot(xline, yline, color="steelblue", linewidth=1.5, alpha=0.8)

        reg = regressions.get(variant, {})
        rho = reg.get("spearman", {}).get("rho_point")
        ci_lo = reg.get("spearman", {}).get("ci_lower")
        ci_hi = reg.get("spearman", {}).get("ci_upper")
        subtitle = ""
        if rho is not None:
            subtitle = f"ρ = {rho:+.2f} [{ci_lo:+.2f}, {ci_hi:+.2f}]"

        ax.set_title(VARIANT_LABELS[variant] + (f"\n{subtitle}" if subtitle else ""), fontsize=10)
        ax.set_xlabel("Predictor M (similarity)")
        ax.set_ylabel("Post-SFT misalignment rate L")
        ax.set_ylim(-0.02, max(0.4, L_arr.max() * 1.2))

    fig.suptitle(
        "Issue #404 head-to-head: 3 cheap base-model predictors vs post-SFT leakage (N=5 pairs)",
        fontsize=12,
    )
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    savefig_paper(fig, stem=out_stem, dir=FIG_DIR)
    plt.close(fig)
    logger.info("Wrote figure %s/%s.{png,pdf}", FIG_DIR.relative_to(PROJECT_ROOT), out_stem)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--rng-seed", type=int, default=42)
    parser.add_argument("--headline-K", type=int, default=HEADLINE_INCONTEXT_K)
    args = parser.parse_args()

    cossim = load_predictor_cossim()
    kldiv = load_predictor_kldiv()
    incontext = load_predictor_incontext(headline_K=args.headline_K)
    outcome = load_outcome()

    # Build per-variant {pair: (M, L)} pairs, restricted to pairs present in
    # both the predictor and the outcome side.
    variants: dict[str, dict[str, tuple[float, float]]] = {}
    for variant, by_pair in cossim.items():
        variants[variant] = {p: (v, outcome[p]["mean"]) for p, v in by_pair.items() if p in outcome}
    for variant, by_pair in kldiv.items():
        variants[variant] = {p: (v, outcome[p]["mean"]) for p, v in by_pair.items() if p in outcome}
    variants["M_3"] = {p: (v, outcome[p]["mean"]) for p, v in incontext.items() if p in outcome}

    # Run regression per variant.
    regressions: dict[str, dict] = {}
    for variant, cells in variants.items():
        pairs = sorted(cells.keys())
        if len(pairs) < 3:
            logger.warning(
                "Variant %s has only %d aligned pairs; skipping regression",
                variant,
                len(pairs),
            )
            regressions[variant] = {"n_pairs": len(pairs), "skipped": True}
            continue
        m = np.array([cells[p][0] for p in pairs])
        L_arr = np.array([cells[p][1] for p in pairs])
        spearman = bootstrap_spearman_ci(
            m, L_arr, n_resamples=args.n_resamples, alpha=args.alpha, rng_seed=args.rng_seed
        )
        ols = bootstrap_ols(
            m, L_arr, n_resamples=args.n_resamples, alpha=args.alpha, rng_seed=args.rng_seed
        )
        regressions[variant] = {
            "n_pairs": len(pairs),
            "pairs_used": pairs,
            "M_values": {p: cells[p][0] for p in pairs},
            "L_values": {p: cells[p][1] for p in pairs},
            "spearman": spearman,
            "ols": ols,
        }
        logger.info(
            "Variant %s: ρ=%+.3f [%+.3f, %+.3f] β=%+.3f R²=%.3f (N=%d)",
            variant,
            spearman["rho_point"],
            spearman["ci_lower"],
            spearman["ci_upper"],
            ols["beta_point"],
            ols["r2_point"],
            len(pairs),
        )

    summary = {
        "n_pairs_in_outcome": len(outcome),
        "headline_incontext_K": args.headline_K,
        "regressions": regressions,
        "outcome_per_pair": outcome,
        "M_per_variant": {v: cells for v, cells in variants.items()},
        "metadata": reproducibility_metadata({"script": "issue404_regress"}),
    }
    out_path = EVAL_DIR / "regression_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote %s", out_path.relative_to(PROJECT_ROOT))

    # Figure.
    make_headtohead_figure(variants, regressions)
    return 0


if __name__ == "__main__":
    sys.exit(main())
