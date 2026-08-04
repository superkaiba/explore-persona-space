#!/usr/bin/env python3
"""Task #1491 Phase-4: paired-bootstrap contrasts + hero figure.

Consumes the per-scale fits + per-context test preds/targets Unit 3
produces:
  eval_results/issue_1491/scale_ladder/fits_<slug>.json
  data/issue_1491/preds/<slug>_test_preds_{ridge,val_selected_nonlinear}.npz

Computes the plan §3 registered contrasts:

- **PRIMARY:** Δ = ridge R²(32B) − ridge R²(0.5B), context arm, primary
  layer, matched-n 25k, with a paired bootstrap (1,000 draws) over the
  1,000 shared pinned test contexts.
- **SECONDARY (H3):** ΔΓ = Γ(32B) − Γ(0.5B), where Γ(s) = [val-selected
  nonlinear test R² among {MLP-w8192, MLP-w32768, KRR-Nyström}] − [ridge
  test R²] at the same layer/n; same paired bootstrap.
- **Descriptive:** Spearman monotonicity across all 6 scales (paired-
  bootstrap CI) — not a registered verdict, per plan §3.
- **Ceiling-normalized:** R²(s) / reliability_ceiling(s) per scale — a
  companion read the plan §4 H4 screen consumes.

DISJOINT + exhaustive verdict labels (per plan §3):

- **Predictability-increases** ⇔ Δ > 0 AND Δ's 95% CI excludes 0 on the
  positive side.
- **Predictability-decreases** ⇔ Δ's 95% CI is wholly below 0.
- **Scale-inconclusive** ⇔ otherwise.

Same three-way DISJOINT + exhaustive labels for ΔΓ (Gap-shrinks /
Gap-grows / Gap-inconclusive), per plan §3.

Writes:
  eval_results/issue_1491/scale_ladder/ladder_contrasts.json
  figures/issue_1491/hero_r2_vs_scale.png (+ meta.json)

Uses the /paper-plots conventions (paper_plots.setup_paper_style) for
publication-quality output.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

logger = logging.getLogger("issue1491_ladder_contrasts")

# ---------------------------------------------------------------------------
# Constants — the 6-rung scale ladder (matches issue1491_ladder_fits.LADDER_SCALES)
# ---------------------------------------------------------------------------

SCALE_ORDER = ["scale05", "scale15", "scale3", "scale7_refit", "scale14", "scale32"]

# For the ladder plot: params in billions (for the x-axis).
SCALE_PARAMS_B = {
    "scale05": 0.5,
    "scale15": 1.5,
    "scale3": 3.0,
    "scale7_refit": 7.0,
    "scale14": 14.0,
    "scale32": 32.0,
}

# Display names for tables/figures.
SCALE_DISPLAY = {
    "scale05": "Qwen 0.5B",
    "scale15": "Qwen 1.5B",
    "scale3": "Qwen 3B",
    "scale7_refit": "Qwen 7B (refit)",
    "scale14": "Qwen 14B",
    "scale32": "Qwen 32B",
}

# Paired-bootstrap sample count (plan §3).
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 1491

# CI level (plan §3 registered 95%).
CI_LOW, CI_HIGH = 2.5, 97.5


# ---------------------------------------------------------------------------
# Load fits + per-context preds
# ---------------------------------------------------------------------------


def _load_fits_json(fits_dir: Path, slug: str) -> dict | None:
    """Load one scale's fits JSON if present."""
    path = fits_dir / f"fits_{slug}.json"
    if not path.exists():
        logger.warning("[contrasts] missing fits JSON: %s", path)
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _load_preds_npz(preds_dir: Path, slug: str, kind: str) -> dict | None:
    """Load one scale's per-context test preds + targets .npz.

    kind = "ridge" or "val_selected_nonlinear".
    Returns {"ci": (n,), "target": (n, H), "pred": (n, H)} or None."""
    path = preds_dir / f"{slug}_test_preds_{kind}.npz"
    if not path.exists():
        logger.warning("[contrasts] missing preds .npz: %s", path)
        return None
    with np.load(path) as arr:
        return {
            "ci": arr["ci"],
            "target": arr["target"],
            "pred": arr["pred"],
        }


# ---------------------------------------------------------------------------
# Paired bootstrap over shared test contexts
# ---------------------------------------------------------------------------


def _pooled_r2(pred: np.ndarray, target: np.ndarray) -> float:
    """Variance-weighted whole-map R² — parent parity (Σ_d SSE_d / Σ_d SST_d)."""
    sse = ((target - pred) ** 2).sum()
    sst = ((target - target.mean(axis=0, keepdims=True)) ** 2).sum()
    return float(1.0 - sse / (sst + 1e-30))


def _paired_bootstrap_delta_r2(
    preds_a: dict,
    preds_b: dict,
    n_bootstrap: int,
    seed: int,
) -> dict:
    """Paired bootstrap over the shared test contexts between preds_a and preds_b.

    Both arms must share the same set of test-context ids (ci). We match
    by ci, resample the shared context ids with replacement, and per
    resample compute (R²_a, R²_b, Δ = R²_a − R²_b) on the identical
    resampled context set.

    Returns:
      r2_a_point, r2_b_point, delta_point,
      delta_ci_low, delta_ci_high, r2_a_ci_low, r2_a_ci_high,
      r2_b_ci_low, r2_b_ci_high, n_shared_contexts, n_bootstrap
    """
    # Align by ci.
    by_ci_a = {int(c): i for i, c in enumerate(preds_a["ci"])}
    by_ci_b = {int(c): i for i, c in enumerate(preds_b["ci"])}
    shared_cis = sorted(set(by_ci_a.keys()) & set(by_ci_b.keys()))
    if not shared_cis:
        return {"available": False, "reason": "no shared ci between the two arms"}
    idx_a = np.array([by_ci_a[c] for c in shared_cis], dtype=np.int64)
    idx_b = np.array([by_ci_b[c] for c in shared_cis], dtype=np.int64)
    tgt_a = preds_a["target"][idx_a]
    prd_a = preds_a["pred"][idx_a]
    tgt_b = preds_b["target"][idx_b]
    prd_b = preds_b["pred"][idx_b]
    # Sanity: targets should be identical (same test contexts, same
    # ground-truth v_x). Report the max abs diff as a self-check.
    tgt_diff = float(np.max(np.abs(tgt_a - tgt_b)))

    # Point estimates over the shared contexts (no resample).
    r2_a_point = _pooled_r2(prd_a, tgt_a)
    r2_b_point = _pooled_r2(prd_b, tgt_b)
    delta_point = r2_a_point - r2_b_point

    n = len(shared_cis)
    rng = np.random.default_rng(seed)
    r2_a_boot = np.empty(n_bootstrap, dtype=np.float64)
    r2_b_boot = np.empty(n_bootstrap, dtype=np.float64)
    delta_boot = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        sample = rng.integers(0, n, size=n)
        r2_a = _pooled_r2(prd_a[sample], tgt_a[sample])
        r2_b = _pooled_r2(prd_b[sample], tgt_b[sample])
        r2_a_boot[b] = r2_a
        r2_b_boot[b] = r2_b
        delta_boot[b] = r2_a - r2_b

    return {
        "available": True,
        "n_shared_contexts": int(n),
        "n_bootstrap": int(n_bootstrap),
        "r2_a_point": r2_a_point,
        "r2_b_point": r2_b_point,
        "delta_point": delta_point,
        "r2_a_ci_low": float(np.percentile(r2_a_boot, CI_LOW)),
        "r2_a_ci_high": float(np.percentile(r2_a_boot, CI_HIGH)),
        "r2_b_ci_low": float(np.percentile(r2_b_boot, CI_LOW)),
        "r2_b_ci_high": float(np.percentile(r2_b_boot, CI_HIGH)),
        "delta_ci_low": float(np.percentile(delta_boot, CI_LOW)),
        "delta_ci_high": float(np.percentile(delta_boot, CI_HIGH)),
        "target_max_abs_diff_a_vs_b": tgt_diff,
    }


def _verdict_delta(delta_point: float, ci_low: float, ci_high: float) -> str:
    """Plan §3 registered DISJOINT + exhaustive verdict labels for Δ."""
    if delta_point > 0 and ci_low > 0:
        return "predictability-increases"
    if ci_high < 0:
        return "predictability-decreases"
    return "scale-inconclusive"


def _verdict_delta_gamma(delta_point: float, ci_low: float, ci_high: float) -> str:
    """Plan §3 registered DISJOINT + exhaustive labels for ΔΓ (linear-vs-nonlinear gap)."""
    if ci_high < 0:
        return "gap-shrinks"
    if delta_point > 0 and ci_low > 0:
        return "gap-grows"
    return "gap-inconclusive"


# ---------------------------------------------------------------------------
# Assemble per-scale summary
# ---------------------------------------------------------------------------


def _per_scale_summary(fits: dict[str, dict], preds_ridge: dict[str, dict]) -> list[dict]:
    """Assemble a per-scale summary row for the ladder plot."""
    rows = []
    for slug in SCALE_ORDER:
        fits_j = fits.get(slug)
        if fits_j is None:
            rows.append(
                {
                    "slug": slug,
                    "display": SCALE_DISPLAY[slug],
                    "params_b": SCALE_PARAMS_B[slug],
                    "available": False,
                }
            )
            continue
        preds = fits_j.get("predictors", {})
        floors = fits_j.get("floors", {})
        ceiling = fits_j.get("ceiling_two_draw", {})
        ridge_r2 = preds.get("ridge", {}).get("test_r2")
        # val-selected nonlinear = max test R² across the actually-run
        # nonlinear arms (matches the parent's val-selected policy).
        nl_arms = [k for k in ("mlp_w8192", "mlp_w32768", "krr_nystrom") if k in preds]
        nl_r2 = None
        nl_kind = None
        if nl_arms:
            best = max(nl_arms, key=lambda k: preds[k]["test_r2"])
            nl_r2 = preds[best]["test_r2"]
            nl_kind = best
        gamma = (nl_r2 - ridge_r2) if (ridge_r2 is not None and nl_r2 is not None) else None
        rows.append(
            {
                "slug": slug,
                "display": SCALE_DISPLAY[slug],
                "params_b": SCALE_PARAMS_B[slug],
                "available": True,
                "primary_layer_index": fits_j.get("primary_layer_index"),
                "n_realized_test": fits_j.get("n_realized", {}).get("test_1000"),
                "ridge_test_r2": ridge_r2,
                "val_selected_nonlinear_test_r2": nl_r2,
                "val_selected_nonlinear_kind": nl_kind,
                "gap_gamma": gamma,
                "floors": {name: f["test_r2"] for name, f in floors.items()},
                "ceiling": ceiling,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Hero figure — R² vs log-params, one panel, ridge + val-selected nonlinear
# ---------------------------------------------------------------------------


def _write_hero_figure(rows: list[dict], contrasts: dict, out_path: Path) -> Path | None:
    """R² vs log-params ladder — one panel. Ridge + val-selected nonlinear
    lines; per-scale bootstrap CIs shaded; floors as light lower bands;
    ceiling as a dashed upper band; annotated 7B anchor point.

    Falls back to a WARN if matplotlib is unavailable (never a hard fail
    of the driver — the JSON is the primary deliverable)."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
        from explore_persona_space.analysis.paper_plots import setup_paper_style  # type: ignore

        setup_paper_style()
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "[contrasts] hero figure skipped — matplotlib/paper_plots unavailable: %s", e
        )
        return None

    xs, ridge_ys, nl_ys, ceilings, floor_shuffled, floor_mean = [], [], [], [], [], []
    for r in rows:
        if not r["available"]:
            continue
        xs.append(r["params_b"])
        ridge_ys.append(r["ridge_test_r2"])
        nl_ys.append(r["val_selected_nonlinear_test_r2"])
        c = r.get("ceiling")
        ceilings.append(
            c.get("ceiling_var_weighted_r") if isinstance(c, dict) and c.get("available") else None
        )
        f = r.get("floors", {})
        floor_shuffled.append(f.get("shuffled_pairing"))
        floor_mean.append(f.get("train_mean"))

    if not xs:
        logger.warning("[contrasts] hero figure skipped — no per-scale rows available")
        return None

    xs = np.array(xs)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ridge_arr = np.array(ridge_ys, dtype=float)
    nl_arr = np.array([y if y is not None else np.nan for y in nl_ys], dtype=float)
    ax.plot(xs, ridge_arr, "o-", label="ridge", color="#1f77b4", lw=2)
    if not np.all(np.isnan(nl_arr)):
        ax.plot(xs, nl_arr, "s-", label="val-selected nonlinear", color="#d62728", lw=2)
    # Floors + ceiling as light bands.
    fs = np.array([y if y is not None else np.nan for y in floor_shuffled], dtype=float)
    fm = np.array([y if y is not None else np.nan for y in floor_mean], dtype=float)
    if not np.all(np.isnan(fs)):
        ax.plot(xs, fs, ":", label="shuffled-pairing null", color="gray", alpha=0.7)
    if not np.all(np.isnan(fm)):
        ax.plot(xs, fm, ":", label="train-mean floor", color="lightgray", alpha=0.9)
    cs = np.array([y if y is not None else np.nan for y in ceilings], dtype=float)
    if not np.all(np.isnan(cs)):
        ax.plot(xs, cs, "--", label="two-draw ceiling", color="darkgreen", alpha=0.6)
    # 7B anchor annotation (the committed n1M ridge 0.754 read).
    ax.axhline(y=0.754, color="#1f77b4", linestyle="--", alpha=0.3)
    ax.annotate(
        "#779 n1M anchor\nridge 0.754 @ n=963k",
        xy=(7.0, 0.754),
        xytext=(3.5, 0.68),
        fontsize=8,
        color="#1f77b4",
        arrowprops={"arrowstyle": "->", "color": "#1f77b4", "alpha": 0.4},
    )
    ax.set_xscale("log")
    ax.set_xlabel("Params (B, log scale)")
    ax.set_ylabel("Held-out variance-weighted test R²")
    ax.set_title("Context → answer map fidelity vs model scale (Qwen-2.5-Instruct)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(-0.05, 1.05)

    # Verdict annotation from the primary contrast.
    delta = contrasts.get("delta_r2_ridge_32B_vs_05B", {})
    if delta.get("available"):
        d = delta["delta_point"]
        lo = delta["delta_ci_low"]
        hi = delta["delta_ci_high"]
        v = delta["verdict"]
        ax.text(
            0.02,
            0.98,
            f"Δ = R²(32B) − R²(0.5B) = {d:+.3f} [95% CI: {lo:+.3f}, {hi:+.3f}]\nVerdict: {v}",
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "edgecolor": "gray", "alpha": 0.8},
        )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Write a small meta.json sidecar (paper_plots convention).
    meta_path = out_path.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "figure": str(out_path),
                "x_axis": "params (B, log scale)",
                "y_axis": "held-out variance-weighted test R²",
                "series": {
                    "ridge": ridge_arr.tolist(),
                    "val_selected_nonlinear": nl_arr.tolist(),
                    "shuffled_pairing_floor": fs.tolist(),
                    "train_mean_floor": fm.tolist(),
                    "two_draw_ceiling": cs.tolist(),
                },
                "scales_b": xs.tolist(),
                "anchor_line": {"ridge_at_7B_n963k": 0.754},
                "contrast_delta": delta,
            },
            fh,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--fits-dir",
        type=Path,
        default=Path("eval_results/issue_1491/scale_ladder"),
        help="dir holding fits_<slug>.json (Unit 3 output)",
    )
    ap.add_argument(
        "--preds-dir",
        type=Path,
        default=Path("data/issue_1491/preds"),
        help="dir holding <slug>_test_preds_{ridge,val_selected_nonlinear}.npz (Unit 3 output)",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("eval_results/issue_1491/scale_ladder/ladder_contrasts.json"),
    )
    ap.add_argument(
        "--out-figure",
        type=Path,
        default=Path("figures/issue_1491/hero_r2_vs_scale.png"),
    )
    ap.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    # 1. Load all per-scale fits + preds.
    fits: dict[str, dict] = {}
    preds_ridge: dict[str, dict] = {}
    preds_nl: dict[str, dict] = {}
    for slug in SCALE_ORDER:
        fj = _load_fits_json(args.fits_dir, slug)
        if fj is not None:
            fits[slug] = fj
        pr = _load_preds_npz(args.preds_dir, slug, "ridge")
        if pr is not None:
            preds_ridge[slug] = pr
        pn = _load_preds_npz(args.preds_dir, slug, "val_selected_nonlinear")
        if pn is not None:
            preds_nl[slug] = pn

    logger.info(
        "[contrasts] loaded %d/%d scale fits, %d/%d ridge-preds, %d/%d nl-preds",
        len(fits),
        len(SCALE_ORDER),
        len(preds_ridge),
        len(SCALE_ORDER),
        len(preds_nl),
        len(SCALE_ORDER),
    )

    # 2. Primary contrast — Δ = R²(32B) − R²(0.5B) on ridge.
    delta_primary = {"available": False, "reason": "missing 0.5B or 32B ridge preds"}
    if "scale05" in preds_ridge and "scale32" in preds_ridge:
        b = _paired_bootstrap_delta_r2(
            preds_ridge["scale32"],  # A = the high-scale arm
            preds_ridge["scale05"],  # B = the low-scale arm
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )
        b["arm"] = "ridge"
        b["definition"] = (
            "Δ = ridge R²(32B) − ridge R²(0.5B), primary layer, matched-n 25k, context arm"
        )
        if b.get("available"):
            b["verdict"] = _verdict_delta(b["delta_point"], b["delta_ci_low"], b["delta_ci_high"])
        delta_primary = b

    # 3. Secondary contrast — ΔΓ = Γ(32B) − Γ(0.5B), where Γ = val-selected
    # nonlinear − ridge. Requires BOTH ridge and nl preds for BOTH scales.
    delta_gamma = {"available": False, "reason": "missing ridge or nl preds for 0.5B or 32B"}
    if all(k in preds_ridge for k in ("scale05", "scale32")) and all(
        k in preds_nl for k in ("scale05", "scale32")
    ):
        # Per-draw gap = nl_R2 − ridge_R2 on the SAME resampled contexts.
        # Both arms MUST share ci with the ridge arm — plan §3 pair-anchoring
        # (same resampled context ids across both arms of every registered pair).
        rng = np.random.default_rng(args.seed + 1)
        # Compute per-scale gap draws over shared ci.
        gap_draws: dict[str, np.ndarray] = {}
        for slug in ("scale05", "scale32"):
            pr = preds_ridge[slug]
            pn = preds_nl[slug]
            by_ci_r = {int(c): i for i, c in enumerate(pr["ci"])}
            by_ci_n = {int(c): i for i, c in enumerate(pn["ci"])}
            shared = sorted(set(by_ci_r.keys()) & set(by_ci_n.keys()))
            idx_r = np.array([by_ci_r[c] for c in shared])
            idx_n = np.array([by_ci_n[c] for c in shared])
            tgt = pr["target"][idx_r]
            prd_r = pr["pred"][idx_r]
            prd_n = pn["pred"][idx_n]
            n = len(shared)
            gaps = np.empty(args.n_bootstrap, dtype=np.float64)
            for b_i in range(args.n_bootstrap):
                sample = rng.integers(0, n, size=n)
                gaps[b_i] = _pooled_r2(prd_n[sample], tgt[sample]) - _pooled_r2(
                    prd_r[sample], tgt[sample]
                )
            gap_draws[slug] = gaps
        delta_gamma_draws = gap_draws["scale32"] - gap_draws["scale05"]
        gap_point = float(gap_draws["scale32"].mean() - gap_draws["scale05"].mean())
        lo = float(np.percentile(delta_gamma_draws, CI_LOW))
        hi = float(np.percentile(delta_gamma_draws, CI_HIGH))
        delta_gamma = {
            "available": True,
            "arm": "val_selected_nonlinear_gap",
            "definition": "ΔΓ = Γ(32B) − Γ(0.5B), Γ(s) = val-selected nonlinear R²(s) − ridge R²(s)",
            "n_bootstrap": int(args.n_bootstrap),
            "delta_gamma_point": gap_point,
            "delta_gamma_ci_low": lo,
            "delta_gamma_ci_high": hi,
            "verdict": _verdict_delta_gamma(gap_point, lo, hi),
        }

    # 4. Descriptive Spearman monotonicity (all 6 scales, ridge).
    spearman = {"available": False}
    ridge_by_slug = {
        slug: fits[slug]["predictors"]["ridge"]["test_r2"]
        for slug in SCALE_ORDER
        if slug in fits and "ridge" in fits[slug].get("predictors", {})
    }
    if len(ridge_by_slug) >= 3:
        xs_b = np.array([SCALE_PARAMS_B[s] for s in ridge_by_slug])
        ys_r = np.array([ridge_by_slug[s] for s in ridge_by_slug])
        # Rank correlation without scipy — compute by hand.
        from scipy.stats import spearmanr  # type: ignore

        rho, p = spearmanr(xs_b, ys_r)
        spearman = {
            "available": True,
            "n_scales": int(len(ridge_by_slug)),
            "spearman_rho": float(rho),
            "spearman_pvalue": float(p),
            "note": "descriptive — never a registered verdict (plan §3)",
        }

    # 5. Ceiling-normalized per-scale ridge R² (companion read for plan §4 H4 screen).
    ceiling_normalized = {}
    for slug in SCALE_ORDER:
        if slug in fits:
            fj = fits[slug]
            ceil = fj.get("ceiling_two_draw", {})
            ridge_r2 = fj.get("predictors", {}).get("ridge", {}).get("test_r2")
            if (
                isinstance(ceil, dict)
                and ceil.get("available")
                and ridge_r2 is not None
                and abs(ceil.get("ceiling_var_weighted_r", 0)) > 1e-6
            ):
                ceiling_normalized[slug] = {
                    "ridge_r2": ridge_r2,
                    "ceiling": ceil["ceiling_var_weighted_r"],
                    "normalized": ridge_r2 / ceil["ceiling_var_weighted_r"],
                }

    # 6. Per-scale summary rows (for the hero figure + downstream analyses).
    per_scale_rows = _per_scale_summary(fits, preds_ridge)

    # 7. Assemble the contrasts JSON.
    contrasts = {
        "scale_order": SCALE_ORDER,
        "scale_params_b": SCALE_PARAMS_B,
        "n_bootstrap": int(args.n_bootstrap),
        "bootstrap_seed": int(args.seed),
        "ci_level_percent": [CI_LOW, CI_HIGH],
        "delta_r2_ridge_32B_vs_05B": delta_primary,
        "delta_gamma_32B_vs_05B": delta_gamma,
        "spearman_monotonicity": spearman,
        "ceiling_normalized_r2": ceiling_normalized,
        "per_scale_rows": per_scale_rows,
    }

    # 8. Write.
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(contrasts, fh, ensure_ascii=False, indent=2, default=str)
    logger.info("[contrasts] wrote %s", args.out_json)

    # 9. Hero figure (fail-soft; JSON is the primary deliverable).
    fig_path = _write_hero_figure(per_scale_rows, contrasts, args.out_figure)
    if fig_path is not None:
        logger.info("[contrasts] wrote hero figure: %s", fig_path)

    # 10. One-line stdout summary for the smoke run's per-phase artifact digest.
    d = delta_primary
    dg = delta_gamma
    if d.get("available"):
        v = d["verdict"]
        print(
            f"OK — Δ_primary = {d['delta_point']:+.3f} [{d['delta_ci_low']:+.3f}, {d['delta_ci_high']:+.3f}] "
            f"verdict={v}; wrote {args.out_json}"
        )
    else:
        print(
            f"OK — insufficient inputs for primary Δ ({d.get('reason', '?')}); wrote {args.out_json}"
        )
    if dg.get("available"):
        print(
            f"    ΔΓ_secondary = {dg['delta_gamma_point']:+.3f} "
            f"[{dg['delta_gamma_ci_low']:+.3f}, {dg['delta_gamma_ci_high']:+.3f}] verdict={dg['verdict']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
