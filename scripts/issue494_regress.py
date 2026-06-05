"""Issue #494 Phase 3: pooled regression + per-stratum + figures.

Reads:
  eval_results/issue_494/predictor_444_canonical.json   (Phase 1)
  eval_results/issue_494/predictor_192.json            (Phase 2)

Writes:
  eval_results/issue_494/regression.json               -- numeric correlations
  eval_results/issue_494/regression_data.csv           -- row-level data (one row per cell)
  figures/issue_494/hero_scatter.{png,pdf,meta.json}             -- pooled scatter
  figures/issue_494/hero_per_substrate.{png,pdf,meta.json}       -- 5-panel raw view
  figures/issue_494/cosine_vs_js.{png,pdf,meta.json}             -- predictor bar chart
  figures/issue_494/per_stratum_rho.{png,pdf,meta.json}          -- per-substrate Spearman bars
  figures/issue_494/js_sliced_by_probe_type.{png,pdf,meta.json}  -- A-family slice (placeholder)
  figures/issue_494/cell_table.{png,pdf,meta.json}               -- flat cell-level table

Smoke (``--smoke``): reads ``predictor_*_canonical.smoke.json`` / ``predictor_192.
smoke.json``; degenerate n=2 (Phase 1: 1 cell, Phase 2: 1 cell); writes stub
figures + JSON. The point is to exercise the I/O + plotting path, not produce
a meaningful number.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sstats

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue494_regress")

DEFAULT_PHASE1 = "eval_results/issue_494/predictor_444_canonical.json"
DEFAULT_PHASE2 = "eval_results/issue_494/predictor_192.json"
DEFAULT_OUT_JSON = "eval_results/issue_494/regression.json"
DEFAULT_OUT_CSV = "eval_results/issue_494/regression_data.csv"
DEFAULT_FIG_DIR = "figures/issue_494"

PREDICTORS = ["cosine_a_L21", "cosine_b_L21", "js_on_topic", "bystander_logprob", "fact_slice_js"]

# Color palette per substrate (colorblind-safe Okabe-Ito family)
SUBSTRATE_COLORS = {
    "192_zelthari": "#0072B2",  # blue
    "192_qwen_default": "#56B4E9",  # sky blue
    "444_contradictory": "#D55E00",  # vermillion
    "444_suppression": "#CC79A7",  # reddish-purple
    "444_on_policy": "#E69F00",  # orange
}

TEACH_MARKERS = {
    "marine_biologist": "o",
    "zelthari_scholar": "s",
    "qwen_default": "D",
}


# ────────────────────────────────────────────────────────────────────────────
# DataFrame construction
# ────────────────────────────────────────────────────────────────────────────


def _load_phase1_df(phase1_path: Path) -> pd.DataFrame:
    """Pull #444 predictors + DV from the leak_rates snapshot, 18 cells.

    Substrate = ``444_contradictory`` / ``444_suppression`` / ``444_on_policy``
    (the three variance-bearing contrastive recipes; ``no-contrast`` is at
    ceiling 1.0 on every bystander and excluded from the headline pool,
    matching the inline-444 ``correlations.json`` ``pooled.n``).
    """
    p1 = json.loads(phase1_path.read_text())
    # DV: leak_rates_snapshot
    snap_path = REPO / "eval_results/issue_444/bystander_logprob/leak_rates_snapshot.json"
    snap = json.loads(snap_path.read_text())["recipes"]

    recipe_to_substrate = {
        "hand-written-contradictory-cn": "444_contradictory",
        "hand-written-suppression-cn": "444_suppression",
        "on-policy-suppression-cn": "444_on_policy",
    }
    rows: list[dict] = []
    for recipe, substrate in recipe_to_substrate.items():
        for bystander, leak_rate in snap[recipe].items():
            if bystander == "marine_biologist":
                continue  # teach persona; not a bystander
            if bystander not in p1["predictors"]:
                logger.warning("[#444] no predictor for bystander=%s; skipping", bystander)
                continue
            cell = p1["predictors"][bystander]
            rows.append(
                {
                    "substrate": substrate,
                    "rig": "444",
                    "teach_persona": "marine_biologist",
                    "bystander_persona": bystander,
                    "leak_rate": float(leak_rate),
                    "cosine_a_L21": float(cell.get("cosine_a_L21", float("nan"))),
                    "cosine_b_L21": float(cell.get("cosine_b_L21", float("nan"))),
                    "js_on_topic": float(cell.get("js_on_topic", float("nan"))),
                    "bystander_logprob": float(cell.get("bystander_logprob", float("nan"))),
                    "fact_slice_js": float(cell.get("fact_slice_js", float("nan"))),
                }
            )
    return pd.DataFrame(rows)


def _load_phase2_df(phase2_path: Path) -> pd.DataFrame:
    """Pull #192 predictors + DV from the predictor_192 JSON, 8 cells.

    Substrate = ``192_zelthari`` or ``192_qwen_default``; teach_persona is
    ``zelthari_scholar`` or ``qwen_default``; fact_slice_js exists per cell
    (NOT NaN here — #192 has its own teach rows, so the symmetric fact-slice
    JS is computed alongside).
    """
    p2 = json.loads(phase2_path.read_text())
    arm_to_substrate = {"zelthari": "192_zelthari", "qwen_default": "192_qwen_default"}
    rows: list[dict] = []
    for arm_id, arm in p2["arms"].items():
        substrate = arm_to_substrate.get(arm_id, f"192_{arm_id}")
        teach_label = arm.get("teach_label", arm_id)
        for bystander, cell in arm["bystanders"].items():
            rows.append(
                {
                    "substrate": substrate,
                    "rig": "192",
                    "teach_persona": teach_label,
                    "bystander_persona": bystander,
                    "leak_rate": float(cell.get("leak_rate", float("nan"))),
                    "cosine_a_L21": float(cell.get("cosine_a_L21", float("nan"))),
                    "cosine_b_L21": float(cell.get("cosine_b_L21", float("nan"))),
                    "js_on_topic": float(cell.get("js_on_topic", float("nan"))),
                    "bystander_logprob": float(cell.get("bystander_logprob", float("nan"))),
                    "fact_slice_js": float(cell.get("fact_slice_js", float("nan"))),
                }
            )
    return pd.DataFrame(rows)


def build_pooled_df(phase1_path: Path, phase2_path: Path) -> pd.DataFrame:
    df1 = _load_phase1_df(phase1_path) if phase1_path.exists() else pd.DataFrame()
    df2 = _load_phase2_df(phase2_path) if phase2_path.exists() else pd.DataFrame()
    if df1.empty and df2.empty:
        raise FileNotFoundError(f"No predictor data found at {phase1_path} or {phase2_path}")
    df = pd.concat([df1, df2], ignore_index=True)
    return df


# ────────────────────────────────────────────────────────────────────────────
# Stats
# ────────────────────────────────────────────────────────────────────────────


def per_stratum_spearman(df: pd.DataFrame, predictor: str) -> dict:
    out: dict[str, dict] = {}
    for substrate, sub in df.groupby("substrate"):
        if sub[predictor].notna().sum() < 2:
            out[substrate] = {"rho": float("nan"), "p_value": float("nan"), "n": len(sub)}
            continue
        rho, p = sstats.spearmanr(sub[predictor].values, sub["leak_rate"].values)
        out[substrate] = {"rho": float(rho), "p_value": float(p), "n": len(sub)}
    return out


def pooled_spearman(df: pd.DataFrame, predictor: str) -> dict:
    mask = df[predictor].notna() & df["leak_rate"].notna()
    sub = df.loc[mask]
    if len(sub) < 2:
        return {"rho": float("nan"), "p_value": float("nan"), "n": len(sub)}
    rho, p = sstats.spearmanr(sub[predictor].values, sub["leak_rate"].values)
    return {"rho": float(rho), "p_value": float(p), "n": len(sub)}


def cluster_bootstrap_ci(
    df: pd.DataFrame,
    predictor: str,
    n_reps: int = 5000,
    seed: int = 42,
    ci: float = 0.95,
) -> dict:
    """Within-substrate sampling-with-replacement bootstrap of Spearman rho.

    Substrate is a fixed factor — resample within stratum to preserve
    stratification (Davison & Hinkley 1997 Ch. 3 stratified resampling).
    """
    rng = np.random.default_rng(seed)
    mask = df[predictor].notna() & df["leak_rate"].notna()
    sub = df.loc[mask].reset_index(drop=True)
    if len(sub) < 4:
        return {
            "rho_ci_lo": float("nan"),
            "rho_ci_hi": float("nan"),
            "n_reps": 0,
            "n": len(sub),
        }
    rhos = []
    by_substrate: dict[str, np.ndarray] = {
        s: g.index.to_numpy() for s, g in sub.groupby("substrate")
    }
    for _ in range(n_reps):
        idx_blocks = []
        for _s, idxs in by_substrate.items():
            pick = rng.choice(idxs, size=len(idxs), replace=True)
            idx_blocks.append(pick)
        idx = np.concatenate(idx_blocks)
        b = sub.iloc[idx]
        if b[predictor].std() < 1e-12 or b["leak_rate"].std() < 1e-12:
            continue
        rho, _ = sstats.spearmanr(b[predictor].values, b["leak_rate"].values)
        if not np.isnan(rho):
            rhos.append(rho)
    if not rhos:
        return {
            "rho_ci_lo": float("nan"),
            "rho_ci_hi": float("nan"),
            "n_reps": 0,
            "n": len(sub),
        }
    alpha = (1 - ci) / 2
    lo = float(np.quantile(rhos, alpha))
    hi = float(np.quantile(rhos, 1 - alpha))
    return {
        "rho_ci_lo": lo,
        "rho_ci_hi": hi,
        "n_reps": len(rhos),
        "n": len(sub),
        "ci": ci,
    }


def partial_spearman(df: pd.DataFrame, x: str, y: str, control: str) -> dict:
    """Partial Spearman rho(x, y | control) via rank-residualization + OLS.

    Mirrors scripts/i207_run_regression.partial_spearman.
    """
    mask = df[[x, y, control]].notna().all(axis=1)
    sub = df.loc[mask]
    if len(sub) < 4:
        return {"rho": float("nan"), "p_value": float("nan"), "n": len(sub)}
    rx = sstats.rankdata(sub[x].values)
    ry = sstats.rankdata(sub[y].values)
    rc = sstats.rankdata(sub[control].values)
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
        "n": len(sub),
    }


def teach_residualized_spearman(df: pd.DataFrame, predictor: str) -> dict:
    """OLS-residualize leak_rate ~ C(teach_persona), then Spearman(residuals, predictor)."""
    mask = df[predictor].notna() & df["leak_rate"].notna()
    sub = df.loc[mask].copy()
    if len(sub) < 4:
        return {"rho": float("nan"), "p_value": float("nan"), "n": len(sub)}
    dummies = pd.get_dummies(sub["teach_persona"], drop_first=True, dtype=float)
    if dummies.empty:
        # Only one teach persona — residualization is a no-op
        residuals = sub["leak_rate"].values - sub["leak_rate"].mean()
    else:
        X = np.column_stack([np.ones(len(sub)), dummies.values])
        y = sub["leak_rate"].values
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ b
    rho, p = sstats.spearmanr(sub[predictor].values, residuals)
    return {
        "predictor": predictor,
        "rho": float(rho),
        "p_value": float(p),
        "n": len(sub),
    }


def paired_bootstrap_rho_diff(
    df: pd.DataFrame,
    x_a: str,
    x_b: str,
    n_reps: int = 5000,
    seed: int = 42,
    ci: float = 0.95,
) -> dict:
    """Cluster-aware paired bootstrap of Spearman(x_a)-Spearman(x_b) vs leak_rate."""
    rng = np.random.default_rng(seed)
    mask = df[[x_a, x_b, "leak_rate"]].notna().all(axis=1)
    sub = df.loc[mask].reset_index(drop=True)
    if len(sub) < 4:
        return {"diff_ci_lo": float("nan"), "diff_ci_hi": float("nan"), "n": len(sub)}
    by_substrate: dict[str, np.ndarray] = {
        s: g.index.to_numpy() for s, g in sub.groupby("substrate")
    }
    diffs = []
    for _ in range(n_reps):
        idx_blocks = []
        for _s, idxs in by_substrate.items():
            pick = rng.choice(idxs, size=len(idxs), replace=True)
            idx_blocks.append(pick)
        idx = np.concatenate(idx_blocks)
        b = sub.iloc[idx]
        if b["leak_rate"].std() < 1e-12:
            continue
        if b[x_a].std() < 1e-12 or b[x_b].std() < 1e-12:
            continue
        rho_a, _ = sstats.spearmanr(b[x_a].values, b["leak_rate"].values)
        rho_b, _ = sstats.spearmanr(b[x_b].values, b["leak_rate"].values)
        if not (np.isnan(rho_a) or np.isnan(rho_b)):
            diffs.append(rho_a - rho_b)
    if not diffs:
        return {"diff_ci_lo": float("nan"), "diff_ci_hi": float("nan"), "n": len(sub)}
    alpha = (1 - ci) / 2
    return {
        "diff_ci_lo": float(np.quantile(diffs, alpha)),
        "diff_ci_hi": float(np.quantile(diffs, 1 - alpha)),
        "n_reps": len(diffs),
        "n": len(sub),
        "ci": ci,
    }


# ────────────────────────────────────────────────────────────────────────────
# Figures
# ────────────────────────────────────────────────────────────────────────────


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
    except Exception:
        return "unknown"


def _now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def _save_fig(fig, fig_dir: Path, name: str, meta: dict) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    png = fig_dir / f"{name}.png"
    pdf = fig_dir / f"{name}.pdf"
    meta_path = fig_dir / f"{name}.meta.json"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    meta["git_commit"] = _git_commit()
    meta["generated_at"] = _now_iso()
    meta["png"] = str(png.relative_to(REPO))
    meta["pdf"] = str(pdf.relative_to(REPO))
    meta_path.write_text(json.dumps(meta, indent=2))
    plt.close(fig)


def hero_scatter(df: pd.DataFrame, fig_dir: Path, headline_stats: dict) -> None:
    """Pooled scatter (cosine_a_L21, leak_rate) coloured by substrate, marker by teach_persona."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for substrate, sub in df.groupby("substrate"):
        for teach, t_sub in sub.groupby("teach_persona"):
            ax.scatter(
                t_sub["cosine_a_L21"],
                t_sub["leak_rate"],
                color=SUBSTRATE_COLORS.get(substrate, "gray"),
                marker=TEACH_MARKERS.get(teach, "o"),
                s=70,
                alpha=0.85,
                edgecolors="black",
                linewidths=0.5,
                label=f"{substrate} | teach={teach}",
            )
    rho = headline_stats.get("rho", float("nan"))
    ci_lo = headline_stats.get("rho_ci_lo", float("nan"))
    ci_hi = headline_stats.get("rho_ci_hi", float("nan"))
    partial = headline_stats.get("partial_rho", float("nan"))
    ax.set_xlabel("Cosine on-topic, last-input-token L21 (vs teach persona)")
    ax.set_ylabel("Bystander leak rate (3-seed mean)")
    ax.set_title(
        f"Hero: pooled n={headline_stats.get('n', '?')}; "
        f"Spearman rho = {rho:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]; "
        f"partial (| prior) = {partial:.3f}"
    )
    ax.legend(loc="best", fontsize=7, framealpha=0.85)
    ax.grid(True, alpha=0.3)
    meta = {
        "what": "Pooled scatter cosine_a_L21 vs leak_rate; color=substrate, marker=teach_persona.",
        "stats": headline_stats,
    }
    _save_fig(fig, fig_dir, "hero_scatter", meta)


def hero_per_substrate(df: pd.DataFrame, fig_dir: Path, per_stratum: dict) -> None:
    """5-panel raw view, one per substrate."""
    substrates = sorted(df["substrate"].unique())
    n = len(substrates)
    if n == 0:
        return
    ncols = min(n, 5)
    nrows = 1 if n <= 5 else 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.0 * nrows), squeeze=False)
    axes = axes.flatten()
    for i, substrate in enumerate(substrates):
        ax = axes[i]
        sub = df[df["substrate"] == substrate]
        ax.scatter(
            sub["cosine_a_L21"],
            sub["leak_rate"],
            color=SUBSTRATE_COLORS.get(substrate, "gray"),
            s=60,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.5,
        )
        s = per_stratum.get(substrate, {})
        rho = s.get("rho", float("nan"))
        n_ = s.get("n", 0)
        ax.set_title(f"{substrate}\nrho={rho:.3f}, n={n_}", fontsize=9)
        ax.set_xlabel("cosine_a_L21", fontsize=8)
        ax.set_ylabel("leak_rate", fontsize=8)
        ax.grid(True, alpha=0.3)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Per-substrate hero (raw): cosine_a_L21 vs leak_rate", fontsize=11)
    fig.tight_layout()
    meta = {
        "what": "Raw counterpart to hero_scatter: same predictor + DV, split per substrate.",
        "per_stratum": per_stratum,
    }
    _save_fig(fig, fig_dir, "hero_per_substrate", meta)


def cosine_vs_js_bars(df: pd.DataFrame, fig_dir: Path, pooled: dict, cis: dict) -> None:
    """Bar chart of pooled Spearman rho for each predictor + 95% CI."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    keys = PREDICTORS
    rhos = [pooled[k]["rho"] for k in keys]
    lo = [cis[k].get("rho_ci_lo", float("nan")) for k in keys]
    hi = [cis[k].get("rho_ci_hi", float("nan")) for k in keys]
    err_lo = [r - lo_ for r, lo_ in zip(rhos, lo, strict=True)]
    err_hi = [hi_ - r for r, hi_ in zip(rhos, hi, strict=True)]
    x = np.arange(len(keys))
    ax.bar(
        x,
        rhos,
        yerr=[err_lo, err_hi],
        capsize=4,
        color="#0072B2",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Pooled Spearman rho vs leak_rate")
    ax.set_title("Pooled Spearman rho per predictor (cluster-aware bootstrap 95% CI)")
    ax.grid(True, axis="y", alpha=0.3)
    meta = {
        "what": "Predictor-comparison bar chart with cluster-aware bootstrap CI.",
        "pooled": pooled,
        "ci": cis,
    }
    _save_fig(fig, fig_dir, "cosine_vs_js", meta)


def per_stratum_rho_bars(df: pd.DataFrame, fig_dir: Path, per_stratum_by_predictor: dict) -> None:
    """Grouped bar chart: per-substrate rho for each predictor."""
    substrates = sorted({s for d in per_stratum_by_predictor.values() for s in d})
    if not substrates:
        return
    fig, ax = plt.subplots(figsize=(max(6, len(substrates) * 1.5), 5))
    keys = PREDICTORS
    x = np.arange(len(substrates))
    width = 0.8 / max(1, len(keys))
    for i, k in enumerate(keys):
        vals = [per_stratum_by_predictor[k].get(s, {}).get("rho", float("nan")) for s in substrates]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width=width, label=k, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(substrates, rotation=15, ha="right", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Per-stratum Spearman rho")
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title("Per-stratum Spearman rho per predictor")
    meta = {
        "what": "Within-substrate Spearman rho per predictor (small n; interpret cautiously).",
        "per_stratum_by_predictor": per_stratum_by_predictor,
    }
    _save_fig(fig, fig_dir, "per_stratum_rho", meta)


def js_sliced_placeholder(df: pd.DataFrame, fig_dir: Path) -> None:
    """Placeholder: per-(probe-family) JS slice requires reading per-probe-family JS
    (only #444's A-family is currently sliced; the predictor JSON aggregates to
    a single value per (persona,recipe) cell). This figure emits a stub note.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.text(
        0.5,
        0.5,
        "js_sliced_by_probe_type — requires per-probe-family JS\n"
        "(not currently emitted by predictor_444). Stub.",
        ha="center",
        va="center",
        fontsize=10,
    )
    ax.axis("off")
    meta = {"what": "Placeholder figure; downstream analyzer can populate from a per-family rerun."}
    _save_fig(fig, fig_dir, "js_sliced_by_probe_type", meta)


def cell_table(df: pd.DataFrame, fig_dir: Path) -> None:
    """Flat table of all cells. Renders as an image so the figure dir carries it."""
    cols = ["substrate", "teach_persona", "bystander_persona", "leak_rate", *PREDICTORS]
    show = df[cols].copy()
    for c in [*PREDICTORS, "leak_rate"]:
        show[c] = show[c].round(4)
    fig, ax = plt.subplots(figsize=(max(8, 1.0 * len(cols)), max(2, 0.35 * (len(show) + 1))))
    ax.axis("off")
    table = ax.table(
        cellText=show.values,
        colLabels=show.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    meta = {"what": f"Flat table of {len(show)} cells with predictors + DV."}
    _save_fig(fig, fig_dir, "cell_table", meta)


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1", default=DEFAULT_PHASE1)
    ap.add_argument("--phase2", default=DEFAULT_PHASE2)
    ap.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    ap.add_argument("--out-csv", default=DEFAULT_OUT_CSV)
    ap.add_argument("--fig-dir", default=DEFAULT_FIG_DIR)
    ap.add_argument("--bootstrap-reps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="read predictor_*.smoke.json; write .smoke.json regression + stub figures.",
    )
    args = ap.parse_args()

    if args.smoke:
        args.phase1 = "eval_results/issue_494/predictor_444_canonical.smoke.json"
        args.phase2 = "eval_results/issue_494/predictor_192.smoke.json"
        args.out_json = "eval_results/issue_494/regression.smoke.json"
        args.out_csv = "eval_results/issue_494/regression_data.smoke.csv"
        args.bootstrap_reps = 50  # speed it up for smoke

    phase1 = REPO / args.phase1
    phase2 = REPO / args.phase2
    out_json = REPO / args.out_json
    out_csv = REPO / args.out_csv
    fig_dir = REPO / args.fig_dir

    df = build_pooled_df(phase1, phase2)
    logger.info(
        "Pooled DataFrame: n=%d cells, substrates=%s", len(df), sorted(df["substrate"].unique())
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info("Wrote %s", out_csv)

    # Per-stratum + pooled + CI + partial + residualized + paired-bootstrap diff
    per_stratum_by_predictor: dict = {p: per_stratum_spearman(df, p) for p in PREDICTORS}
    pooled: dict = {p: pooled_spearman(df, p) for p in PREDICTORS}
    cis: dict = {
        p: cluster_bootstrap_ci(df, p, n_reps=args.bootstrap_reps, seed=args.seed)
        for p in PREDICTORS
    }
    partials: dict = {}
    for p in ("cosine_a_L21", "cosine_b_L21", "js_on_topic", "fact_slice_js"):
        partials[f"{p}_given_prior"] = partial_spearman(
            df, x=p, y="leak_rate", control="bystander_logprob"
        )
    partials["bystander_logprob_given_cosine_a"] = partial_spearman(
        df, x="bystander_logprob", y="leak_rate", control="cosine_a_L21"
    )
    teach_resid: dict = {p: teach_residualized_spearman(df, p) for p in PREDICTORS}

    # Paired-bootstrap diffs: cosine_a vs JS and cosine_a vs prior
    diffs: dict = {}
    for a, b in [
        ("cosine_a_L21", "js_on_topic"),
        ("cosine_a_L21", "bystander_logprob"),
        ("cosine_a_L21", "cosine_b_L21"),
    ]:
        diffs[f"{a}__minus__{b}"] = paired_bootstrap_rho_diff(
            df, a, b, n_reps=args.bootstrap_reps, seed=args.seed
        )

    results = {
        "_doc": (
            "#494 Phase 3 — pooled regression across #192 (n=8) + #444 (n=18, three "
            "contrastive recipes excluding no-contrast). Per-stratum + pooled Spearman, "
            "cluster-aware bootstrap CI (within-substrate resampling), partial Spearman, "
            "teach-persona-residualized Spearman, paired-bootstrap rho-diff."
        ),
        "git_commit": _git_commit(),
        "generated_at": _now_iso(),
        "smoke": args.smoke,
        "n_cells": len(df),
        "substrates": sorted(df["substrate"].unique().tolist()),
        "predictors": PREDICTORS,
        "bootstrap_reps": args.bootstrap_reps,
        "seed": args.seed,
        "per_stratum_by_predictor": per_stratum_by_predictor,
        "pooled": pooled,
        "pooled_ci": cis,
        "partial_spearman": partials,
        "teach_persona_residualized": teach_resid,
        "paired_bootstrap_diffs": diffs,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))
    logger.info("Wrote %s", out_json)

    # ── Figures ──
    pooled_cos_a = pooled["cosine_a_L21"]
    ci_cos_a = cis["cosine_a_L21"]
    partial_cos_a = partials.get("cosine_a_L21_given_prior", {}).get("rho", float("nan"))
    headline_stats = {
        "rho": pooled_cos_a["rho"],
        "p_value": pooled_cos_a["p_value"],
        "n": pooled_cos_a["n"],
        "rho_ci_lo": ci_cos_a.get("rho_ci_lo", float("nan")),
        "rho_ci_hi": ci_cos_a.get("rho_ci_hi", float("nan")),
        "partial_rho": partial_cos_a,
    }
    hero_scatter(df, fig_dir, headline_stats)
    hero_per_substrate(df, fig_dir, per_stratum_by_predictor.get("cosine_a_L21", {}))
    cosine_vs_js_bars(df, fig_dir, pooled, cis)
    per_stratum_rho_bars(df, fig_dir, per_stratum_by_predictor)
    js_sliced_placeholder(df, fig_dir)
    cell_table(df, fig_dir)

    print(f"\n================ #494 Phase 3 regression (n={len(df)}) ================")
    print("Pooled Spearman:")
    for p in PREDICTORS:
        s = pooled[p]
        c = cis[p]
        print(
            f"  {p:24}  rho={s['rho']:+.3f}  p={s['p_value']:.3f}  "
            f"n={s['n']:3d}  ci=[{c.get('rho_ci_lo', float('nan')):+.3f}, "
            f"{c.get('rho_ci_hi', float('nan')):+.3f}]"
        )
    print("\nPartial Spearman (rank-residualized):")
    for k, v in partials.items():
        print(f"  {k:36}  rho={v['rho']:+.3f}  p={v['p_value']:.3f}  n={v['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
