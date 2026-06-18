"""Issue #522 clean-result figure generation.

Three figures, all blog-style:
  hero.png             — scatter: full-response JS-similarity vs #474 ΔG,
                          loc_ep=1 full panel (240 ordered pairs).
  by_panel_epoch.png   — bar grid: ρ_ΔG and CV_R²_ΔG across panel × epoch.
  metric_compare.png   — bar: CV_R² at loc_ep=1, comparing this run's
                          full-response JS vs Phase 1's gauss_kl, mmd,
                          wass2, and cosine (all layer L22, N=500).

Run from repo root or worktree:
    uv run python scripts/issue522_make_figures.py

Outputs under figures/issue_522/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

# ───────────────────────── paths ─────────────────────────

WT = Path(__file__).resolve().parent.parent
EVAL = WT / "eval_results" / "issue_522"
G_EP1 = WT / "eval_results" / "issue_474" / "cross_eval" / "loc_ep1" / "G_logprob_matrix.json"

# Stylized cond_ids — A3 Pirate captain, A4 Stand-up comedian, A5 Villainous mastermind
STY_CIDS = frozenset({"A3", "A4", "A5"})

# Plain-English persona names (i406_conditions.py)
PERSONA_NAMES = {
    "A1": "Helpful assistant",
    "A2": "Software engineer",
    "A3": "Pirate captain",
    "A4": "Stand-up comedian",
    "A5": "Villainous mastermind",
    "B1": "Bare question",
    "B2": "Imperative tell-me",
    "B3": "Polite request",
    "B4": "Formal request",
    "B5": "Socratic hypothetical",
    "C1": "Standard Qwen template",
    "D1": "Formal register rewrite",
    "D2": "Casual register rewrite",
    "D3": "Indirect framing rewrite",
    "D4": "Declarative form rewrite",
    "D5": "Enumerated framing rewrite",
}


# ───────────────────────── helpers ─────────────────────────


def _load_js_matrix() -> tuple[dict[str, dict[str, float]], list[str]]:
    data = json.loads((EVAL / "js_matrix.json").read_text())
    return data["JS"], list(data["cond_ids"])


def _load_g_ep1() -> dict[str, dict[str, dict[str, float]]]:
    return json.loads(G_EP1.read_text())["G"]


def _load_regression() -> list[dict[str, Any]]:
    return json.loads((EVAL / "js_regression.json").read_text())["rows"]


def _pairs(cond_ids: list[str], *, nonstylized: bool) -> list[tuple[str, str]]:
    out = []
    for a in cond_ids:
        for b in cond_ids:
            if a == b:
                continue
            if nonstylized and (a in STY_CIDS or b in STY_CIDS):
                continue
            out.append((a, b))
    return out


# ───────────────────────── figure 1: hero scatter ─────────────────────────


def hero_scatter() -> None:
    """Scatter: full-response JS (loc_ep1) vs #474 ΔG, full 240-pair panel.

    Colour: stylized-touching pairs vs nonstylized-only pairs.
    Annotation: ρ + CV_R² from the regression JSON (panel-CI 95%).
    """
    set_paper_style("blog")
    JS, cond_ids = _load_js_matrix()
    G = _load_g_ep1()
    rows = _load_regression()

    # 240 ordered pairs (a != b)
    pairs = _pairs(cond_ids, nonstylized=False)
    xs = np.array([float(JS[a][b]) for a, b in pairs], dtype=np.float64)
    ys = np.array([float(G[a][b]["delta_g"]) for a, b in pairs], dtype=np.float64)
    is_sty = np.array([a in STY_CIDS or b in STY_CIDS for a, b in pairs])

    # Regression row at full / ep1
    r_full_ep1 = next(r for r in rows if r["panel"] == "full" and r["epoch"] == 1)
    rho = r_full_ep1["point_estimate"]["rho"]
    p = r_full_ep1["point_estimate"]["p"]
    rho_lo = r_full_ep1["panel_ci"]["rho"]["lo"]
    rho_hi = r_full_ep1["panel_ci"]["rho"]["hi"]
    cv = r_full_ep1["point_estimate"]["cv_r2"]
    cv_lo = r_full_ep1["panel_ci"]["cv_r2"]["lo"]
    cv_hi = r_full_ep1["panel_ci"]["cv_r2"]["hi"]
    n = r_full_ep1["n_pairs"]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))

    c_ns = paper_palette_role("primary")
    c_sty = paper_palette_role("accent")
    ax.scatter(
        xs[~is_sty],
        ys[~is_sty],
        s=22,
        alpha=0.65,
        color=c_ns,
        label=f"nonstylized pairs (n={(~is_sty).sum()})",
        edgecolors="white",
        linewidths=0.5,
    )
    ax.scatter(
        xs[is_sty],
        ys[is_sty],
        s=22,
        alpha=0.85,
        color=c_sty,
        label=f"pairs touching a stylized persona (n={is_sty.sum()})",
        edgecolors="white",
        linewidths=0.5,
    )

    ax.set_xlabel("Full-response JS divergence (base-model, bits)")
    ax.set_ylabel("Marker-leakage transfer ΔG (nats; from #474)")

    # Anthropic-blog title block
    subtitle = (
        f"Spearman ρ = {rho:.2f} [{rho_lo:.2f}, {rho_hi:.2f}] · "
        f"LOCO CV R² = {cv:.2f} [{cv_lo:.2f}, {cv_hi:.2f}] · "
        f"n = {n}, p = {p:.1e}"
    )
    set_title_subtitle(
        ax,
        "Most of the JS predictor lives in the stylized-vs-rest gap",
        subtitle,
        source="full 240-pair panel · 16×16 personas · loc-arm epoch 1",
    )

    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)

    savefig_paper(fig, "issue_522/hero")
    plt.close(fig)


# ───────────────────────── figure 2: by panel × epoch grid ─────────────────────────


def by_panel_epoch_bars() -> None:
    """Bar chart: ρ_ΔG and CV_R²_ΔG for panel ∈ {full, nonstylized}, ep ∈ {1,2,3,5}.

    Two side-by-side subplots; panel-CI error bars; coloured by panel.
    """
    set_paper_style("blog")
    import matplotlib as mpl

    mpl.rcParams["figure.constrained_layout.use"] = False
    rows = _load_regression()

    epochs = [1, 2, 3, 5]
    panels = ["full", "nonstylized"]
    colors = {"full": paper_palette_role("primary"), "nonstylized": paper_palette_role("baseline")}
    n_pairs = {"full": 240, "nonstylized": 156}

    fig, (ax_rho, ax_cv) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    x = np.arange(len(epochs))
    w = 0.36
    for k, panel in enumerate(panels):
        rhos, cvs, rho_lo, rho_hi, cv_lo, cv_hi = [], [], [], [], [], []
        for ep in epochs:
            r = next(rr for rr in rows if rr["panel"] == panel and rr["epoch"] == ep)
            rhos.append(r["point_estimate"]["rho"])
            cvs.append(r["point_estimate"]["cv_r2"])
            rho_lo.append(r["panel_ci"]["rho"]["lo"])
            rho_hi.append(r["panel_ci"]["rho"]["hi"])
            cv_lo.append(r["panel_ci"]["cv_r2"]["lo"])
            cv_hi.append(r["panel_ci"]["cv_r2"]["hi"])
        rho_err = np.array([np.array(rhos) - np.array(rho_lo), np.array(rho_hi) - np.array(rhos)])
        cv_err = np.array([np.array(cvs) - np.array(cv_lo), np.array(cv_hi) - np.array(cvs)])

        offset = (k - 0.5) * w
        ax_rho.bar(
            x + offset,
            rhos,
            width=w,
            color=colors[panel],
            edgecolor="white",
            linewidth=0.7,
            label=f"{panel} panel (n={n_pairs[panel]})",
        )
        ax_rho.errorbar(
            x + offset,
            rhos,
            yerr=rho_err,
            fmt="none",
            ecolor="#2a2a2a",
            elinewidth=0.9,
            capsize=2.5,
        )

        ax_cv.bar(
            x + offset,
            cvs,
            width=w,
            color=colors[panel],
            edgecolor="white",
            linewidth=0.7,
            label=f"{panel} panel (n={n_pairs[panel]})",
        )
        ax_cv.errorbar(
            x + offset,
            cvs,
            yerr=cv_err,
            fmt="none",
            ecolor="#2a2a2a",
            elinewidth=0.9,
            capsize=2.5,
        )

    for ax in (ax_rho, ax_cv):
        ax.axhline(0, color="#666666", linewidth=0.7, linestyle="--", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"epoch {ep}" for ep in epochs])
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_xlabel("Source-persona training amount")

    ax_rho.set_ylabel("Spearman ρ vs ΔG (length-partialled)")
    ax_cv.set_ylabel("LOCO CV R² vs ΔG")
    ax_rho.legend(loc="upper right", frameon=False, fontsize=8.5)
    ax_cv.legend(loc="upper right", frameon=False, fontsize=8.5)

    fig.suptitle(
        "Drop the three stylized personas and out-of-sample CV R² loses "
        "any demonstrated skill at every training amount",
        x=0.04,
        y=1.00,
        ha="left",
        fontsize=11.5,
        fontweight="semibold",
    )
    fig.text(
        0.04,
        0.95,
        "Panel-row bootstrap 95% CIs · n_boot = 2000 · seed = 42",
        ha="left",
        fontsize=9.5,
        color="#454545",
    )
    import matplotlib as mpl

    mpl.rcParams["figure.constrained_layout.use"] = False
    fig.set_constrained_layout(False)
    fig.subplots_adjust(top=0.85, bottom=0.13, left=0.07, right=0.98, wspace=0.22)

    savefig_paper(fig, "issue_522/by_panel_epoch")
    plt.close(fig)


# ───────────────────────── figure 3: predictor comparison ─────────────────────────


def metric_comparison() -> None:
    """Bar chart: CV R² at loc_ep=1, full panel, layer L22 (where applicable).

    Includes:
      - Full-response JS (this run, base-model, N=200 probes × R=8)
      - gauss_kl (Phase 1 L22 cell, N=500, R=10)
      - mmd       (Phase 1 L22 cell, N=500, R=10)
      - wass2     (Phase 1 L22 cell, N=500, R=10)
      - cosine    (Phase 1 L22 cell, N=500, R=10)
    """
    set_paper_style("blog")
    import matplotlib as mpl

    mpl.rcParams["figure.constrained_layout.use"] = False
    rows_js = _load_regression()
    sweep = json.loads((EVAL / "probe_count_sweep_results.json").read_text())

    # JS at full / ep1
    r_js = next(r for r in rows_js if r["panel"] == "full" and r["epoch"] == 1)
    js_cv = r_js["point_estimate"]["cv_r2"]
    js_cv_lo = r_js["panel_ci"]["cv_r2"]["lo"]
    js_cv_hi = r_js["panel_ci"]["cv_r2"]["hi"]

    metric_results: list[tuple[str, float]] = []
    # Pull all subset_idx rows at L22 / N=500 / ep=1 / loc / each metric, take mean
    for metric in ["gauss_kl", "mmd", "wass2", "cosine"]:
        cell = f"last_prompt__L22__{metric}__raw"
        cvs = [
            r["cv_r2"]
            for r in sweep["rows"]
            if r["cell_id"] == cell and r["epoch"] == 1 and r["N"] == 500 and r["arm"] == "loc"
        ]
        if not cvs:
            raise RuntimeError(f"No N=500 rows for {cell}")
        metric_results.append((metric, float(np.mean(cvs))))

    labels = [
        "Full-response JS\n(output-distribution,\nbase model)",
        "Gaussian KL\n(activations L22)",
        "MMD\n(activations L22)",
        "Wasserstein-2\n(activations L22)",
        "Cosine\n(activations L22)",
    ]
    cvs = [js_cv] + [m[1] for m in metric_results]
    colors = [
        paper_palette_role("accent"),  # JS — this run, called out
        paper_palette_role("primary"),  # gauss_kl — Phase 1 headline
        paper_palette_role("baseline"),
        paper_palette_role("baseline"),
        paper_palette_role("control"),
    ]

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    bars = ax.bar(
        np.arange(len(labels)),
        cvs,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
    )
    # CI on JS only (the activation predictors don't carry panel-row CIs in the
    # sweep JSON; their subset-σ across the 10 replicates is shown as a small
    # error bar)
    ax.errorbar(
        [0],
        [js_cv],
        yerr=[[js_cv - js_cv_lo], [js_cv_hi - js_cv]],
        fmt="none",
        ecolor="#2a2a2a",
        elinewidth=1.0,
        capsize=3,
    )
    # Subset-σ error bars for the four activation metrics
    for i, metric in enumerate(["gauss_kl", "mmd", "wass2", "cosine"]):
        cell = f"last_prompt__L22__{metric}__raw"
        cvs_subsets = [
            r["cv_r2"]
            for r in sweep["rows"]
            if r["cell_id"] == cell and r["epoch"] == 1 and r["N"] == 500 and r["arm"] == "loc"
        ]
        if len(cvs_subsets) >= 2:
            sd = float(np.std(cvs_subsets, ddof=1))
            ax.errorbar(
                [i + 1],
                [float(np.mean(cvs_subsets))],
                yerr=[[sd], [sd]],
                fmt="none",
                ecolor="#2a2a2a",
                elinewidth=1.0,
                capsize=3,
            )

    # Value annotations
    for i, c in enumerate(cvs):
        ax.text(i, c + 0.025, f"{c:.2f}", ha="center", fontsize=9.5, color="#2a2a2a")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("LOCO CV R² vs ΔG (loc-arm, epoch 1)")
    ax.set_ylim(0, max(cvs) * 1.18)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)

    set_title_subtitle(
        ax,
        "On the full 240-pair panel, activation-geometry predictors beat "
        "full-response JS by 2.4× CV R²",
        "Out-of-sample CV R² predicting marker-leakage transfer ΔG · full panel · loc-arm epoch 1",
        source="JS: base-model on-policy pass · activation metrics: cached residual-stream + cloud-aware ridge",
    )
    import matplotlib as mpl

    mpl.rcParams["figure.constrained_layout.use"] = False
    fig.set_constrained_layout(False)
    fig.subplots_adjust(top=0.85, bottom=0.18, left=0.10, right=0.97)

    savefig_paper(fig, "issue_522/metric_compare")
    plt.close(fig)


# ───────────────────────── main ─────────────────────────


def main() -> int:
    hero_scatter()
    by_panel_epoch_bars()
    metric_comparison()
    print("Wrote figures/issue_522/{hero, by_panel_epoch, metric_compare}.{png,pdf,meta.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
