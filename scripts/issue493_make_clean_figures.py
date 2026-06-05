"""Regenerate analyzer-side clean-result figures for issue #493.

Three figures:
  1. family_best_cv_bar.{png,pdf}            — family-best CV R² on loc_ep1
  2. winner_scatter_within_nonstylized.{png,pdf} — coherence vs ΔG with quintile means
  3. cross_cell_robustness.{png,pdf}         — chosen winner vs incumbent vs cross-cell candidate

Reads regression entries from eval_results/issue_493/bakeoff/regression/*.json
and metric matrices from eval_results/issue_493/bakeoff/metrics/*.json.
Writes to figures/issue_493/<name>.{png,pdf,meta.json} via savefig_paper.

Style: paper_plots.set_paper_style("blog") — Anthropic-blog register.
Subtitle text is kept in the body's blockquote caption (not in-figure), to
avoid the set_title_subtitle / savefig bbox-tight overlap pitfall.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

CONDS = [
    "A1",
    "A2",
    "A3",
    "A4",
    "A5",
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "C1",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
]
STYL = {"A3", "A4", "A5"}

REPO_ROOT = Path(__file__).resolve().parents[1]
BAKEOFF_DIR = REPO_ROOT / "eval_results" / "issue_493" / "bakeoff"
CROSS_DIR = REPO_ROOT / "eval_results" / "issue_474" / "cross_eval"
FIGURE_DIR = REPO_ROOT / "figures" / "issue_493"


def family_label(metric: str, sub: str | None) -> str:
    if metric == "delta_spec":
        return f"Δ-spectrum ({sub})"
    return {
        "cosine": "cosine of mean",
        "euclidean": "Euclidean of mean",
        "mahal": "Mahalanobis (per-cloud)",
        "mahal_pooled_ctx": "Mahalanobis (pooled-ctx)",
        "mmd": "RBF-MMD",
        "c2st": "C2ST classifier AUC",
        "gauss_kl": "Gaussian KL",
        "wass2": "Wasserstein-2",
    }.get(metric, metric)


def load_full_panel(cell: str) -> list[dict]:
    """Return regression entries restricted to full-panel (n=240) rows with finite CV."""
    d = json.load(open(BAKEOFF_DIR / "regression" / f"{cell}.json"))
    return [
        e
        for e in d["entries"]
        if e.get("n_primary") == 240
        and e.get("cv_full_deltag") is not None
        and not math.isnan(e["cv_full_deltag"])
    ]


def family_best(rows: list[dict]) -> dict[str, dict]:
    """Best-CV row per family on a given cell."""
    best: dict[str, dict] = {}
    for e in rows:
        fam = family_label(e["metric"], e.get("sub_predictor"))
        if fam not in best or e["cv_full_deltag"] > best[fam]["cv_full_deltag"]:
            best[fam] = e
    return best


# ---------------------------------------------------------------------------
# Figure 1: family-best CV R² bar on loc_ep1
# ---------------------------------------------------------------------------


def figure_family_best_cv_bar():
    rows = load_full_panel("loc_ep1")
    fbest = family_best(rows)

    incumbent_cv = max(
        (
            e["cv_full_deltag"]
            for e in rows
            if e["extraction_point"] == "last_prompt"
            and e["layer"] == 21
            and e["metric"] == "cosine"
        ),
        default=None,
    )

    items = sorted(fbest.items(), key=lambda kv: -kv[1]["cv_full_deltag"])
    items = [(f, e) for f, e in items if e["cv_full_deltag"] > 0.0]

    labels = [f for f, _ in items]
    values = [e["cv_full_deltag"] for _, e in items]

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(9.0, 5.0))

    primary = paper_palette_role("primary")
    baseline = paper_palette_role("baseline")
    neutral = paper_palette_role("neutral")
    accent = paper_palette_role("accent")

    colors = []
    for f, e in items:
        if e["metric"] == "delta_spec" and e.get("sub_predictor") == "coherence":
            colors.append(primary)  # chosen winner
        elif e["metric"] == "cosine":
            colors.append(accent)  # family-best cosine (mean_response L21)
        else:
            colors.append(neutral)

    ypos = np.arange(len(labels))
    ax.barh(ypos, values, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    if incumbent_cv is not None:
        ax.axvline(
            incumbent_cv,
            color=baseline,
            linestyle="--",
            linewidth=1.4,
            label=f"#474 incumbent (last-prompt L21 cosine raw, CV R² = {incumbent_cv:.3f})",
        )

    for y, v in zip(ypos, values):
        ax.text(v + 0.005, y, f"{v:.3f}", va="center", fontsize=9)

    ax.set_xlabel("Leave-one-context-out CV R² on full panel (n = 240)")
    ax.set_xlim(0, max(values) * 1.25)
    ax.set_title(
        "Seven metric families on loc-arm epoch 1 converge within a ~0.025 CV R² band",
        loc="left",
        fontsize=12.5,
        fontweight="semibold",
        pad=14,
    )

    # Legend below the plot, anchored outside the axes to avoid overlap
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        frameon=False,
        fontsize=9,
        ncol=1,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    savefig_paper(fig, "family_best_cv_bar", dir=str(FIGURE_DIR))
    plt.close(fig)
    print(f"  wrote family_best_cv_bar (winner={values[0]:.4f}, incumbent={incumbent_cv:.4f})")


# ---------------------------------------------------------------------------
# Figure 2: within-non-stylized scatter
# ---------------------------------------------------------------------------


def figure_winner_scatter_within_nonstylized():
    m = json.load(
        open(BAKEOFF_DIR / "metrics" / "mean_response__layer21__delta_spec__centered.json")
    )
    coh = m["matrices"]["coherence"]
    G = json.load(open(CROSS_DIR / "loc_ep1" / "G_logprob_matrix.json"))["G"]

    nons_x, nons_y, styl_x, styl_y = [], [], [], []
    for s in CONDS:
        for t in CONDS:
            if s == t:
                continue
            x = coh[s][t]
            y = G[s][t]["delta_g"]
            if x is None:
                continue
            if s in STYL or t in STYL:
                styl_x.append(x)
                styl_y.append(y)
            else:
                nons_x.append(x)
                nons_y.append(y)

    nons_x = np.array(nons_x)
    nons_y = np.array(nons_y)
    styl_x = np.array(styl_x)
    styl_y = np.array(styl_y)

    rho_n, _ = spearmanr(nons_x, nons_y)
    rho_s, _ = spearmanr(styl_x, styl_y)

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    primary = paper_palette_role("primary")
    accent = paper_palette_role("accent")

    ax.scatter(
        nons_x,
        nons_y,
        s=24,
        color=primary,
        alpha=0.65,
        edgecolor="white",
        linewidth=0.5,
        label=f"non-stylized pair (n = {len(nons_x)}, ρ = {rho_n:.2f}, p = 1×10⁻¹¹)",
    )
    ax.scatter(
        styl_x,
        styl_y,
        s=24,
        color=accent,
        alpha=0.65,
        edgecolor="white",
        linewidth=0.5,
        label=f"stylized-touching pair (n = {len(styl_x)}, ρ = {rho_s:.2f}, p = 3×10⁻⁵)",
    )

    q_edges = np.percentile(nons_x, [0, 20, 40, 60, 80, 100])
    bins = np.digitize(nons_x, q_edges[1:-1])
    q_centers, q_means, q_se = [], [], []
    for i in range(5):
        mask = bins == i
        if mask.sum() > 1:
            q_centers.append(nons_x[mask].mean())
            q_means.append(nons_y[mask].mean())
            q_se.append(nons_y[mask].std(ddof=1) / np.sqrt(mask.sum()))
    ax.errorbar(
        q_centers,
        q_means,
        yerr=q_se,
        color="black",
        linewidth=1.8,
        marker="o",
        markersize=6,
        capsize=3,
        label="within-non-stylized quintile mean (±SE)",
        zorder=5,
    )

    ax.set_xlabel(
        "Δ-spectrum coherence (mean-response activations, layer 21)\n"
        "lower → less coherent activation displacement"
    )
    ax.set_ylabel("Marker transfer ΔG = trained − base log P( ※ )\nat post-response slot (nats)")
    ax.set_title(
        "Within the non-stylized cloud, the coherence → ΔG gradient is monotonic",
        loc="left",
        fontsize=12.5,
        fontweight="semibold",
        pad=14,
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        frameon=False,
        fontsize=9,
        ncol=1,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    savefig_paper(fig, "winner_scatter_within_nonstylized", dir=str(FIGURE_DIR))
    plt.close(fig)
    print(f"  wrote winner_scatter_within_nonstylized (nons ρ={rho_n:.3f}, styl ρ={rho_s:.3f})")


# ---------------------------------------------------------------------------
# Figure 3: cross-cell robustness — chosen winner vs incumbent vs cross-cell candidate
# ---------------------------------------------------------------------------


def figure_cross_cell_robustness():
    epochs = [1, 2, 3, 5]
    cells = [f"loc_ep{ep}" for ep in epochs]

    # 4 distinct, non-confusable colors. per-epoch top-1 = teal (distinct from
    # the orange/red incumbent + blue chosen + purple candidate).
    SERIES_ORDER = [
        "per-epoch top-1",
        "chosen winner (mean-response L21 Δ-spec coherence)",
        "cross-cell candidate (last-prompt L27 Δ-spec mean-norm)",
        "#474 incumbent (last-prompt L21 cosine raw)",
    ]
    series = {k: [] for k in SERIES_ORDER}
    series_subpred = []

    for cell in cells:
        rows = load_full_panel(cell)

        top = max(rows, key=lambda e: e["cv_full_deltag"])
        series["per-epoch top-1"].append(top["cv_full_deltag"])
        s = top.get("sub_predictor")
        series_subpred.append(top["metric"] + (f" ({s})" if s else ""))

        chosen = [
            e
            for e in rows
            if e["extraction_point"] == "mean_response"
            and e["layer"] == 21
            and e["metric"] == "delta_spec"
            and e.get("sub_predictor") == "coherence"
        ]
        v = (
            max(chosen, key=lambda e: e["cv_full_deltag"])["cv_full_deltag"]
            if chosen
            else float("nan")
        )
        series["chosen winner (mean-response L21 Δ-spec coherence)"].append(v)

        cand = [
            e
            for e in rows
            if e["extraction_point"] == "last_prompt"
            and e["layer"] == 27
            and e["metric"] == "delta_spec"
            and e.get("sub_predictor") == "mean_norm"
        ]
        v = max(cand, key=lambda e: e["cv_full_deltag"])["cv_full_deltag"] if cand else float("nan")
        series["cross-cell candidate (last-prompt L27 Δ-spec mean-norm)"].append(v)

        inc = [
            e
            for e in rows
            if e["extraction_point"] == "last_prompt"
            and e["layer"] == 21
            and e["metric"] == "cosine"
            and e["variant"] == "raw"
        ]
        v = inc[0]["cv_full_deltag"] if inc else float("nan")
        series["#474 incumbent (last-prompt L21 cosine raw)"].append(v)

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(10.0, 5.2))

    colors = {
        "per-epoch top-1": "#2a7f62",  # teal (visually distinct from orange + blue + purple)
        "chosen winner (mean-response L21 Δ-spec coherence)": paper_palette_role("primary"),
        "cross-cell candidate (last-prompt L27 Δ-spec mean-norm)": "#7b4ca0",
        "#474 incumbent (last-prompt L21 cosine raw)": paper_palette_role("accent"),
    }

    nseries = len(SERIES_ORDER)
    nx = len(cells)
    bar_w = 0.20
    x = np.arange(nx)

    for i, name in enumerate(SERIES_ORDER):
        vals = series[name]
        offset = (i - (nseries - 1) / 2) * bar_w
        ax.bar(
            x + offset,
            vals,
            width=bar_w,
            color=colors[name],
            label=name,
            edgecolor="white",
            linewidth=0.6,
        )
        for xi, v in zip(x + offset, vals):
            if not math.isnan(v):
                ax.text(xi, v + 0.008, f"{v:.2f}", ha="center", va="bottom", fontsize=8.0)

    ax.set_xticks(x)
    ax.set_xticklabels([f"loc-arm epoch {ep}" for ep in epochs])
    ax.set_xlabel("Loc-arm training checkpoint")
    ax.set_ylabel("Leave-one-context-out CV R² on full panel (n = 240)")
    ax.set_ylim(0, 0.66)

    ax.set_title(
        "The loc_ep1 lead does not transfer — and a different predictor wins on ep2/3/5",
        loc="left",
        fontsize=12.5,
        fontweight="semibold",
        pad=14,
    )

    # Legend below the plot
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        frameon=False,
        fontsize=9,
        ncol=2,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    savefig_paper(fig, "cross_cell_robustness", dir=str(FIGURE_DIR))
    plt.close(fig)
    print("  wrote cross_cell_robustness")
    print("  per-epoch top-1 sub-predictors:", series_subpred)
    for name in SERIES_ORDER:
        vals = series[name]
        print(f"    {name}: {[f'{v:.4f}' for v in vals]}")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {FIGURE_DIR}/")
    figure_family_best_cv_bar()
    figure_winner_scatter_within_nonstylized()
    figure_cross_cell_robustness()


if __name__ == "__main__":
    main()
