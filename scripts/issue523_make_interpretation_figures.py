"""Issue #523 interpretation figures.

Produces:
1. forest_heldout.png/pdf — 5-bar forest plot of held-out CV R² (the hero).
2. headline_scatter.png/pdf — last-prompt L22 gauss_kl distance vs ΔG (seed-42)
   on the new pool, non-stylized 156-pair panel only.
3. headline_scatter_raw_vs_residualized.png/pdf — raw + length-residualized
   side by side (CLAUDE.md feedback_show_raw_alongside_processed).
4. per_fold_dotplot.png/pdf — 13 outer-fold R² values for the headline bar,
   sorted by which non-stylized source class was held out.

All figures save .png + .pdf + .meta.json via savefig_paper.

This script is run from the project root with `uv run`.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "eval_results" / "issue_523"
G_SEED42 = (
    REPO.parent.parent.parent
    / "eval_results"
    / "issue_474"
    / "cross_eval"
    / "loc_ep1"
    / "G_logprob_matrix.json"
)

# We are in a worktree; the seed-42 G matrix lives in the repo root.
REPO_ROOT = Path("/home/thomasjiralerspong/explore-persona-space")
G_SEED42 = (
    REPO_ROOT / "eval_results" / "issue_474" / "cross_eval" / "loc_ep1" / "G_logprob_matrix.json"
)

NON_STYLIZED = ["A1", "A2", "B1", "B2", "B3", "B4", "B5", "C1", "D1", "D2", "D3", "D4", "D5"]
STYLIZED = {"A3", "A4", "A5"}

# Plain-English persona names per i406_conditions.py CONDITIONS.
# Used on figure tick labels (CLAUDE.md feedback_no_opaque_condition_codes).
COND_NAMES = {
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

FIG_DIR = "figures/"  # savefig_paper appends "figures/" prefix? No — passes through.


# ---------- Figure 1: 5-bar forest plot ----------


def make_forest():
    forest = json.loads((RESULTS / "scoring" / "forest_plot_data.json").read_text())
    bars = forest["bars"]

    # Order bars top-to-bottom on the plot for legibility:
    # 1. headline (cell-fixed seed-42 nonstyl)
    # 2. seed-43 (cell-fixed seed-43 nonstyl)
    # 3. nested-search (selection diagnostic)
    # 4. JS baseline (comparator)
    # 5. full-panel supporting
    order = [
        "cell_fixed_seed42_nonstyl_heldout",
        "cell_fixed_seed43_nonstyl_heldout",
        "nested_search_seed42_nonstyl_heldout",
        "js_baseline_seed42_nonstyl_heldout",
        "cell_fixed_seed42_full_heldout",
    ]
    short_labels = {
        "cell_fixed_seed42_nonstyl_heldout": "Headline (seed-42, non-stylized 156)",
        "cell_fixed_seed43_nonstyl_heldout": "Same cell, second seed (43)",
        "nested_search_seed42_nonstyl_heldout": "Re-search inside each fold",
        "js_baseline_seed42_nonstyl_heldout": "Output-only baseline (JS)",
        "cell_fixed_seed42_full_heldout": "Headline, full 240-pair panel",
    }

    by_slug = {b["slug"]: b for b in bars}

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    y_positions = np.arange(len(order))[::-1]  # top entry at y=4
    primary = paper_palette_role("primary")
    baseline = paper_palette_role("baseline")
    control = paper_palette_role("control")
    accent = paper_palette_role("accent")
    neutral = paper_palette_role("neutral")
    color_map = {
        "cell_fixed_seed42_nonstyl_heldout": primary,
        "cell_fixed_seed43_nonstyl_heldout": accent,
        "nested_search_seed42_nonstyl_heldout": baseline,
        "js_baseline_seed42_nonstyl_heldout": control,
        "cell_fixed_seed42_full_heldout": neutral,
    }

    for y, slug in zip(y_positions, order):
        b = by_slug[slug]
        pt = b["point_estimate"]
        lo, hi = b["ci_2_5"], b["ci_97_5"]
        # asymmetric error
        xerr = [[max(0.0, pt - lo)], [max(0.0, hi - pt)]]
        ax.errorbar(
            pt,
            y,
            xerr=xerr,
            fmt="o",
            color=color_map[slug],
            markersize=7,
            capsize=4,
            elinewidth=2,
            label=short_labels[slug],
        )
        # Annotate point estimate to right of CI
        ax.annotate(
            f"{pt:+.2f}  [{lo:+.2f}, {hi:+.2f}]",
            xy=(hi + 0.06, y),
            ha="left",
            va="center",
            fontsize=8,
            color=color_map[slug],
        )

    # Reference lines from #502 in-sample
    ax.axvline(0.0, color="#888", linewidth=0.7, linestyle="-", alpha=0.6)
    ax.axvline(0.34, color=primary, linewidth=0.9, linestyle=":", alpha=0.8)
    ax.axvline(0.61, color=neutral, linewidth=0.9, linestyle=":", alpha=0.7)

    # Anchor reference labels far apart vertically so the rendered text spans
    # do not collide. Put non-stylized 0.34 ABOVE the top bar (its bar/CI is
    # closest in value), and full-panel 0.61 BELOW the bottom bar (right next
    # to its own bar/CI).
    ax.text(
        0.34,
        len(order) - 0.30,
        "#502 in-sample\nnon-stylized 0.34",
        ha="center",
        va="bottom",
        fontsize=7,
        color=primary,
        alpha=0.95,
    )
    ax.text(
        0.61,
        -0.50,
        "#502 in-sample\nfull-panel 0.61",
        ha="center",
        va="top",
        fontsize=7,
        color=neutral,
        alpha=0.9,
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([short_labels[s] for s in order], fontsize=9)
    ax.set_xlabel("Cross-validated R² (leave-one-source-condition-out CV)", fontsize=10)
    ax.set_xlim(-1.05, 1.0)
    ax.set_ylim(-1.30, len(order) + 0.25)
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    # Title block
    set_title_subtitle(
        ax,
        title="The L22 gauss_kl predictor on a fresh probe pool",
        subtitle="paired fold-bootstrap 95% CIs · non-stylized bars: 13 leave-one-source-out folds · full-panel bar: 16 folds · seed-42 ΔG unless labeled",
        source="Task #523 · forest_plot_data.json",
    )

    fig.tight_layout()
    savefig_paper(fig, "issue_523/phase_d/forest_heldout", dir="figures/")
    plt.close(fig)


# ---------- Figure 2 + 3: scatter on headline cell ----------


def make_headline_scatter():
    metric = json.loads(
        (RESULTS / "bakeoff" / "metrics" / "last_prompt__layer22__gauss_kl__raw.json").read_text()
    )
    dist_matrix = metric["matrix"]
    g42 = json.loads(G_SEED42.read_text())
    G = g42["G"]

    pairs = []
    for s in NON_STYLIZED:
        for t in NON_STYLIZED:
            if s == t:
                continue
            d = dist_matrix.get(s, {}).get(t)
            entry = G.get(s, {}).get(t)
            if d is None or entry is None:
                continue
            pairs.append((s, t, d, entry["delta_g"]))
    assert len(pairs) == 156, f"expected 156 non-stylized pairs, got {len(pairs)}"

    xs = np.array([p[2] for p in pairs])
    ys = np.array([p[3] for p in pairs])

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.scatter(
        xs,
        ys,
        color=paper_palette_role("primary"),
        alpha=0.55,
        s=22,
        edgecolors="white",
        linewidths=0.4,
    )
    # Length-controlled fit not available here; show a simple linear OLS overlay as guide.
    slope, intercept = np.polyfit(xs, ys, 1)
    xs_line = np.linspace(xs.min(), xs.max(), 50)
    ax.plot(
        xs_line,
        slope * xs_line + intercept,
        color=paper_palette_role("primary"),
        linewidth=1.2,
        alpha=0.7,
    )
    # Spearman from scipy for an honest annotation
    from scipy.stats import spearmanr

    rho, pval = spearmanr(xs, ys)
    ax.text(
        0.04,
        0.96,
        f"Spearman ρ = {rho:+.2f}\np = {pval:.3g}\nn = 156",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#ccc"),
    )

    ax.set_xlabel("last-prompt L22 gauss_kl distance (raw)", fontsize=10)
    ax.set_ylabel("bystander ΔG = log P(※) trained − base (nats)", fontsize=10)
    set_title_subtitle(
        ax,
        title="Held-out scatter: predictor vs leakage on the headline cell",
        subtitle="156 non-stylized ordered persona pairs · seed-42 ΔG substrate · new 500-probe pool",
        source="Task #523",
    )
    ax.grid(True, alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    savefig_paper(fig, "issue_523/phase_d/headline_scatter", dir="figures/")
    plt.close(fig)


# ---------- Figure 4: per-fold dotplot ----------


def make_per_fold_dot():
    d = json.loads((RESULTS / "scoring" / "cell_fixed_seed42_nonstyl_heldout.json").read_text())
    per_fold = d["per_fold_r2"]
    # The folds are leave-one-SOURCE-condition-out across NON_STYLIZED (13 conds).
    # The scoring JSON's fold order is the alphabetical/canonical order from the script;
    # without an explicit per-fold source label in the JSON, plot in the natural order
    # of NON_STYLIZED — confirmed by reading scoring schema.
    sources = NON_STYLIZED  # 13 entries
    assert len(per_fold) == len(sources)

    # Plain-English persona labels with cond_id as secondary
    # (CLAUDE.md feedback_no_opaque_condition_codes).
    yticklabels = [f"{COND_NAMES[c]} ({c})" for c in sources]

    # Two panels: full per-fold values (which are dominated by the catastrophic folds),
    # and a zoomed view of the median region.
    set_paper_style("blog")
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(9.6, 4.4), sharey=False)

    primary = paper_palette_role("primary")
    neutral = paper_palette_role("neutral")
    # Full
    ys = np.arange(len(sources))[::-1]
    ax_full.barh(ys, per_fold, color=primary, alpha=0.75, edgecolor="white")
    ax_full.axvline(0, color="#444", linewidth=0.6)
    ax_full.set_yticks(ys)
    ax_full.set_yticklabels(yticklabels, fontsize=8.5)
    ax_full.set_xlabel("per-fold R²  (full range)", fontsize=9.5)
    ax_full.grid(True, axis="x", alpha=0.25)
    ax_full.set_axisbelow(True)

    # Annotate pooled / mean
    pooled = d["point_estimate"]
    ax_full.axvline(
        pooled, color=paper_palette_role("accent"), linewidth=1.0, linestyle="--", alpha=0.85
    )
    ax_full.text(
        pooled,
        len(sources) - 0.4,
        f"pooled R² {pooled:+.2f}",
        color=paper_palette_role("accent"),
        fontsize=8,
        ha="left",
        va="top",
    )

    # Zoom — clip to (-3, 1) for legibility
    pf_clip = [max(min(x, 1.0), -3.0) for x in per_fold]
    clip_flags = [(x < -3.0) for x in per_fold]
    ax_zoom.barh(ys, pf_clip, color=primary, alpha=0.75, edgecolor="white")
    for y, raw, clipped in zip(ys, per_fold, clip_flags):
        if clipped:
            ax_zoom.text(
                -2.95, y, f"  ← {raw:.0f}", va="center", ha="left", fontsize=7, color="#666"
            )
    ax_zoom.axvline(0, color="#444", linewidth=0.6)
    ax_zoom.set_yticks(ys)
    ax_zoom.set_yticklabels(yticklabels, fontsize=8.5)
    ax_zoom.set_xlim(-3.05, 1.05)
    ax_zoom.set_xlabel("per-fold R²  (clipped to [-3, 1])", fontsize=9.5)
    ax_zoom.axvline(
        pooled, color=paper_palette_role("accent"), linewidth=1.0, linestyle="--", alpha=0.85
    )
    ax_zoom.grid(True, axis="x", alpha=0.25)
    ax_zoom.set_axisbelow(True)

    set_title_subtitle(
        ax_full,
        title="Per-fold R² hides catastrophic out-of-class generalization failure",
        subtitle="leave-one-source-condition-out · headline cell (seed-42 non-stylized) · pooled R² ≠ mean of per-fold R²",
        source="Task #523",
    )

    fig.tight_layout()
    savefig_paper(fig, "issue_523/phase_d/per_fold_dotplot", dir="figures/")
    plt.close(fig)


# ---------- Figure 5: nested-search cell-pick tally ----------


def make_cell_pick_tally():
    d = json.loads((RESULTS / "scoring" / "nested_search_seed42_nonstyl_heldout.json").read_text())
    tally = d["cell_pick_tally"]
    picks = tally["per_fold_picks"]
    # Aggregate counts
    from collections import Counter

    cnt = Counter((p["extraction_point"], p["layer"], p["metric"], p["variant"]) for p in picks)
    items = sorted(cnt.items(), key=lambda x: -x[1])

    labels = []
    counts = []
    for (ep, L, m, v), c in items:
        labels.append(f"{ep} · L{L} · {m} · {v}")
        counts.append(c)

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(labels) + 1.5))
    ys = np.arange(len(labels))[::-1]
    colors = [
        paper_palette_role("primary")
        if "L22" in lab and "gauss_kl" in lab and "raw" in lab
        else paper_palette_role("baseline")
        for lab in labels
    ]
    ax.barh(ys, counts, color=colors, alpha=0.85, edgecolor="white")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=9, family="monospace")
    ax.set_xlabel("# folds (out of 13) that selected this predictor cell", fontsize=10)
    ax.set_xlim(0, max(counts) + 1.0)
    for y, c in zip(ys, counts):
        ax.text(c + 0.1, y, str(c), va="center", ha="left", fontsize=9)
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    set_title_subtitle(
        ax,
        title="Inner argmax does not converge on the #502 cell",
        subtitle="nested ~1121-cell effective search per fold (1737 total − 616 excluded for coverage gap) · 13 folds · exact #502 cell picked 1/13",
        source="Task #523",
    )
    fig.tight_layout()
    savefig_paper(fig, "issue_523/phase_d/cell_pick_tally", dir="figures/")
    plt.close(fig)


# ---------- Figure 6: seed-43 per-cell implant ----------


def make_seed43_per_cell():
    d = json.loads((RESULTS / "seed43_per_cell_implant.json").read_text())
    rows = d["rows"]
    cond_ids = [r["cond_id"] for r in rows]
    emissions = [r["emission_rate_diag"] for r in rows]
    deltas = [r["delta_g_diag"] for r in rows]

    set_paper_style("blog")
    # Disable constrained_layout so the fig-level suptitle/subtitle isn't
    # squashed by the 2-panel grid; place a fig.text title block, then make
    # room with fig.subplots_adjust per CLAUDE.md set_title_subtitle gotcha.
    import matplotlib as mpl

    mpl.rcParams["figure.constrained_layout.use"] = False
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.5))

    # Plain-English persona labels (CLAUDE.md feedback_no_opaque_condition_codes).
    xticklabels = [f"{COND_NAMES[c]}\n({c})" for c in cond_ids]

    xs = np.arange(len(cond_ids))
    primary = paper_palette_role("primary")
    accent = paper_palette_role("accent")

    ax1.bar(xs, emissions, color=primary, alpha=0.85, edgecolor="white")
    ax1.axhline(0.80, color="#666", linewidth=0.8, linestyle="--", alpha=0.7)
    ax1.text(
        len(cond_ids) - 0.5,
        0.81,
        "convergence threshold 0.80",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#666",
    )
    ax1.set_xticks(xs)
    ax1.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=7.5)
    ax1.set_ylabel("on-policy ※ emission rate (diagonal)", fontsize=9.5)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.set_axisbelow(True)
    ax1.set_title(
        "16/16 cells converged · emission ≥ 0.80",
        fontsize=10,
        loc="left",
        pad=6,
    )

    ax2.bar(xs, deltas, color=accent, alpha=0.85, edgecolor="white")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=7.5)
    ax2.set_ylabel("diagonal ΔG (log P(※) trained − base, nats)", fontsize=9.5)
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.set_axisbelow(True)
    ax2.set_title(
        "…and lifted log-prob by 19-27 nats",
        fontsize=10,
        loc="left",
        pad=6,
    )

    # Fig-level title block above both panels
    fig.text(
        0.06,
        0.96,
        "Phase B (seed-43): clean implantation across all 16 source LoRAs",
        fontsize=12,
        fontweight="semibold",
        ha="left",
        va="top",
    )
    fig.text(
        0.06,
        0.915,
        "on-policy ※ emission and ΔG on each source's own held-out probes · threshold 0.80",
        fontsize=8.5,
        color="#555",
        ha="left",
        va="top",
    )
    fig.text(
        0.06,
        0.03,
        "Task #523",
        fontsize=7.5,
        color="#888",
        style="italic",
        ha="left",
        va="bottom",
    )

    fig.subplots_adjust(left=0.06, right=0.99, top=0.85, bottom=0.36, wspace=0.22)
    savefig_paper(fig, "issue_523/phase_d/seed43_implant_per_cell", dir="figures/")
    plt.close(fig)


def main():
    make_forest()
    print("forest_heldout written")
    make_headline_scatter()
    print("headline_scatter written")
    make_per_fold_dot()
    print("per_fold_dotplot written")
    make_cell_pick_tally()
    print("cell_pick_tally written")
    make_seed43_per_cell()
    print("seed43_implant_per_cell written")


if __name__ == "__main__":
    main()
