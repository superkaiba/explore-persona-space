"""Analyzer round-2 figure revisions for #591 (VM-side, zero GPU).

Addresses the round-1 interpretation-critique figure findings:
- e2 hero + cross-matrix: reader-facing adapter/persona labels (no
  hyphenated slug-like compounds such as "software-engineer").
- e1 leak map: reader-facing source/bystander labels ("default Qwen
  (no persona)" instead of "qwen default", "AI assistant" instead of
  "ai assistant") and behavior titles without issue-number tags.
- e1 factor forest: the anti-conservative panel-constant self-delta
  coefficients are REMOVED from the plot (the registered between-source
  permutation is the binding implant-strength test); an in-figure note
  says so.
- e1 panel self-delta scatter: staggered annotation offsets so point
  labels near the floor no longer collide.

All figures are regenerated from the saved e1/e2 JSONs - no stats are
re-run. Output file names are unchanged so the interpretation body only
needs a new commit SHA.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[2]
E2 = REPO / "eval_results/issue_591/e2/extended_panel_results.json"
E1_CT = REPO / "eval_results/issue_591/e1/cell_table.json"
E1_FA = REPO / "eval_results/issue_591/e1/factor_analysis.json"
FIGDIR = REPO / "figures" / "issue_591"

BEHAVIORS = ["sycophancy", "refusal", "em"]
SOURCES = [
    "assistant",
    "comedian",
    "kindergarten_teacher",
    "qwen_default",
    "software_engineer",
    "villain",
]

# Reader-facing labels (plain English, no slug-style hyphenated compounds)
PERSONA_LABELS = {
    "supervillain": "supervillain",
    "evil_mastermind": "evil mastermind",
    "dark_overlord": "dark overlord",
    "criminal_mastermind": "criminal mastermind",
    "standup_comic": "stand-up comic",
    "improv_comedian": "improv comedian",
    "late_night_host": "late-night host",
    "daycare_teacher": "daycare teacher",
    "preschool_teacher": "preschool teacher",
    "nursery_school_teacher": "nursery-school teacher",
    "elementary_school_teacher": "elementary-school teacher",
    "web_developer": "web developer",
    "fullstack_programmer": "full-stack programmer",
    "virtual_assistant": "virtual assistant",
    "digital_helper": "digital helper",
}
ADAPTER_LABELS = {
    "villain": "villain adapter",
    "comedian": "comedian adapter",
    "kindergarten_teacher": "kindergarten teacher adapter",
    "software_engineer": "software engineer adapter (known-leaking)",
}
SOURCE_LABELS = {
    "assistant": "assistant",
    "comedian": "comedian",
    "kindergarten_teacher": "kindergarten teacher",
    "qwen_default": "default Qwen (no persona)",
    "software_engineer": "software engineer",
    "villain": "villain",
}
BYSTANDER_LABELS = {
    "ai": "AI",
    "ai_assistant": "AI assistant",
    "qwen_default": "default Qwen (no persona)",
    "zelthari_scholar": "Zelthari scholar",
}
BEHAVIOR_LABELS = {
    "sycophancy": "sycophancy",
    "refusal": "refusal",
    "em": "emergent misalignment",
}
FACTOR_LABELS = {
    "cos_to_source": "cosine to source",
    "bystander_base_rate": "bystander base propensity",
    "neg_member": "training-negative membership",
}


def _bystander_label(b: str) -> str:
    return BYSTANDER_LABELS.get(b, b.replace("_", " "))


def fig_hero(cells, frozen_cells):
    set_paper_style("blog")
    srcs = ["villain", "comedian", "kindergarten_teacher", "software_engineer"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4), squeeze=False)
    c_twin = paper_palette_role("primary")
    c_pc = paper_palette_role("accent")
    for ax, src in zip(axes[0], srcs, strict=True):
        fr = [c for c in frozen_cells if c["source"] == src]
        ax.scatter(
            [c["cos_to_source"] for c in fr],
            [c["delta"] for c in fr],
            s=14,
            color="lightgrey",
            zorder=1,
        )
        for c in cells:
            if c["adapter_source"] != src or c.get("cos_to_adapter_source") is None:
                continue
            role = c["role_assigned"]
            color = c_pc if role.startswith("positive_control") else c_twin
            ax.errorbar(
                c["cos_to_adapter_source"],
                c["delta_raw"],
                yerr=[
                    [max(0.0, c["delta_raw"] - c["ci95_low"])],
                    [max(0.0, c["ci95_high"] - c["delta_raw"])],
                ],
                fmt="o",
                ms=5,
                color=color,
                zorder=3,
            )
        ax.axvspan(0.95, 0.97, alpha=0.10, color="orange", zorder=0)
        ax.axvspan(0.97, 1.005, alpha=0.10, color="green", zorder=0)
        ax.axhline(0.10, color="grey", ls="--", lw=0.8)
        ax.set_title(ADAPTER_LABELS[src], fontsize=11)
        ax.set_xlabel("cosine to adapter source (layer 20)")
    axes[0][0].set_ylabel("agreement-rate delta (trained - base)")
    handles = [
        Line2D([], [], marker="o", ls="", color="lightgrey", label="frozen 23-bystander panel"),
        Line2D([], [], marker="o", ls="", color=c_twin, label="new synthesized persona (95% CI)"),
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            color=c_pc,
            label="positive-control twins (software engineer / assistant)",
        ),
        Line2D([], [], ls="--", color="grey", label="leak threshold (delta = +0.10)"),
    ]
    axes[0][0].legend(handles=handles, fontsize=7.5, loc="upper left")
    savefig_paper(fig, "e2_delta_vs_cosine_hero", dir=FIGDIR)
    plt.close(fig)


def fig_cross_matrix(cells):
    set_paper_style("blog")
    adapters = ["villain", "comedian", "kindergarten_teacher", "software_engineer"]
    col_order = [
        "supervillain",
        "evil_mastermind",
        "dark_overlord",
        "criminal_mastermind",
        "standup_comic",
        "improv_comedian",
        "late_night_host",
        "daycare_teacher",
        "preschool_teacher",
        "nursery_school_teacher",
        "elementary_school_teacher",
        "web_developer",
        "fullstack_programmer",
        "virtual_assistant",
        "digital_helper",
    ]
    lut = {(c["adapter_source"], c["new_persona"]): c for c in cells}
    mat = np.full((len(adapters), len(col_order)), np.nan)
    for i, a in enumerate(adapters):
        for j, p in enumerate(col_order):
            c = lut.get((a, p))
            if c is not None:
                mat[i, j] = c["delta_raw"]
    fig, ax = plt.subplots(figsize=(12.5, 3.6))
    vmax = np.nanmax(np.abs(mat))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                continue
            c = lut[(adapters[i], col_order[j])]
            leak = c["delta_raw"] >= 0.10 and c["ci95_low"] > 0
            ax.text(
                j,
                i,
                f"{v:+.2f}",
                ha="center",
                va="center",
                fontsize=7.5,
                fontweight="bold" if leak else "normal",
                color="white" if abs(v) > 0.45 else "black",
            )
            if c["diagonal_cell"]:
                ax.add_patch(
                    plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", lw=1.6)
                )
    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels(
        [PERSONA_LABELS[p] for p in col_order],
        rotation=30,
        ha="right",
        fontsize=8,
    )
    ax.set_yticks(range(len(adapters)))
    ax.set_yticklabels([ADAPTER_LABELS[a] for a in adapters], fontsize=9)
    ax.set_title(
        "Cross-matrix: agreement-rate delta for every new persona under every adapter "
        "(black outline = twin under its own source)",
        fontsize=10.5,
        loc="left",
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("delta (trained - base)", fontsize=8)
    savefig_paper(fig, "e2_cross_matrix", dir=FIGDIR)
    plt.close(fig)


def fig_leak_map(cells, panels):
    set_paper_style("blog")
    bystanders = sorted({c["bystander"] for c in cells})
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), constrained_layout=True)
    titles = {
        "sycophancy": "Sycophancy",
        "refusal": "Refusal",
        "em": "Emergent misalignment (survivor-rate measure)",
    }
    for ax, beh in zip(axes, BEHAVIORS, strict=True):
        grid = np.full((len(SOURCES), len(bystanders)), np.nan)
        for c in (c for c in cells if c["behavior"] == beh):
            i = SOURCES.index(c["source"])
            if c["bystander"] in bystanders:
                grid[i, bystanders.index(c["bystander"])] = c["delta"]
        vmax = max(0.2, np.nanmax(np.abs(grid)))
        im = ax.imshow(grid, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        for c in (c for c in cells if c["behavior"] == beh):
            i = SOURCES.index(c["source"])
            j = bystanders.index(c["bystander"])
            if c["leak"]:
                ax.add_patch(
                    plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="black", lw=1.6)
                )
            if c["neg_member"]:
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, fill=False, ec="grey", lw=0.8, hatch="///"
                    )
                )
        max_cos = {p["source"]: p["max_bystander_cos"] for p in panels if p["behavior"] == beh}
        ax.set_yticks(range(len(SOURCES)))
        ax.set_yticklabels(
            [f"{SOURCE_LABELS[s]} (max cos {max_cos.get(s, float('nan')):.3f})" for s in SOURCES],
            fontsize=8,
        )
        ax.set_xticks(range(len(bystanders)))
        ax.set_xticklabels([_bystander_label(b) for b in bystanders], rotation=90, fontsize=7)
        ax.set_title(titles[beh], fontsize=11)
        fig.colorbar(im, ax=ax, label="leakage delta (trained - base)", shrink=0.8)
    fig.suptitle(
        "Per-cell leakage delta by behavior — leak cells outlined, training negatives hatched",
        fontsize=12,
    )
    savefig_paper(fig, "e1_leak_map_hero", dir=FIGDIR)
    plt.close(fig)


def fig_forest(per_behavior_fits, pooled_fit):
    """Factor forest WITHOUT the panel-constant self-delta coefficients.

    The cell-level Firth p-values for self-delta treat a panel-constant
    covariate as if it varied over 138 cells (effective n = 6 sources per
    behavior), so they are anti-conservative; the registered between-source
    permutation is the binding implant-strength inference. The coefficients
    are therefore omitted here rather than plotted with a warning.
    """
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(8, 5.2))
    rows = []
    for beh, fit in {**per_behavior_fits, "pooled": pooled_fit}.items():
        if fit is None or "names" not in fit:
            continue
        for j, name in enumerate(fit["names"]):
            if name == "intercept" or name.startswith("behavior_") or name == "self_delta":
                continue
            lo = fit.get("ci95_low_coef", [None] * len(fit["names"]))[j]
            hi = fit.get("ci95_high_coef", [None] * len(fit["names"]))[j]
            beh_label = BEHAVIOR_LABELS.get(beh, beh)
            rows.append(
                (
                    f"{beh_label}: {FACTOR_LABELS.get(name, name.replace('_', ' '))}",
                    fit["coef"][j],
                    lo,
                    hi,
                )
            )
    ys = np.arange(len(rows))[::-1]
    for y, (label, coef, lo, hi) in zip(ys, rows, strict=True):
        color = (
            paper_palette_role("primary") if "pooled" in label else paper_palette_role("neutral")
        )
        if lo is not None:
            ax.plot([lo, hi], [y, y], color=color, lw=1.5)
        ax.plot(coef, y, "o", color=color)
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel("Firth log-odds coefficient (z-scored factor), 95% profile CI")
    ax.set_title("Cell-level factor coefficients per behavior and pooled")
    ax.text(
        0.0,
        -0.16,
        "Implant-strength (self-delta) coefficients omitted: the covariate is panel-constant\n"
        "(effective n = 6 sources per behavior), so cell-level p-values are anti-conservative;\n"
        "the registered between-source permutation is the binding test (all p ≥ 0.24).",
        transform=ax.transAxes,
        fontsize=7,
        va="top",
        color="dimgrey",
    )
    savefig_paper(fig, "e1_factor_forest", dir=FIGDIR)
    plt.close(fig)


def fig_panel_table(panels):
    """Implant strength vs panel leakage with hand-placed, collision-free labels.

    The 18 panel points are a fixed dataset, so the crowded floor cluster
    (six points at n_leak = 0 between self-delta 0.65 and 0.97, plus the
    near-floor refusal cluster at self-delta ~1.0) gets explicit per-point
    offsets/anchors instead of an automatic stagger.
    """
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    markers = {"sycophancy": "o", "refusal": "s", "em": "^"}
    # (behavior, source) -> (dx pts, dy pts, ha); fallback (5, 5, "left")
    placements = {
        ("sycophancy", "qwen_default"): (-2, 8, "right"),
        ("refusal", "qwen_default"): (-2, -14, "right"),
        ("sycophancy", "comedian"): (-2, 8, "right"),
        ("sycophancy", "villain"): (0, -14, "center"),
        ("sycophancy", "kindergarten_teacher"): (-4, -26, "right"),
        ("refusal", "villain"): (-2, 8, "right"),
        ("refusal", "comedian"): (0, -12, "center"),
        ("refusal", "assistant"): (-6, -6, "right"),
        ("refusal", "kindergarten_teacher"): (-6, 8, "right"),
        ("em", "assistant"): (3, 6, "left"),
        ("em", "qwen_default"): (3, -14, "left"),
        ("em", "comedian"): (3, 8, "left"),
        ("em", "software_engineer"): (3, -14, "left"),
    }
    for beh in BEHAVIORS:
        sub = [p for p in panels if p["behavior"] == beh and p["self_delta"] is not None]
        if not sub:
            continue
        ax.scatter(
            [p["self_delta"] for p in sub],
            [p["n_leak_cells"] for p in sub],
            marker=markers[beh],
            label=BEHAVIOR_LABELS[beh],
            color=paper_palette_role(
                {"sycophancy": "primary", "refusal": "accent", "em": "control"}[beh]
            ),
        )
        for p in sub:
            x, y = p["self_delta"], p["n_leak_cells"]
            dx, dy, ha = placements.get((beh, p["source"]), (5, 5, "left"))
            ax.annotate(
                SOURCE_LABELS[p["source"]],
                (x, y),
                fontsize=6.5,
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
            )
    ax.set_xlabel("source self-implant delta (manipulation check)")
    ax.set_ylabel("leak cells on the panel (of 23)")
    ax.set_ylim(bottom=-3.5)
    ax.set_title("Implant strength vs panel leakage (18 panels)")
    ax.legend(fontsize=8)
    savefig_paper(fig, "e1_panel_self_delta_vs_leak", dir=FIGDIR)
    plt.close(fig)


def main():
    e2 = json.loads(E2.read_text())
    ct = json.loads(E1_CT.read_text())
    fa = json.loads(E1_FA.read_text())
    frozen_syco = [r for r in ct["cells"] if r["behavior"] == "sycophancy"]
    fig_hero(e2["cells"], frozen_syco)
    fig_cross_matrix(e2["cells"])
    fig_leak_map(ct["cells"], ct["panels"])
    fig_forest(fa["firth_per_behavior"], fa["firth_pooled"])
    fig_panel_table(ct["panels"])
    print("wrote round-2 figures to", FIGDIR)


if __name__ == "__main__":
    main()
