"""Reader-facing figures for the #923 pooled-span-features follow-up round.

Regenerates the three body-embedded figures from the committed round JSONs
(`eval_results/issue_923/fits_pooled/` vs the parent `fits/`), with
plain-English labels per the paper-plots skill:

1. ``pooled_hero_L18_uc`` — paired bars, last-token vs span-mean summary,
   five read-out arms at layer 18 on the UltraChat grid (95% bootstrap CIs).
2. ``pooled_family_cells_L18_uc`` — per-held-out-family skill dots (top) and
   per-cell predicted-vs-actual scatters for the span-mean fits (bottom).
3. ``pooled_layer_curves_uc`` — pooled span-mean skill by layer for the five
   headline arms, with the parent last-token full/stitched curves as
   references.

Run from the issue-923 worktree root:
    uv run python scripts/issue923_fig_pooled_body.py
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
)

ROOT = Path(__file__).resolve().parents[1]
POOLED = ROOT / "eval_results/issue_923/fits_pooled"
PARENT = ROOT / "eval_results/issue_923/fits"
FIGDIR = "issue_923"

ARMS = ["arm_ctx", "arm_qry_i", "arm_concat_i", "arm_blend", "arm_full"]
ARM_LABELS = {
    "arm_ctx": "Context-only",
    "arm_qry_i": "Query-only\n(empty system)",
    "arm_concat_i": "Stitched pair",
    "arm_blend": "Blended\npredictions",
    "arm_full": "Full prompt",
}
FAMILY_LABELS = {
    "persona": "persona",
    "wildchat": "WildChat",
    "icl": "in-context\nlearning",
    "rephrase": "rephrase",
    "format": "format",
    "behavior": "behavior",
    "default": "default",
}


def load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def bar_data(headline: dict, genre: str) -> dict[str, tuple[float, float, float]]:
    """arm -> (skill, err_lo, err_hi) at the headline layer."""
    l18 = headline["stats"][genre]["L18"]
    out = {}
    for arm in ARMS:
        e = l18[arm]
        s = e["skill"]
        lo, hi = e["ci95"]
        out[arm] = (s, s - lo, hi - s)
    return out


def fig_hero(pooled: dict, parent: dict) -> None:
    last = bar_data(parent, "uc")
    pool = bar_data(pooled, "uc")
    x = np.arange(len(ARMS))
    w = 0.38
    fig, ax = plt.subplots()
    for off, data, label, role in (
        (-w / 2, last, "Last-token summary (parent round)", "baseline"),
        (+w / 2, pool, "Span-mean summary (this round)", "primary"),
    ):
        vals = [data[a][0] for a in ARMS]
        yerr = np.array([[data[a][1] for a in ARMS], [data[a][2] for a in ARMS]])
        ax.bar(
            x + off,
            vals,
            width=w,
            yerr=yerr,
            capsize=3,
            color=paper_palette_role(role),
            label=label,
            error_kw={"elinewidth": 1.1},
        )
    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABELS[a] for a in ARMS])
    ax.set_ylabel("Pooled held-out skill R² (layer 18)")
    ax.set_title("Read-out arm skill under the two feature summaries, UltraChat grid", pad=14)
    ax.legend(loc="upper left")
    savefig_paper(fig, f"{FIGDIR}/pooled_hero_L18_uc", dir="figures/")
    plt.close(fig)


def fam_skills(decomp: dict, genre: str, arm: str) -> list[float]:
    e = decomp["genres"][genre]["18"]["arms"][arm]
    return [1.0 - r / t for r, t in zip(e["fam_res"], e["fam_tot"])]


def fig_family_cells(pooled_decomp: dict, parent_decomp: dict, families: list[str]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.2))
    xt = np.arange(len(families))
    for col, arm in enumerate(["arm_full", "arm_concat_i"]):
        ax = axes[0][col]
        ax.scatter(
            xt,
            fam_skills(parent_decomp, "uc", arm),
            color=paper_palette_role("baseline"),
            s=42,
            label="last-token (parent)",
            zorder=3,
        )
        ax.scatter(
            xt,
            fam_skills(pooled_decomp, "uc", arm),
            color=paper_palette_role("primary"),
            s=42,
            label="span-mean (this round)",
            zorder=4,
        )
        ax.set_xticks(xt)
        ax.set_xticklabels([FAMILY_LABELS[f] for f in families], fontsize=8)
        ax.set_ylim(-0.12, 0.72)
        ax.set_title(
            f"{'Full prompt' if arm == 'arm_full' else 'Stitched pair'}: per-held-out-family skill",
            fontsize=11,
        )
        if col == 0:
            ax.set_ylabel("Held-out family skill R²")
            ax.legend(loc="lower left", fontsize=8)
    for col, arm in enumerate(["arm_full", "arm_concat_i"]):
        ax = axes[1][col]
        e = pooled_decomp["genres"]["uc"]["18"]["arms"][arm]
        act = np.asarray(e["cell_act_pc1"])
        pred = np.asarray(e["cell_pred_pc1"])
        ax.scatter(act, pred, s=4, alpha=0.12, color=paper_palette_role("primary"))
        lims = [min(act.min(), pred.min()), max(act.max(), pred.max())]
        ax.plot(lims, lims, ls="--", lw=0.8, color="0.4")
        ax.set_xlabel("Actual (top target principal component)")
        ax.set_title(
            f"{'Full prompt' if arm == 'arm_full' else 'Stitched pair'}: per-cell predictions, span-mean (n=7200)",
            fontsize=11,
        )
        if col == 0:
            ax.set_ylabel("Predicted")
    fig.tight_layout()
    savefig_paper(fig, f"{FIGDIR}/pooled_family_cells_L18_uc", dir="figures/")
    plt.close(fig)


def fig_layer_curves(pooled: dict, parent: dict) -> None:
    curves_pool = pooled["stats"]["uc"]["layer_curves"]
    curves_last = parent["stats"]["uc"]["layer_curves"]
    layers = np.arange(28)
    fig, ax = plt.subplots()
    role_map = [
        ("arm_ctx", "Context-only", "control"),
        ("arm_qry_i", "Query-only (empty system)", "accent"),
        ("arm_concat_i", "Stitched pair", "primary"),
        ("arm_blend", "Blended predictions", "neutral"),
        ("arm_full", "Full prompt", "baseline"),
    ]
    for arm, label, role in role_map:
        ax.plot(
            layers,
            curves_pool[arm],
            label=f"{label} (span-mean)",
            color=paper_palette_role(role),
            lw=1.8,
        )
    ax.plot(
        layers,
        curves_last["arm_full"],
        ls="--",
        color="0.25",
        lw=1.4,
        label="Full prompt (last-token, parent)",
    )
    ax.plot(
        layers,
        curves_last["arm_concat_i"],
        ls=":",
        color="0.25",
        lw=1.6,
        label="Stitched pair (last-token, parent)",
    )
    ax.axvline(18, color="0.7", lw=0.9, ls="-")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Pooled held-out skill R²")
    ax.set_title("Skill by layer under the span-mean summary, UltraChat grid", pad=14)
    ax.legend(fontsize=8, ncol=2)
    savefig_paper(fig, f"{FIGDIR}/pooled_layer_curves_uc", dir="figures/")
    plt.close(fig)


def main() -> None:
    set_paper_style("blog")
    pooled = load(POOLED / "headline.json")
    parent = load(PARENT / "headline.json")
    pooled_decomp = load(POOLED / "decomposition_skill.json")
    parent_decomp = load(PARENT / "decomposition_skill.json")
    families = parent["stats"]["uc"]["families"]
    fig_hero(pooled, parent)
    fig_family_cells(pooled_decomp, parent_decomp, families)
    fig_layer_curves(pooled, parent)
    print("[script-complete] 3 figures written to figures/issue_923/")


if __name__ == "__main__":
    main()
