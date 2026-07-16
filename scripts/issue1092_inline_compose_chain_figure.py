#!/usr/bin/env python3
"""Two-panel figure for the #1092 inline composition-chain test.

Panel A: STAGE 1 held-out R2(v̂_C, v_C) per h-form vs the identity floor
         (does the CONTEXT state compose from disjoint parts; operator vs additive).
Panel B: END-TO-END answer R2 (pca48 headline) — prefix-only, query-only, the
         additive answer stitch, the chain M'(v̂_C) per h-form, and the
         full-context ceiling (does routing through v̂_C close the stitch->full gap).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

OUT_JSON = PROJECT_ROOT / "eval_results/issue_1092/inline_compose_chain/compose_chain.json"
FIG_DIR = PROJECT_ROOT / "figures/issue_1092"
FIG_STEM = "inline_compose_chain"

S1_LABELS = {
    "identity": "identity\n(v̂_C = v_P)",
    "additive_meanoffset": "additive\n(query mean offset)",
    "additive_state": "additive\n(query-state ridge)",
    "joint_linear": "joint linear\n[v_P; v_q]",
}


def main() -> None:
    d = json.loads(OUT_JSON.read_text())
    s1 = d["stage1_context_reconstruction"]
    meta = d["meta"]
    ranks = meta["operator"]["ranks"]
    s2 = d["stage2_end_to_end"]["pca48"]
    commit = meta.get("git_commit", "?")[:9]

    set_paper_style()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.4), layout="constrained")

    # ---- Panel A: stage-1 context reconstruction ----
    s1_order = ["identity", "additive_meanoffset", "additive_state", "joint_linear"]
    s1_order += [f"operator_r{r}" for r in ranks]
    labelsA, valsA, colorsA = [], [], []
    for f in s1_order:
        if f.startswith("operator_r"):
            labelsA.append(f"operator\n(rank {f.split('_r')[1]})")
            colorsA.append(paper_palette_role("accent"))
        else:
            labelsA.append(S1_LABELS[f])
            colorsA.append(
                paper_palette_role("baseline") if f == "identity" else paper_palette_role("primary")
            )
        valsA.append(s1[f]["r2_ambient"])
    xA = range(len(labelsA))
    axA.bar(xA, valsA, color=colorsA, width=0.68, zorder=3)
    axA.axhline(
        s1["identity"]["r2_ambient"],
        ls="--",
        lw=1.2,
        color=paper_palette_role("baseline"),
        zorder=2,
    )
    for x, v in zip(xA, valsA, strict=False):
        axA.annotate(
            f"{v:.3f}",
            (x, v),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    axA.set_xticks(list(xA))
    axA.set_xticklabels(labelsA, fontsize=8.5)
    axA.set_ylabel("held-out R² (v̂_C vs true v_C, ambient)")
    axA.set_ylim(0, max(valsA) * 1.12)
    axA.grid(axis="y", alpha=0.3, zorder=0)
    set_title_subtitle(
        axA,
        "Stage 1 — does the context state compose from prefix + query?",
        "held-out reconstruction of v_C; identity floor = v̂_C = v_P (dashed)",
    )

    # ---- Panel B: end-to-end answer prediction (pca48) ----
    chain = s2["chain_through_vhat_C"]
    base = s2["baselines"]
    best_op = max((f"operator_r{r}" for r in ranks), key=lambda f: chain[f])
    barsB = [
        ("prefix-only\n(direct)", base["prefix_only"], paper_palette_role("neutral")),
        ("query-only\n(direct)", base["query_only"], paper_palette_role("neutral")),
        (
            "answer stitch\n[v_P; v_q]→ans",
            base["additive_answer_stitch"],
            paper_palette_role("baseline"),
        ),
        ("chain: additive", chain["additive_state"], paper_palette_role("primary")),
        ("chain: joint linear", chain["joint_linear"], paper_palette_role("primary")),
        (
            f"chain: operator\n({best_op.replace('operator_', 'rank ')})",
            chain[best_op],
            paper_palette_role("accent"),
        ),
        ("full-context\n(ceiling)", base["full_context_ceiling"], paper_palette_role("control")),
    ]
    labelsB = [b[0] for b in barsB]
    valsB = [b[1] for b in barsB]
    colorsB = [b[2] for b in barsB]
    xB = range(len(labelsB))
    axB.bar(xB, valsB, color=colorsB, width=0.68, zorder=3)
    axB.axhline(
        base["additive_answer_stitch"],
        ls="--",
        lw=1.0,
        color=paper_palette_role("baseline"),
        zorder=2,
    )
    axB.axhline(
        base["full_context_ceiling"], ls="--", lw=1.0, color=paper_palette_role("control"), zorder=2
    )
    for x, v in zip(xB, valsB, strict=False):
        axB.annotate(
            f"{v:.3f}",
            (x, v),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    axB.set_xticks(list(xB))
    axB.set_xticklabels(labelsB, fontsize=8.0)
    axB.set_ylabel("held-out R² (answer target, pca48)")
    axB.set_ylim(min(0, min(valsB) * 1.1), max(valsB) * 1.12)
    axB.grid(axis="y", alpha=0.3, zorder=0)
    set_title_subtitle(
        axB,
        "Stage 2 — does routing through v̂_C beat the answer stitch?",
        "M'(v̂_C) chain vs the additive [v_P; v_q]→answer stitch (dashed) and full-context ceiling",
    )

    src = (
        f"#1092 cell_inst_own L14, n_be={meta['n_be']} battery-excluded rows, "
        f"novel-prefix {meta['n_prefix']}-prefix grouped 6-fold (seed 0); teacher-forced state "
        f"capture, own-policy greedy answers; PRESS-LOO ridge. commit {commit}"
    )
    fig.text(0.5, -0.02, src, ha="center", va="top", fontsize=7.2, color="#7A7A7A")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = savefig_paper(fig, FIG_STEM, dir=str(FIG_DIR), formats=("png",))
    print(f"wrote {paths}")


if __name__ == "__main__":
    main()
