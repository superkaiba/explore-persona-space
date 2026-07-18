"""Issue #1417 — analyzer figures (first-pass interpretation).

Four figures over the committed eval JSONs (no recompute):
  1. fit_pathology_fingerprint — per cell: L19 held-out R^2 for the all-rows
     (n=4724), judge-kept, and matched-n (516, 5 draws) fits. Shows the
     GCV-collapse fingerprint (kept fits crash while super- and sub-sets fit).
  2. ceiling_vs_numerator — per ctx battery pair: within-reference ceiling
     (the REL denominator) vs composed-transport numerator at L19, with the
     full-n anchor values. Shows the verdict denominators are broken-negative
     while numerators are healthy-positive.
  3. healthy_pair_battery — the one fully-healthy battery pair
     (base model, non-user addressee vs helpful-instruction reference).
  4. c2_manipulation_checks — rude-but-informative cell: judge keep fraction
     + answer-variance ratio vs the pre-registered floors.

Usage: uv run python scripts/issue1417_analyzer_figs.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.analysis.paper_plots import (
    paper_palette,
    savefig_paper,
    set_paper_style,
)

OUT = Path("figures/issue_1417")
EV = Path("eval_results/issue_1417")
CELLS = ("c1_helpful_ctrl", "c2_rude", "c3_evasive", "c4_exposition", "c5_ai_addressee")
CELL_LABELS = {
    "c1_helpful_ctrl": "Helpful\ninstruction",
    "c2_rude": "Rude-but-\ninformative",
    "c3_evasive": "Evasive",
    "c4_exposition": "Addressee-free\nexposition",
    "c5_ai_addressee": "Non-user\naddressee",
}
MODELS = ("instruct", "pretrained")
MODEL_TITLES = {"instruct": "Instruct", "pretrained": "Base"}
ANCHORS = {  # G1-verified full-n reference map R^2 at L19
    ("instruct", "chat"): 0.6542,
    ("instruct", "nat"): 0.6249,
    ("pretrained", "chat"): 0.5416,
    ("pretrained", "nat"): 0.5783,
}


def _cell_fit(cell_id: str) -> dict | None:
    p = EV / "cells" / f"cells_{cell_id}.json"
    d = json.loads(p.read_text())
    if d.get("skipped_empty_rows") or d.get("skipped_too_few_rows"):
        return None
    return d


def fig_fingerprint() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
    pal = paper_palette(3)
    for ax, model in zip(axes, MODELS):
        for i, cell in enumerate(CELLS):
            xs = {"all": i - 0.22, "kept": i, "matched": i + 0.22}
            d_all = _cell_fit(f"{cell}__{model}__ctx__all")
            d_kept = _cell_fit(f"{cell}__{model}__ctx")
            r_all = d_all["r2_per_layer_obs"][19]
            r_kept = d_kept["r2_per_layer_obs"][19]
            n_kept = d_kept["r2_bootstrap_ci_frozen_layers"]["19"]["n"]
            matched = []
            for k in range(5):
                dm = _cell_fit(f"{cell}__{model}__ctx__matched{k}")
                if dm is not None:
                    matched.append(dm["r2_per_layer_obs"][19])
            ax.scatter(
                [xs["all"]],
                [r_all],
                s=90,
                color=pal[0],
                zorder=3,
                marker="o",
                label="all rows (n=4724)" if i == 0 else None,
            )
            ax.scatter(
                [xs["kept"]],
                [r_kept],
                s=110,
                color=pal[1],
                zorder=3,
                marker="s",
                label="judge-kept rows" if i == 0 else None,
            )
            ax.scatter(
                [xs["matched"]] * len(matched),
                matched,
                s=32,
                color=pal[2],
                zorder=3,
                marker="D",
                alpha=0.85,
                label="matched-n draws (n=516)" if i == 0 else None,
            )
            ax.plot(
                [xs["all"], xs["kept"], xs["matched"]],
                [r_all, r_kept, float(np.mean(matched))],
                color="0.65",
                lw=1.0,
                zorder=2,
            )
            ax.text(
                xs["kept"],
                r_kept - 0.09,
                f"n={n_kept}",
                ha="center",
                va="top",
                fontsize=9,
                color="0.35",
            )
        ax.axhline(0.0, color="0.5", lw=0.8, ls=":")
        ax.axhline(-0.03, color="0.5", lw=0.8, ls="--")
        ax.set_xticks(range(len(CELLS)))
        ax.set_xticklabels([CELL_LABELS[c] for c in CELLS], fontsize=10)
        ax.set_title(MODEL_TITLES[model], loc="left", fontweight="bold")
    axes[0].set_ylabel("held-out $R^2$ at layer 19 (context arm)")
    axes[0].set_ylim(-1.78, 0.9)
    axes[0].legend(loc="center left", fontsize=9)
    fig.suptitle(
        "Same rows, three fits: judge-kept fits collapse while their supersets and subsets fit"
        " (dashed: shuffle-null level)",
        fontsize=12,
        y=1.005,
    )
    fig.tight_layout()
    savefig_paper(fig, "fit_pathology_fingerprint", dir=OUT)
    plt.close(fig)


def fig_ceiling_vs_numerator() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.0), sharex=True)
    pal = paper_palette(2)
    pair_specs = [
        ("c1_helpful_ctrl", "c0_chat", "chat", "helpful instr. vs chat ref"),
        ("c2_rude", "c0_chat", "chat", "rude vs chat ref"),
        ("c2_rude", "c1", None, "rude vs helpful-instr ref"),
        ("c3_evasive", "c0_chat", "chat", "evasive vs chat ref"),
        ("c3_evasive", "c1", None, "evasive vs helpful-instr ref"),
        ("c4_exposition", "c0_chat", "chat", "exposition vs chat ref"),
        ("c4_exposition", "c0p_nat", "nat", "exposition vs plain-text ref"),
        ("c5_ai_addressee", "c0_chat", "chat", "AI-addressee vs chat ref"),
        ("c5_ai_addressee", "c1", None, "AI-addressee vs helpful-instr ref"),
    ]
    for ax, model in zip(axes, MODELS):
        ylabels = []
        for j, (cell, ref, anchor_key, label) in enumerate(pair_specs):
            p = EV / "battery" / f"battery_{model}__{cell}__vs_{ref}__ctx.json"
            if not p.exists():
                ylabels.append(label + " (absent)")
                continue
            d = json.loads(p.read_text())
            L = d["rel_by_layer"]["19"]
            num, ceil = L["numerator_r2"], L["ceiling_r2"]
            n = d["n_rows"]
            y = len(pair_specs) - 1 - j
            ax.plot([ceil, num], [y, y], color="0.7", lw=1.4, zorder=2)
            ax.scatter(
                [ceil],
                [y],
                s=80,
                marker="o",
                color=pal[1],
                zorder=3,
                label="within-reference ceiling (REL denominator)" if j == 0 else None,
            )
            ax.scatter(
                [num],
                [y],
                s=80,
                marker="s",
                color=pal[0],
                zorder=3,
                label="composed-transport numerator" if j == 0 else None,
            )
            ax.text(max(num, ceil) + 0.06, y, f"n={n}", va="center", fontsize=8.5, color="0.35")
            ylabels.append(label)
        ax.axvline(0.0, color="0.4", lw=0.9, ls=":")
        for key, ls in (("chat", "--"), ("nat", "-.")):
            ax.axvline(ANCHORS[(model, key)], color="0.55", lw=0.9, ls=ls)
        ax.set_yticks(range(len(pair_specs))[::-1])
        ax.set_yticklabels([lbl for lbl in ylabels], fontsize=9.5)
        ax.set_title(MODEL_TITLES[model], loc="left", fontweight="bold")
        ax.set_xlabel("held-out $R^2$ at layer 19")
        ax.set_xlim(-1.6, 1.0)
    axes[0].legend(loc="lower left", fontsize=9)
    fig.suptitle(
        "REL verdict components per battery pair: denominators (ceilings) broke negative;"
        " numerators stayed positive (dashed/dash-dot: full-n chat / plain-text anchors)",
        fontsize=12,
        y=1.005,
    )
    fig.tight_layout()
    savefig_paper(fig, "ceiling_vs_numerator", dir=OUT)
    plt.close(fig)


def fig_healthy_pair() -> None:
    d = json.loads(
        (EV / "battery" / "battery_pretrained__c5_ai_addressee__vs_c1__ctx.json").read_text()
    )
    b = d["battery_per_layer"]["19"]
    L = d["rel_by_layer"]["19"]
    pc = d["procrustes_cosine_null_l19"]
    tr_f = d["transfer_ref_on_cell"]["r2_by_layer"]["19"]
    tr_f_null = d["transfer_ref_on_cell"]["null_p975_by_layer"]["19"]
    tr_r = d["transfer_cell_on_ref"]["r2_by_layer"]["19"]
    tr_r_null = d["transfer_cell_on_ref"]["null_p975_by_layer"]["19"]
    names = [
        "reference map,\nown rows (ceiling)",
        "cell map,\nown rows (ceiling)",
        "composed transport\ncell→reference",
        "composed transport\nreference→cell",
        "frozen ref map\non cell rows",
        "frozen cell map\non ref rows",
    ]
    vals = [
        b["ceilings"]["within_instruct"],
        b["ceilings"]["within_base"],
        L["numerator_r2"],
        L["rel_reverse"] * b["ceilings"]["within_base"],
        tr_f,
        tr_r,
    ]
    nulls = [None, None, None, None, tr_f_null, tr_r_null]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [3.2, 1]})
    pal = paper_palette(6)
    xs = np.arange(len(names))
    ax.bar(xs, vals, color=[pal[0], pal[0], pal[1], pal[1], pal[2], pal[2]], width=0.62)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.012, f"{v:.3f}", ha="center", fontsize=9.5)
    for x, nl in zip(xs, nulls):
        if nl is not None:
            ax.plot([x - 0.31, x + 0.31], [nl, nl], color="0.3", lw=1.2, ls="--")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("held-out $R^2$ at layer 19")
    ax.set_title(
        "Base model, non-user addressee vs helpful-instruction reference (n=1588)"
        " — REL 0.937 [0.928, 0.946]",
        loc="left",
        fontsize=11,
    )
    cos_vals = [pc["raw_vec_cosine"], pc["observed_aligned_cosine"]]
    ax2.bar([0, 1], cos_vals, color=[pal[3], pal[4]], width=0.55)
    for x, v in zip([0, 1], cos_vals):
        ax2.text(x, v + 0.012, f"{v:.3f}", ha="center", fontsize=9.5)
    ax2.plot([-0.35, 1.35], [pc["null_p975"]] * 2, color="0.3", lw=1.2, ls="--")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["raw map\ncosine", "rotation-aligned\ncosine"], fontsize=9)
    ax2.set_ylabel("cosine similarity")
    ax2.set_title("map cosines\n(dashed: 100-draw chance band)", fontsize=10, loc="left")
    fig.tight_layout()
    savefig_paper(fig, "healthy_pair_battery", dir=OUT)
    plt.close(fig)


def fig_c2_checks() -> None:
    yr = json.loads((EV / "judge" / "yield_report.json").read_text())["cells"]
    bs = json.loads((EV / "battery_summary.json").read_text())["cells"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.6))
    pal = paper_palette(2)
    width = 0.38
    xs = np.arange(len(CELLS))
    for k, model in enumerate(MODELS):
        yv = [yr[f"{model}_{c}"]["yield_frac"] for c in CELLS]
        ax1.bar(xs + (k - 0.5) * width, yv, width=width, color=pal[k], label=MODEL_TITLES[model])
        for x, v in zip(xs + (k - 0.5) * width, yv):
            ax1.text(x, v + 0.012, f"{v:.2f}", ha="center", fontsize=8.5)
        vv = [bs[f"{model}__{c}"]["y_var_ratio_vs_c0"] for c in CELLS]
        ax2.bar(xs + (k - 0.5) * width, vv, width=width, color=pal[k], label=MODEL_TITLES[model])
        for x, v in zip(xs + (k - 0.5) * width, vv):
            ax2.text(x, v + 0.012, f"{v:.2f}", ha="center", fontsize=8.5)
    for ax, floor, lab in (
        (ax1, 0.5, "judge keep fraction"),
        (ax2, 0.5, "answer-variance ratio vs chat reference"),
    ):
        ax.axhline(floor, color="0.3", lw=1.1, ls="--")
        ax.set_xticks(xs)
        ax.set_xticklabels([CELL_LABELS[c] for c in CELLS], fontsize=9)
        ax.set_ylabel(lab)
    ax1.set_title("Register-compliance yield (dashed: 50% primary floor)", loc="left", fontsize=11)
    ax2.set_title(
        "Content-collapse diagnostic (dashed: 0.5 demotion floor)", loc="left", fontsize=11
    )
    ax1.legend(fontsize=9)
    fig.tight_layout()
    savefig_paper(fig, "c2_manipulation_checks", dir=OUT)
    plt.close(fig)


def main() -> None:
    set_paper_style("blog")
    OUT.mkdir(parents=True, exist_ok=True)
    fig_fingerprint()
    fig_ceiling_vs_numerator()
    fig_healthy_pair()
    fig_c2_checks()
    print("wrote 4 figures to", OUT)


if __name__ == "__main__":
    main()
