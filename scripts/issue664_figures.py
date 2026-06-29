#!/usr/bin/env python
"""issue #664 figures (analysis-time). Hero + supporting figures for the marker
gate spine ĝ^real ground-truth-leakage story.

Figures:
  1. HERO: gate heatmap — bystander ĝ^real (best-SNR layer) for the 4 marker
     sources × bystander contexts (grouped by #594 family). The near→far gate
     structure / leakage map. (contrastive dose-1, seed 42.)
  2. bystander ĝ^real distribution per source (strip + box) — the low-level
     per-context data behind the heatmap means.
  3. leakage-variation SNR vs probe-split noise floor, per cell (forest) — the
     kill 3(b) read: does cross-context leakage variation exceed the floor?
  4. arm × dose ĝ^real bystander-spread comparison (the install-confound read).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

REPO = Path(__file__).resolve().parent.parent
GATE = REPO / "eval_results" / "issue_664" / "gate_real"
FIGDIR = "figures/"

# #594 family of each context id (prefix-coded f1..f8 -> family)
FAMILY = {
    "f1": "persona",
    "f2": "wildchat",
    "f3": "icl",
    "f4": "rephrase",
    "f5": "format",
    "f6": "default",
    "f8": "behavior",
}
SOURCE_LABEL = {
    "default": "default assistant",
    "librarian": "librarian",
    "surgeon": "surgeon",
    "programmer": "programmer",
}


def fam_of(cid: str) -> str:
    return FAMILY.get(cid.split("_")[0], "other")


def load_cell(cell):
    p = GATE / cell / "g_real.json"
    return json.load(open(p)) if p.exists() else None


def hero_heatmap(summary):
    """ĝ^real bystander map: 4 sources (rows) × bystander contexts (cols), best layer."""
    sources = ["default", "librarian", "surgeon", "programmer"]
    cells = {
        c["source"]: c
        for c in summary["cells"]
        if c["behavior"] == "marker"
        and c["arm"] == "contra"
        and c["dose"] == "d1"
        and c["seed"] == 42
    }
    # use a FIXED mid-layer (14) for the hero — selection-bias-free; the
    # near-zero bystander message is layer-robust, so a cherry-picked layer
    # would only overstate structure.
    layer = 14
    # build the matrix from per-cell g_real (bystander rows only), ordered by family
    full = {s: load_cell(cells[s]["cell"]) for s in sources if s in cells}
    # context ordering: by family, then id (use first source's order)
    any_rows = next(iter(full.values()))["rows"]
    bys_ids = [r["context_id"] for r in any_rows if r["target_context_role"] == "bystander"]
    bys_ids = sorted(
        bys_ids,
        key=lambda c: (
            list(FAMILY.values()).index(fam_of(c)) if fam_of(c) in FAMILY.values() else 99,
            c,
        ),
    )
    mat = np.full((len(sources), len(bys_ids)), np.nan)
    for si, s in enumerate(sources):
        rec = full.get(s)
        if rec is None:
            continue
        gmap = {r["context_id"]: r["ghat_by_layer"][layer] for r in rec["rows"]}
        for ci, cid in enumerate(bys_ids):
            mat[si, ci] = gmap.get(cid, np.nan)

    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(12, 3.4))
    vmax = np.nanpercentile(np.abs(mat), 98)
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(sources)))
    ax.set_yticklabels([SOURCE_LABEL[s] for s in sources])
    # family band separators on x
    fams = [fam_of(c) for c in bys_ids]
    tick_pos, tick_lab, last = [], [], None
    for i, fm in enumerate(fams):
        if fm != last:
            tick_pos.append(i)
            tick_lab.append(fm)
            last = fm
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, rotation=30, ha="right", fontsize=8)
    ax.set_xlabel(
        f"bystander context (49 held-out contexts, grouped by family) — read at layer {layer}"
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.015)
    cbar.set_label("gate value (1.0 = full source write)", fontsize=8)
    set_title_subtitle(
        ax,
        "Marker write barely leaks: bystander gate values cluster near zero",
        "trained−base activation gate, 4 marker sources × 49 bystander contexts (contrastive, dose-1, seed 42); source ĝ=1 by construction, excluded",
        source="issue #664 trained store",
    )
    savefig_paper(fig, "issue_664/hero_gate_heatmap", dir=FIGDIR)
    plt.close(fig)
    return layer, bys_ids


def bystander_strip(summary, layer):
    """Per-source bystander ĝ distribution (strip + box) — the low-level data."""
    sources = ["default", "librarian", "surgeon", "programmer"]
    cells = {
        c["source"]: c
        for c in summary["cells"]
        if c["behavior"] == "marker"
        and c["arm"] == "contra"
        and c["dose"] == "d1"
        and c["seed"] == 42
    }
    data = []
    labels = []
    for s in sources:
        if s not in cells:
            continue
        rec = load_cell(cells[s]["cell"])
        vals = [
            r["ghat_by_layer"][layer]
            for r in rec["rows"]
            if r["target_context_role"] == "bystander"
        ]
        data.append(vals)
        labels.append(SOURCE_LABEL[s])
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    rng = np.random.default_rng(42)
    c_primary = paper_palette_role("primary")
    for i, vals in enumerate(data):
        x = np.full(len(vals), i) + rng.uniform(-0.13, 0.13, len(vals))
        ax.scatter(x, vals, s=18, alpha=0.55, color=c_primary, edgecolors="none", zorder=3)
    bp = ax.boxplot(
        data, positions=range(len(data)), widths=0.5, showfliers=False, patch_artist=True, zorder=2
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("none")
        patch.set_edgecolor(paper_palette_role("neutral"))
    ax.axhline(0, color="grey", lw=0.8, ls="--", zorder=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("bystander ĝ$^{real}$ (fraction of source write)")
    set_title_subtitle(
        ax,
        "Each marker source leaks a near-zero, narrowly-spread gate to bystanders",
        f"per-context ĝ$^{{real}}$ at layer {layer}; each point = one of 49 bystander contexts (contrastive, dose-1, seed 42)",
        source="issue #664 trained store",
    )
    savefig_paper(fig, "issue_664/bystander_ghat_strip", dir=FIGDIR)
    plt.close(fig)


def snr_forest(summary):
    """kill 3(b): per-cell signal vs probe-split floor (SNR forest). Shows BOTH
    the best-of-28-layers SNR (optimistic, selection-biased) and the fixed
    mid-layer (14) SNR (selection-bias-free) so the gap is visible."""
    cells = [c for c in summary["cells"] if c["behavior"] == "marker"]
    cells = sorted(cells, key=lambda c: (c["source"], c["arm"], c["dose"], c["seed"]))
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(8.5, max(4.0, 0.34 * len(cells))))
    ys = list(range(len(cells)))
    snr_best = [c["best_layer_snr"] for c in cells]
    snr_l14 = [c["fixed_L14_snr"] for c in cells]
    labels = [
        f"{SOURCE_LABEL.get(c['source'], c['source'])} · {'contrastive' if c['arm'] == 'contra' else 'positive-only'} · {c['dose']}"
        + (f" · seed{c['seed']}" if c["seed"] != 42 else "")
        for c in cells
    ]
    for y, sb, sl in zip(ys, snr_best, snr_l14):
        ax.plot([sl, sb], [y, y], color="lightgrey", lw=1, zorder=1)
    ax.scatter(
        snr_best,
        ys,
        color=paper_palette_role("primary"),
        s=42,
        zorder=3,
        label="best of 28 layers (selection-biased)",
    )
    ax.scatter(
        snr_l14,
        ys,
        color=paper_palette_role("baseline"),
        s=42,
        marker="D",
        zorder=3,
        label="fixed layer 14 (selection-bias-free)",
    )
    ax.axvline(1.0, color="crimson", lw=1.0, ls="--", zorder=2, label="noise floor (SNR=1)")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel(
        "leakage-variation SNR = cross-context ĝ std ÷ within-context probe-split noise floor"
    )
    ax.legend(loc="lower right", fontsize=7.5)
    set_title_subtitle(
        ax,
        "Marker leakage variation barely clears the noise floor — and at a fixed layer, doesn't",
        "per marker cell: bystander ĝ$^{real}$ spread vs within-context probe-split floor (kill-criterion 3b read)",
        source="issue #664 trained store",
    )
    savefig_paper(fig, "issue_664/snr_forest", dir=FIGDIR)
    plt.close(fig)


def behavior_snr(summary):
    """Content behaviors vs marker: per-cell best-layer SNR, grouped by behavior.
    The secondary finding — content-behavior gate variation >> marker's."""
    BEH_LABEL = {
        "marker": "marker (hidden token)",
        "fact": "taught fact",
        "bad_medical": "bad-medical advice",
        "em": "insecure-code / EM",
        "tf_rev": "reversed-fact (null)",
        "ic_edu": "educational-code (null)",
    }
    order = ["marker", "fact", "bad_medical", "em", "ic_edu", "tf_rev"]
    by_beh = defaultdict(list)
    for c in summary["cells"]:
        by_beh[c["behavior"]].append(c["best_layer_snr"])
    behs = [b for b in order if b in by_beh]
    set_paper_style("blog")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    rng = np.random.default_rng(42)
    for i, b in enumerate(behs):
        vals = by_beh[b]
        col = paper_palette_role("primary") if b == "marker" else paper_palette_role("accent")
        x = np.full(len(vals), i) + rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(x, vals, s=34, alpha=0.7, color=col, edgecolors="none", zorder=3)
        ax.scatter(
            [i], [np.median(vals)], marker="_", s=600, color=paper_palette_role("neutral"), zorder=4
        )
    ax.axhline(1.0, color="crimson", lw=1.0, ls="--", zorder=2, label="noise floor (SNR=1)")
    ax.set_xticks(range(len(behs)))
    ax.set_xticklabels([BEH_LABEL.get(b, b) for b in behs], rotation=18, ha="right", fontsize=8.5)
    ax.set_ylabel("leakage SNR (best-layer)")
    ax.legend(loc="upper right", fontsize=8)
    set_title_subtitle(
        ax,
        "Content behaviors leak with far more cross-context structure than the marker",
        "per-cell best-of-28-layers ĝ$^{real}$ SNR by behavior; each point one trained cell, bar = median",
        source="issue #664 trained store",
    )
    savefig_paper(fig, "issue_664/behavior_snr", dir=FIGDIR)
    plt.close(fig)


def main():
    summary = json.load(open(GATE / "gate_real_summary.json"))
    layer, _ = hero_heatmap(summary)
    bystander_strip(summary, layer)
    snr_forest(summary)
    behavior_snr(summary)
    print(f"figures written to figures/issue_664/ (hero layer {layer})")


if __name__ == "__main__":
    main()
