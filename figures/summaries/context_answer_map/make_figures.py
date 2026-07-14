"""Synthesis figures for the context->answer map research line.

Two figures:
  1. map_strength_across_regimes.png  (two panels)
  2. transfer_specificity.png

All numbers are hardcoded from the source task bodies (verified); per-value
source issue ids are recorded in the augmented <stem>.meta.json under
"value_sources". Follows the /paper-plots conventions (blog rcParams,
colorblind-safe palette, no plot-area text/arrow annotation overlays,
reference lines only).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
    set_title_subtitle,
)

OUT_DIR = "figures/summaries/context_answer_map"

set_paper_style("blog")
PRIMARY = paper_palette_role("primary")  # deep blue  -> context-based map
BASELINE = paper_palette_role("baseline")  # warm orange -> prefix-based map
CONTROL = paper_palette_role("control")  # green      -> shuffled floor
ACCENT = paper_palette_role("accent")  # warm red   -> MLP probe
NEUTRAL = paper_palette_role("neutral")  # slate grey -> reference lines


def _augment_meta(stem: str, value_sources: dict) -> None:
    """Add a value_sources block (per-value source issue ids) to the sidecar."""
    meta_path = Path(OUT_DIR) / f"{stem}.meta.json"
    meta = json.loads(meta_path.read_text())
    meta["value_sources"] = value_sources
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Figure 1 -- map_strength_across_regimes.png (two panels)
# ---------------------------------------------------------------------------
def figure1() -> None:
    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(16.5, 5.4), gridspec_kw={"width_ratios": [1.6, 1.0]}
    )

    # ---- Panel A: does the linear map exist? (held-out R^2, ridge) ----
    # Each entry: (x position, label, context R^2, prefix R^2 or None, shuffled R^2 or None)
    # Regime groups separated by gaps in x.
    # Group 1 (#825 single-turn chat): x 0,1
    # Group 2 (#1092 real conversations): x 3,4,5,6 (context+prefix), 7,8 (shuffled floor)
    # Group 3 (#931 fiction raw text): x 10,11
    ctx_x, ctx_h, ctx_lbl = [], [], []
    pfx_x, pfx_h = [], []
    shf_x, shf_h = [], []

    def add_ctx(x, label, h):
        ctx_x.append(x)
        ctx_h.append(h)
        ctx_lbl.append((x, label))

    # Group 1 -- single-turn chat (LMSYS n=5000, #825)
    add_ctx(0, "Single-turn chat\ninstruct", 0.673)
    add_ctx(1, "Single-turn chat\npretrained", 0.588)
    # Group 2 -- real conversations (prefix x query, #1092): context + prefix arms
    real_cells = [
        (3, "Real conv\ninstruct,\nown ans", 0.799, 0.043),
        (4, "Real conv\npretrained,\nown ans", 0.714, 0.051),
        (5, "Real conv\npretrained,\nClaude ans", 0.742, 0.056),
        (6, "Real conv\npretrained,\ninstruct ans", 0.493, 0.079),
    ]
    W = 0.36
    for x, label, cval, pval in real_cells:
        add_ctx(x - W / 2, label, cval)  # left-shifted context bar
        pfx_x.append(x + W / 2)
        pfx_h.append(pval)
    # replace the last four ctx labels' anchor x with the group center for ticks
    # (handled below by using cell-center ticks)
    # Shuffled-pairing context floor (#1092)
    shf_cells = [(7, "Shuffled\nfloor\ninstruct", 0.08), (8, "Shuffled\nfloor\npretrained", 0.057)]
    for x, label, val in shf_cells:
        shf_x.append(x)
        shf_h.append(val)
        ctx_lbl.append((x, label))
    # Group 3 -- fiction raw text (#931)
    add_ctx(10, "Fiction\nnovels\n(author-blk)", -0.065)
    add_ctx(11, "Fiction\nmodel\nstories", 0.16)

    axA.axhline(0.0, color=NEUTRAL, linewidth=0.8, zorder=1)
    axA.bar(ctx_x, ctx_h, width=W, color=PRIMARY, label="context-based map", zorder=3)
    axA.bar(
        pfx_x,
        pfx_h,
        width=W,
        color=BASELINE,
        hatch="////",
        edgecolor="white",
        label="prefix-based map",
        zorder=3,
    )
    axA.bar(shf_x, shf_h, width=W, color=CONTROL, label="shuffled floor", zorder=3)

    # X ticks: cell centers. For the paired real-conv cells the tick sits at the
    # cell integer (between the two offset bars).
    tick_pos = [0, 1, 3, 4, 5, 6, 7, 8, 10, 11]
    tick_lbl = [
        "Single-turn\nchat\ninstruct",
        "Single-turn\nchat\npretrained",
        "Real conv\ninstruct,\nown ans",
        "Real conv\npretrained,\nown ans",
        "Real conv\npretrained,\nClaude ans",
        "Real conv\npretrained,\ninstruct ans",
        "Shuffled\nfloor\ninstruct",
        "Shuffled\nfloor\npretrained",
        "Fiction\nnovels\n(author-blk)",
        "Fiction\nmodel\nstories",
    ]
    axA.set_xticks(tick_pos)
    axA.set_xticklabels(tick_lbl, fontsize=7)
    # faint regime separators (reference lines, allowed)
    for xsep in (2.0, 9.0):
        axA.axvline(xsep, color="#DDDDDD", linewidth=0.8, zorder=0)
    axA.set_ylabel("held-out $R^2$ (ridge)")
    axA.set_ylim(-0.55, 0.92)
    axA.legend(loc="upper right", fontsize=8)
    set_title_subtitle(
        axA,
        "Does the linear map exist?",
        subtitle="held-out $R^2$, ridge; regimes: single-turn chat (#825) | real conversations (#1092) | fiction (#931)",
    )

    # ---- Panel B: two-turn cells -- format gates only the LINEAR map ----
    cellsB = [
        ("instruct\nchat", 0.076, 0.557),
        ("instruct\nnaturalistic", -0.078, 0.534),
        ("pretrained\nchat", -0.461, 0.487),
        ("pretrained\nnaturalistic", -0.390, 0.499),
    ]
    xb = list(range(len(cellsB)))
    ridge_h = [c[1] for c in cellsB]
    mlp_h = [c[2] for c in cellsB]
    wB = 0.38
    axB.axhline(0.0, color=NEUTRAL, linewidth=0.8, zorder=1)
    axB.bar(
        [x - wB / 2 for x in xb], ridge_h, width=wB, color=PRIMARY, label="ridge (linear)", zorder=3
    )
    axB.bar(
        [x + wB / 2 for x in xb],
        mlp_h,
        width=wB,
        color=ACCENT,
        label="MLP probe (nonlinear)",
        zorder=3,
    )
    axB.set_xticks(xb)
    axB.set_xticklabels([c[0] for c in cellsB], fontsize=8)
    axB.set_ylabel("held-out $R^2$")
    axB.set_ylim(-0.55, 0.65)
    axB.legend(loc="upper right", fontsize=8)
    set_title_subtitle(
        axB,
        "Two-turn cells: format gates only the LINEAR map",
        subtitle="#825, n=2000; ridge collapses, MLP recovers",
    )

    written = savefig_paper(fig, "map_strength_across_regimes", dir=OUT_DIR)
    plt.close(fig)
    _augment_meta(
        "map_strength_across_regimes",
        {
            "panelA_context": {
                "single_turn_chat_instruct=0.673": 825,
                "single_turn_chat_pretrained=0.588": 825,
                "realconv_instruct_own_answers=0.799": 1092,
                "realconv_pretrained_own_answers=0.714": 1092,
                "realconv_pretrained_claude_answers=0.742": 1092,
                "realconv_pretrained_instruct_answers=0.493": 1092,
                "fiction_novels_author_blocked=-0.065": 931,
                "fiction_model_stories=0.16": 931,
            },
            "panelA_prefix": {
                "realconv_instruct_own_answers=0.043": 1092,
                "realconv_pretrained_own_answers=0.051": 1092,
                "realconv_pretrained_claude_answers=0.056": 1092,
                "realconv_pretrained_instruct_answers=0.079": 1092,
            },
            "panelA_shuffled_floor": {
                "instruct=0.08": 1092,
                "pretrained=0.057": 1092,
            },
            "panelB_two_turn": {
                "instruct_chat_ridge=0.076,mlp=0.557": 825,
                "instruct_naturalistic_ridge=-0.078,mlp=0.534": 825,
                "pretrained_chat_ridge=-0.461,mlp=0.487": 825,
                "pretrained_naturalistic_ridge=-0.390,mlp=0.499": 825,
            },
        },
    )
    print("wrote", written["png"])


# ---------------------------------------------------------------------------
# Figure 2 -- transfer_specificity.png
# ---------------------------------------------------------------------------
def figure2() -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    # horizontal bars: fraction of within-regime ceiling
    rows = [
        ("chat -> novels", 0.05, 931),
        ("novels -> chat", 0.048, 931),
        ("separator-control -> chat (pretrained)", 0.057, 825),
        ("separator-control -> chat (instruct)", 0.109, 825),
    ]
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    ypos = list(range(len(rows)))
    ax.barh(ypos, vals, color=PRIMARY, height=0.55, zorder=3)
    ax.axvline(
        0.5,
        color=NEUTRAL,
        linestyle="--",
        linewidth=1.2,
        label="same-map bar (0.5, pre-registered)",
        zorder=2,
    )
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("cross-regime transfer (fraction of within-regime ceiling)")
    ax.set_xlim(0.0, 0.6)
    ax.legend(loc="lower right", fontsize=8)
    set_title_subtitle(
        ax,
        "Cross-regime transfer stays far below the same-map bar",
        subtitle="fraction of within-regime ceiling; #931 / #825 round 6",
    )
    written = savefig_paper(fig, "transfer_specificity", dir=OUT_DIR)
    plt.close(fig)
    _augment_meta(
        "transfer_specificity",
        {
            "chat_to_novels=0.05": 931,
            "novels_to_chat=0.048": 931,
            "separator_control_to_chat_pretrained=0.057": 825,
            "separator_control_to_chat_instruct=0.109": 825,
            "same_map_reference_line=0.5": "pre-registered",
        },
    )
    print("wrote", written["png"])


if __name__ == "__main__":
    figure1()
    figure2()
