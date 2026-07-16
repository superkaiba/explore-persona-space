#!/usr/bin/env python
"""Rebuild the #1345 figures, organized to map cleanly onto the two results.

All figures use the CONTEXT arm at frozen layer 19 (`headline_layer`), both
models where available, read from the LOCAL eval_results/issue_1345 JSONs.
Framings: r1 = chat, r2 = no-template, r3 = story (ARIA).

RESULT 1 — "the assistant map is the same with vs without the chat template,
            up to a linear change of coordinates":
  result1_reparam_recovery.png
      Grouped bars, chat<->no-template only, per (model x direction): the
      target framing's own CEILING (dashed marker) vs naive TRANSFER R^2 vs
      general-linear REPARAM-RECOVERED R^2 vs matched-capacity NULL. Recovered
      reaches ceiling; naive transfer falls short; null ~= -0.03.
  result1_operator_cosine.png
      Raw vs rotation-aligned operator cosine (chat vs no-template), base +
      instruct, with the rotation-null (~0) line. Raw 0.29/0.65 -> aligned
      0.73/0.85: instruction tuning pulls the coordinate systems together.

RESULT 2 — "the assistant map is NOT the same when the assistant is a story
            character":
  result2_withinregime_by_framing.png
      Within-regime held-out R^2 by framing (Chat / No-template / Story.ARIA),
      grouped by model, layer 19, with bootstrap CIs. Chat/no-template ~= 0.6
      (well above zero); story ~= -0.75 (below zero AND below the answer-mean
      baseline). Answer-mean-baseline (R^2=0) and shuffle-null reference lines
      drawn. Base story = N/A (not tested; 96/500 stories below floor).
  result2_transfer_heatmap.png
      Two 3x3 heatmaps (base | instruct), rows = SOURCE, cols = TARGET framing,
      cell = layer-19 transfer R^2, diagonal = within-regime R^2. Colour
      clipped to [-1, 0.7] so the chat<->no-template block shows real gradation
      while catastrophic story cells saturate; every cell annotated with its
      true value.

Reproduce:
    uv run python scripts/issue1345_clear_figs.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

from explore_persona_space.analysis.paper_plots import (
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)

RESULTS = Path("eval_results/issue_1345")
FIG_DIR = "figures/"  # savefig_paper prepends this; stems carry the issue_1345/ prefix
LAYER = "19"
LAYER_IDX = 19

FRAMINGS = ["Chat", "No-template", "Story·ARIA"]  # r1, r2, r3
PAIR_TO_IDX = {"r1": 0, "r2": 1, "r3": 2}

# Diverging colour-scale clip (the clarity fix): story cells saturate here.
CLIP_LO, CLIP_HI = -1.0, 0.7


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


# --------------------------------------------------------------------------- #
# RESULT 2 — transfer heatmap                                                  #
# --------------------------------------------------------------------------- #
def build_transfer_matrix(arm: str) -> np.ndarray:
    """3x3 layer-19 transfer R^2, rows=SOURCE, cols=TARGET; NaN = untested.

    Off-diagonal[src, tgt] = matrix['<src>-><tgt>'].transfer_r2_by_layer[19].
    Diagonal[f, f]         = within-regime R^2 = target_within of a pair INTO f.
    """
    d = _load(f"cross_regime_transfer_{arm}_context.json")
    m = d["matrix"]
    mat = np.full((3, 3), np.nan, dtype=float)
    for key, pair in m.items():
        s, t = key.split("->")
        mat[PAIR_TO_IDX[s], PAIR_TO_IDX[t]] = pair["transfer_r2_by_layer"][LAYER]
    for key, pair in m.items():
        _, t = key.split("->")
        ti = PAIR_TO_IDX[t]
        if np.isnan(mat[ti, ti]):
            mat[ti, ti] = pair["target_within_r2_by_layer"][LAYER]
    return mat


def _annotate_cell(ax, j: int, i: int, val: float, clipped: float) -> None:
    if np.isnan(val):
        ax.text(j, i, "N/A", ha="center", va="center", fontsize=9, color="#6A6A6A", style="italic")
        return
    txt_color = "white" if (clipped <= -0.55 or clipped >= 0.55) else "#1A1A1A"
    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9.5, color=txt_color)


def plot_result2_transfer_heatmap() -> dict:
    mats = {"base": build_transfer_matrix("base"), "instruct": build_transfer_matrix("instruct")}

    norm = TwoSlopeNorm(vmin=CLIP_LO, vcenter=0.0, vmax=CLIP_HI)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#E8E8E8")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.4))
    im = None
    for ax, arm in zip(axes, ["base", "instruct"], strict=True):
        mat = mats[arm]
        clipped = np.clip(mat, CLIP_LO, CLIP_HI)
        im = ax.imshow(np.ma.masked_invalid(clipped), cmap=cmap, norm=norm, aspect="equal")
        for i in range(3):
            for j in range(3):
                _annotate_cell(ax, j, i, mat[i, j], clipped[i, j])
                if i == j:
                    ax.add_patch(
                        Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#111111", lw=2.2)
                    )
        ax.set_xticks(range(3), FRAMINGS, fontsize=10)
        ax.set_yticks(range(3), FRAMINGS, fontsize=10)
        ax.set_xlabel("Target framing", fontsize=10.5)
        if arm == "base":
            ax.set_ylabel("Source framing", fontsize=10.5)
        ax.set_title(f"{arm.capitalize()} model", fontsize=12, pad=8)
        ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
        ax.grid(which="minor", color="white", lw=1.5)
        ax.tick_params(which="both", length=0)

    cbar = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.03, extend="both")
    cbar.set_label("Layer-19 transfer $R^2$  (colour clipped to [-1, 0.7])", fontsize=10)

    fig.suptitle(
        "Persona map transfers across chat / no-template, but not story (context arm, layer 19)",
        fontsize=13.5,
        y=1.03,
        x=0.5,
        ha="center",
    )
    fig.text(
        0.5,
        -0.05,
        "Rows = source framing, cols = target framing; diagonal (boxed) = "
        "within-regime $R^2$. Colour clipped to [-1, 0.7]; catastrophic story "
        "cells saturate but are annotated with their true value. "
        "Chat<->no-template retains 83-84% of ceiling (instruct); story "
        "transfer is deeply negative. Base story = N/A (not tested).",
        ha="center",
        va="top",
        fontsize=8.6,
        color="#5A5A5A",
        wrap=True,
    )
    paths = savefig_paper(fig, "issue_1345/result2_transfer_heatmap", dir=FIG_DIR)
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------- #
# RESULT 2 — within-regime held-out R^2 by framing                            #
# --------------------------------------------------------------------------- #
# (model, framing) -> cells_R file. Base has no story (r3).
CELLS = {
    ("base", 0): "cells_R_base_r1_context.json",
    ("base", 1): "cells_R_base_r2_context.json",
    ("instruct", 0): "cells_R_instruct_r1_context.json",
    ("instruct", 1): "cells_R_instruct_r2_context.json",
    ("instruct", 2): "cells_R_instruct_r3_context.json",
}
NULLS = {k: v.replace("cells_R", "nulls_R") for k, v in CELLS.items()}


def collect_withinregime() -> dict:
    """Per (model, framing): within-regime R^2 + bootstrap CI + shuffle-null."""
    out: dict = {}
    for (model, fi), fname in CELLS.items():
        ci = _load(fname)["r2_bootstrap_ci_frozen_layers"][LAYER]
        nm = np.array(_load(NULLS[(model, fi)])["null_matrix"])  # draws x layers
        col = nm[:, LAYER_IDX]
        out[(model, fi)] = {
            "r2": ci["r2"],
            "lo": ci["ci_lo"],
            "hi": ci["ci_hi"],
            "null_mean": float(col.mean()),
        }
    return out


def plot_result2_withinregime() -> dict:
    d = collect_withinregime()
    c_base = paper_palette_role("baseline")
    c_inst = paper_palette_role("primary")
    model_color = {"base": c_base, "instruct": c_inst}

    fig, ax = plt.subplots(figsize=(9.0, 5.3))
    x = np.arange(3)
    bw = 0.36
    off = {"base": -bw / 2, "instruct": bw / 2}

    for model in ["base", "instruct"]:
        for fi in range(3):
            key = (model, fi)
            xpos = x[fi] + off[model]
            if key not in d:
                ax.text(
                    xpos,
                    0.03,
                    "N/A",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    color="#6A6A6A",
                    style="italic",
                )
                continue
            c = d[key]
            err = np.array([[c["r2"] - c["lo"]], [c["hi"] - c["r2"]]])
            ax.bar(
                xpos,
                c["r2"],
                bw,
                color=model_color[model],
                yerr=err,
                capsize=3,
                ecolor="#333333",
                label=model.capitalize() if fi == 0 else None,
            )
            va = "bottom" if c["r2"] >= 0 else "top"
            dy = 0.02 if c["r2"] >= 0 else -0.02
            ax.text(
                xpos,
                c["r2"] + dy + (c["hi"] - c["r2"] if c["r2"] >= 0 else -(c["r2"] - c["lo"])),
                f"{c['r2']:.2f}",
                ha="center",
                va=va,
                fontsize=8.6,
                color="#1A1A1A",
            )

    # Reference lines.
    ax.axhline(0.0, color="#111111", ls="--", lw=1.6, label="Answer-mean baseline ($R^2=0$)")
    null_nonstory = np.mean([d[k]["null_mean"] for k in d if k[1] != 2])
    ax.axhline(
        null_nonstory,
        color="#7A7A7A",
        ls=":",
        lw=1.4,
        label=f"Shuffle-null ≈ {null_nonstory:.2f} (chat/no-template)",
    )
    # Story's own shuffle-null is far lower (off-scale) — note it honestly, in
    # the empty base-story slot so it never overlaps the instruct-story bar.
    story_null = d[("instruct", 2)]["null_mean"]
    ax.text(
        x[2] - 0.62,
        -0.40,
        f"story shuffle-null ≈ {story_null:.1f}\n(off-scale): story $R^2$ is\n"
        "above its null but below\nthe answer-mean baseline",
        fontsize=7.2,
        color="#5A5A5A",
        ha="center",
        va="center",
    )

    ax.set_xticks(x, FRAMINGS, fontsize=10.5)
    ax.set_ylabel("Within-regime held-out $R^2$ (layer 19)", fontsize=10.5)
    ax.set_ylim(-0.9, 0.78)
    ax.axhline(0.0, color="#111111", ls="--", lw=1.6)  # keep baseline on top
    ax.legend(loc="lower left", fontsize=8.4, frameon=True, framealpha=0.92)
    fig.suptitle(
        "The story-character assistant map is not linearly decodable (context arm, layer 19)",
        fontsize=13,
        y=1.03,
        x=0.5,
        ha="center",
    )
    fig.text(
        0.5,
        -0.03,
        "Bars = within-regime held-out $R^2$ with 95% bootstrap CI. "
        "Chat and no-template sit ~0.6 (well above the answer-mean baseline); "
        "story sits at -0.75, below zero — the held-out probe does worse than "
        "predicting the mean. Base story = N/A (not tested).",
        ha="center",
        va="top",
        fontsize=8.6,
        color="#5A5A5A",
        wrap=True,
    )
    paths = savefig_paper(fig, "issue_1345/result2_withinregime_by_framing", dir=FIG_DIR)
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------- #
# RESULT 1 — reparameterization recovery                                       #
# --------------------------------------------------------------------------- #
# direction_key (both files): b2i = "reparam r2(no-template) recovered in
# r1(chat)" -> target = chat -> the no-template -> chat direction.  i2b =
# "reparam r1(chat) recovered in r2(no-template)" -> target = no-template ->
# the chat -> no-template direction.
DIR_LABEL = {"i2b": "Chat → No-template", "b2i": "No-template → Chat"}
DIR_TRANSFER_PAIR = {"i2b": "r1->r2", "b2i": "r2->r1"}
DIR_ORDER = ["i2b", "b2i"]


def collect_recovery(arm: str) -> dict:
    op = _load(f"operator_comparison_{arm}_context.json")
    tr = _load(f"cross_regime_transfer_{arm}_context.json")
    dl = op["delta_reparam_l19"]
    nulls = op["reparam_r1r2"][LAYER]["matched_capacity_nulls"]
    tmat = tr["matrix"]
    out: dict = {}
    for dkey in DIR_ORDER:
        out[dkey] = {
            "ceiling": dl["within_r2"][dkey],
            "recovered": dl["recovered_r2"][dkey],
            "transfer": tmat[DIR_TRANSFER_PAIR[dkey]]["transfer_r2_by_layer"][LAYER],
            "null": nulls[dkey]["null_recovery_r2"],
        }
    ap = op["reparam_r1r2"][LAYER]["activation_procrustes"]
    out["_cosine"] = {
        "raw": ap["raw_vec_cosine"],
        "aligned": ap["observed_aligned_cosine"],
        "null_p975": ap["null_p975"],
    }
    return out


def _bar_labels(ax, bars, fmt="{:.2f}", dy=0.012):
    for b in bars:
        h = b.get_height()
        va = "bottom" if h >= 0 else "top"
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + (dy if h >= 0 else -dy),
            fmt.format(h),
            ha="center",
            va=va,
            fontsize=8.2,
            color="#1A1A1A",
        )


def plot_result1_reparam_recovery() -> dict:
    data = {arm: collect_recovery(arm) for arm in ["base", "instruct"]}
    c_transfer = paper_palette_role("baseline")
    c_recover = paper_palette_role("primary")
    c_null = paper_palette_role("neutral")

    fig, (ax_b, ax_i) = plt.subplots(1, 2, figsize=(10.5, 5.0), sharey=True)
    bw = 0.26
    group_x = np.array([0.0, 1.15])
    offsets = {"transfer": -bw, "recover": 0.0, "null": bw}

    for ax, arm in zip([ax_b, ax_i], ["base", "instruct"], strict=True):
        d = data[arm]
        for gi, dkey in enumerate(DIR_ORDER):
            cell = d[dkey]
            x0 = group_x[gi]
            bt = ax.bar(
                x0 + offsets["transfer"],
                cell["transfer"],
                bw,
                color=c_transfer,
                label="Naive transfer" if gi == 0 else None,
            )
            br = ax.bar(
                x0 + offsets["recover"],
                cell["recovered"],
                bw,
                color=c_recover,
                label="Reparam-recovered" if gi == 0 else None,
            )
            bn = ax.bar(
                x0 + offsets["null"],
                cell["null"],
                bw,
                color=c_null,
                label="Matched-capacity null" if gi == 0 else None,
            )
            _bar_labels(ax, list(bt) + list(br) + list(bn))
            ceil = cell["ceiling"]
            ax.plot(
                [x0 - bw * 1.6, x0 + bw * 1.6],
                [ceil, ceil],
                ls="--",
                lw=1.8,
                color="#111111",
                label="Within-regime ceiling" if gi == 0 else None,
            )
            ax.text(
                x0 - bw * 1.6,
                ceil + 0.015,
                f"ceiling {ceil:.2f}",
                ha="left",
                va="bottom",
                fontsize=7.4,
                color="#111111",
            )
        ax.axhline(0.0, color="#999999", lw=0.8)
        ax.set_xticks(group_x, [DIR_LABEL[k] for k in DIR_ORDER], fontsize=9.2)
        ax.set_ylim(-0.12, 0.82)
        ax.set_title(f"{arm.capitalize()} model", fontsize=12, pad=8)
        if arm == "base":
            ax.set_ylabel("Layer-19 $R^2$", fontsize=10.5)
    ax_b.legend(loc="upper left", fontsize=8.2, frameon=True, framealpha=0.92)

    fig.suptitle(
        "A general-linear reparameterization recovers chat<->no-template "
        "transfer to the ceiling (context arm, layer 19)",
        fontsize=12.5,
        y=1.02,
        x=0.5,
        ha="center",
    )
    fig.text(
        0.5,
        -0.03,
        "Per model x direction: recovered $R^2$ lands within 0.005-0.008 of "
        "each framing's within-regime ceiling (dashed); naive transfer falls "
        "short and the matched-capacity null sits near -0.03. The two maps "
        "differ only by a linear change of coordinates.",
        ha="center",
        va="top",
        fontsize=8.6,
        color="#5A5A5A",
        wrap=True,
    )
    paths = savefig_paper(fig, "issue_1345/result1_reparam_recovery", dir=FIG_DIR)
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------- #
# RESULT 1 — operator cosine (raw vs aligned)                                  #
# --------------------------------------------------------------------------- #
def plot_result1_operator_cosine() -> dict:
    data = {arm: collect_recovery(arm)["_cosine"] for arm in ["base", "instruct"]}
    c_raw = paper_palette_role("baseline")
    c_aligned = paper_palette_role("accent")

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    x = np.arange(2)  # Base, Instruct
    bw = 0.34
    raw = [data["base"]["raw"], data["instruct"]["raw"]]
    ali = [data["base"]["aligned"], data["instruct"]["aligned"]]
    null_line = max(data["base"]["null_p975"], data["instruct"]["null_p975"])

    br = ax.bar(x - bw / 2, raw, bw, color=c_raw, label="Raw operator cosine")
    ba = ax.bar(x + bw / 2, ali, bw, color=c_aligned, label="Rotation-aligned (Procrustes)")
    _bar_labels(ax, list(br) + list(ba), dy=0.015)

    # raw -> aligned lift arrows
    for i in range(2):
        ax.annotate(
            "",
            xy=(x[i] + bw / 2, ali[i] - 0.02),
            xytext=(x[i] - bw / 2, raw[i] + 0.02),
            arrowprops=dict(arrowstyle="->", color="#555555", lw=1.3),
        )

    ax.axhline(
        null_line,
        color="#7A7A7A",
        ls=":",
        lw=1.4,
        label=f"Rotation null ≈ 0 (p97.5 = {null_line:.3f})",
    )
    ax.set_xticks(x, ["Base", "Instruct"], fontsize=11)
    ax.set_ylabel("Operator cosine (chat vs no-template)", fontsize=10.5)
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="upper center", fontsize=8.6, frameon=True, framealpha=0.92)
    fig.suptitle(
        "Instruction tuning pulls the chat / no-template operators together "
        "(context arm, layer 19)",
        fontsize=12.5,
        y=1.05,
        x=0.5,
        ha="center",
    )
    fig.text(
        0.5,
        -0.03,
        "Cosine between the chat and no-template linear operators, raw vs after "
        "an optimal rotation (Procrustes). Aligning lifts cosine far above the "
        "~0 rotation null; the instruct model starts closer (raw 0.65 vs 0.29) "
        "and aligns higher (0.85 vs 0.73).",
        ha="center",
        va="top",
        fontsize=8.6,
        color="#5A5A5A",
        wrap=True,
    )
    paths = savefig_paper(fig, "issue_1345/result1_operator_cosine", dir=FIG_DIR)
    plt.close(fig)
    return paths


def main() -> None:
    set_paper_style("blog")
    figs = {
        "result1_reparam_recovery": plot_result1_reparam_recovery(),
        "result1_operator_cosine": plot_result1_operator_cosine(),
        "result2_withinregime_by_framing": plot_result2_withinregime(),
        "result2_transfer_heatmap": plot_result2_transfer_heatmap(),
    }
    for tag, paths in figs.items():
        for fmt, path in paths.items():
            print(f"  {tag} [{fmt}] -> {path}  ({Path(path).stat().st_size} bytes)")


if __name__ == "__main__":
    main()
