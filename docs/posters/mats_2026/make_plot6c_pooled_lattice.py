"""MATS 2026 poster, section 6 ("Is it a persona mapping?"): the #2054 replacement.

Supersedes make_plot6_persona.py's plot6b for the poster's section-6 slot. Same
figure grammar (pooled map vs own map, held-out R^2, two bars per group), but
every group is drawn from ONE #2054 pooled lattice, so all six are directly
comparable and the plot6b "*different pool + grain" asterisk disappears.

SIX GROUPS, on-policy condition, attributed-quote boundary for every story cell,
sorted by closeness to the assistant chat-template map (see below):
  assistant chat template  `<|im_start|>user\\n<Q><|im_end|>\\n
                           <|im_start|>assistant\\n<A><|im_end|>`.
  assistant in story       the assistant identity inside a narrative scaffold,
                           boundary `Assistant replied: "<A>"`.
  HELIOS / Wren / Dana / Vex   four fictional characters in the SAME story
                           frame, boundary `<Name> replied: "<A>"`.
(The assistant bare-text framing was dropped from the figure per user directive
2026-08-21; its numbers stay in the sidecar's `dropped_groups` for the record.)

TWO BARS per group:
  light  pooled map (M0)  — ONE ridge map fit jointly on all 56 #2054 cells
                            (both models, every framing/identity/condition),
                            scored on this cell's held-out folds.
  dark   own map          — this cell's banked within-cell ceiling (the same
                            estimator fit on this cell alone).

X ORDER + the printed closeness number: the DIRECT (rung-1) transfer of the
assistant chat-template context->answer map into that cell — "how well does the
assistant's own map already work here, with no refit". Read from the #2054
9-rung ladder via eval_results/issue_2054/chat_closeness_ladder.json.

CONDITION MISMATCH, disclosed on the x-axis label and in the sidecar (user
directive 2026-08-21, having seen the alternative): the BARS are on-policy, but
the CLOSENESS is measured on the INSERTED cells — the ladder's chat anchor is
inserted-only by construction, so an on-policy chat source has no transfer pair.
The ordering axis and the bars therefore come from different conditions.

TWO FIGURES, one per model (user directive 2026-08-21), on a SHARED y-limit and
a SHARED x order (the instruct closeness ranking) so the panels read side by
side. Each panel prints its OWN model's closeness values, so the base panel's
numbers are deliberately non-monotone: base ranks HELIOS above assistant-in-
story, the one order difference between the models.

Numbers read ONLY from committed
  eval_results/issue_2054/pool_specialize/digest.json     (bars)
  eval_results/issue_2054/chat_closeness_ladder.json      (x order + closeness)
Never hand-typed.

Writes docs/posters/mats_2026/figures/plot6c_pooled_lattice_{instruct,base}
.{png,pdf,meta.json} + plot6c_pooled_lattice_data.json.
"""

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_color,
    savefig_paper,
    set_paper_style,
)

REPO = Path(__file__).resolve().parents[3]
DIGEST = REPO / "eval_results" / "issue_2054" / "pool_specialize" / "digest.json"
CLOSENESS = REPO / "eval_results" / "issue_2054" / "chat_closeness_ladder.json"
OUT_DIR = REPO / "docs" / "posters" / "mats_2026" / "figures"
SRC_BARS = "eval_results/issue_2054/pool_specialize/digest.json"
SRC_CLOSE = "eval_results/issue_2054/chat_closeness_ladder.json"

FIGSIZE = (9.4, 3.7)
CONDITION = "on_policy"
ORDER_MODEL = "instruct"  # the shared x order both panels use

MODEL_SUFFIX = {"instruct": "qwen2.5-7b-instruct", "base": "qwen2.5-7b"}
MODEL_TITLE = {"instruct": "Qwen-2.5-7B-Instruct", "base": "Qwen-2.5-7B (base)"}

A = "conversation_paired_stories_assistant"
# key -> (variant, form, who-line, frame-line, example-line). `key` matches the
# closeness JSON's labels; the chat cell is the transfer SOURCE, not a target.
GROUPS = {
    "assistant chat template": (A, "chat", "assistant", "chat template", "<|im_start|>assistant"),
    "assistant in story": (A, "attrib_quoted", "assistant", "in story", 'Assistant replied: "…"'),
    "HELIOS": ("char_helios", "attrib_quoted", "HELIOS", "in story", 'HELIOS replied: "…"'),
    "Wren": ("char_wren", "attrib_quoted", "Wren", "in story", 'Wren replied: "…"'),
    "Dana": ("char_dana", "attrib_quoted", "Dana", "in story", 'Dana replied: "…"'),
    "Vex": ("char_vex", "attrib_quoted", "Vex", "in story", 'Vex replied: "…"'),
}
SOURCE_KEY = "assistant chat template"
DROPPED = ["assistant bare text"]  # user directive 2026-08-21; kept in the sidecar


def x_order(closeness: dict) -> list[str]:
    """Group keys, transfer SOURCE first, then targets by descending closeness."""
    rows = closeness["per_model"][MODEL_SUFFIX[ORDER_MODEL]]
    targets = [k for k in GROUPS if k != SOURCE_KEY]
    return [SOURCE_KEY, *sorted(targets, key=lambda k: -rows[k]["direct_transfer_r2"])]


def load_cells(model: str, order: list[str], closeness: dict) -> list[dict]:
    """Per-group bars + that model's own closeness value, in `order`."""
    by_cell = {
        c["cell"]: c for c in json.loads(DIGEST.read_text())["per_cell"] if c["arm"] == "context"
    }
    close_rows = closeness["per_model"][MODEL_SUFFIX[model]]
    out = []
    for key in order:
        variant, form, who, frame, example = GROUPS[key]
        cell = f"{variant}__{CONDITION}__{form}__{MODEL_SUFFIX[model]}"
        if cell not in by_cell:
            raise KeyError(f"cell {cell!r} absent from {SRC_BARS}")
        c = by_cell[cell]
        close = close_rows[key].get("direct_transfer_r2")  # None for the source
        out.append(
            {
                "group": key,
                "cell": cell,
                "who": who,
                "frame": frame,
                "example": example,
                "is_transfer_source": key == SOURCE_KEY,
                "chat_map_direct_transfer_r2": close,
                "chat_map_reparam9_ratio_of_ceiling": close_rows[key].get(
                    "reparam9_ratio_of_ceiling"
                ),
                "pooled_m0_r2": c["r2_pooled_m0"],
                "own_map_ceiling_r2": c["ceiling_r2"],
                "pooled_bias_m1_r2": c.get("r2_m1_bias"),
                "frac_pooled_of_own": c["fraction_of_ceiling"]["m0"],
                "frac_pooled_bias_of_own": c["fraction_of_ceiling"]["m1"],
                "banked_null_r2_p95": c.get("banked_null_r2_pooled_p95"),
            }
        )
    return out


def plot_model(model: str, cells: list[dict], ylim: tuple[float, float]) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    xs = np.arange(len(cells))
    w = 0.34
    strong = paper_color("instruct" if model == "instruct" else "base")
    light = tuple(0.35 * ch + 0.65 for ch in mcolors.to_rgb(strong))

    ax.bar(
        xs - w / 2,
        [c["pooled_m0_r2"] for c in cells],
        width=w,
        color=light,
        label="one map, all characters/framings pooled",
    )
    ax.bar(
        xs + w / 2,
        [c["own_map_ceiling_r2"] for c in cells],
        width=w,
        color=strong,
        label="that character/framing's own map",
    )
    ax.axhline(0.0, color="0.6", lw=0.6, ls=":", zorder=1)

    # Tick label carries who/frame; the verbatim boundary form and the closeness
    # value ride separate lines so each can take its own face and size.
    ax.set_xticks(xs, [f"{c['who']}\n{c['frame']}" for c in cells])
    for x, c in zip(xs, cells, strict=True):
        ax.annotate(
            c["example"],
            xy=(x, 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, -34),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=6.0,
            family="monospace",
            color="0.35",
        )
        close = (
            "(source)" if c["is_transfer_source"] else f"{c['chat_map_direct_transfer_r2']:+.2f}"
        )
        ax.annotate(
            close,
            xy=(x, 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, -48),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=7.5,
            color="0.15",
        )

    ax.set_ylabel("held-out $R^2$")
    # Floor below zero on purpose: the base chat-template cell's pooled bar is
    # NEGATIVE (-0.016), and a 0.0 floor renders it as no bar at all — reading
    # as missing data rather than as a map that fails on that cell.
    ax.set_ylim(*ylim)
    ax.set_xlabel(
        "closeness to the assistant chat-template map: direct transfer $R^2$ "
        "(measured on the inserted cells)",
        labelpad=30,
    )
    ax.set_title(f"Pooled vs specific fit, evaluated per character/framing — {MODEL_TITLE[model]}")
    ax.legend(frameon=False, loc="upper left", ncols=1, handlelength=1.4)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.34)
    savefig_paper(fig, f"plot6c_pooled_lattice_{model}", dir=OUT_DIR)
    plt.close(fig)
    print(f"WROTE {OUT_DIR / f'plot6c_pooled_lattice_{model}.png'}")


def main() -> None:
    set_paper_style("iclr", font_scale=1.5)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    closeness = json.loads(CLOSENESS.read_text())
    order = x_order(closeness)
    per_model = {m: load_cells(m, order, closeness) for m in ("instruct", "base")}

    vals = [
        v
        for cells in per_model.values()
        for c in cells
        for v in (c["pooled_m0_r2"], c["own_map_ceiling_r2"])
    ]
    ylim = (
        float(np.floor(min(0.0, min(vals) - 0.02) * 20) / 20),
        float(np.ceil((max(vals) * 1.28) * 20) / 20),  # headroom for the legend
    )

    for model in ("instruct", "base"):
        plot_model(model, per_model[model], ylim)

    base_own_order = [
        c["group"]
        for c in sorted(
            per_model["base"],
            key=lambda c: (not c["is_transfer_source"], -(c["chat_map_direct_transfer_r2"] or 0)),
        )
    ]
    (OUT_DIR / "plot6c_pooled_lattice_data.json").write_text(
        json.dumps(
            {
                "sources": [SRC_BARS, SRC_CLOSE],
                "issue": 2054,
                "layer": 19,
                "arm": "context",
                "bars_condition": CONDITION,
                "story_boundary_form": "attrib_quoted",
                "shared_ylim": list(ylim),
                "x_order": order,
                "x_order_derived_from": f"{ORDER_MODEL} direct_transfer_r2, source first",
                "base_own_order_differs": base_own_order,
                "fit": "GCV ridge (dof cap 0.9), shared conversation-grouped 5-fold CV, seed 137",
                "pooled_map_def": "ONE map fit jointly on all 56 #2054 cells (both models, "
                "every framing x identity x condition), scored on each cell's held-out folds",
                "own_map_def": "that cell's banked within-cell ceiling (same estimator, "
                "fit on that cell alone)",
                "closeness_def": "direct (rung-1) transfer R^2 of the assistant chat-template "
                "map into that cell, pooled fold-mean; the chat cell is the source itself",
                "bars_drawn": ["pooled_m0_r2", "own_map_ceiling_r2"],
                "caveats": {
                    "condition_mismatch": "BARS are on-policy; CLOSENESS is measured on the "
                    "INSERTED cells, because the #2054 ladder's chat anchor is inserted-only "
                    "by construction (an on-policy chat source has no transfer pair). Chosen "
                    "by user directive 2026-08-21 over switching the bars to inserted; "
                    "disclosed on the x-axis label.",
                    "bare_label_form_excluded": "the bare-label story boundary carries a "
                    "trailing-space tokenization artifact (23.2% digit-start onsets); the "
                    "attributed-quote form is drawn instead",
                },
                "dropped_groups": {
                    g: "removed from the figure by user directive 2026-08-21" for g in DROPPED
                },
                "cells": per_model,
            },
            indent=1,
        )
    )
    print(f"WROTE {OUT_DIR / 'plot6c_pooled_lattice_data.json'}")


if __name__ == "__main__":
    main()
