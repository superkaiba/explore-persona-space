"""MATS 2026 poster, section 6 ("Is it a persona mapping?"): the #2054 replacement.

Supersedes make_plot6_persona.py's plot6b for the poster's section-6 slot. Same
figure grammar (one shared map vs each one's own map, held-out R^2), but every
group is drawn
from ONE #2054 pooled lattice, so all six are directly comparable and the
plot6b "*different pool + grain" asterisk disappears.

ONE PANEL, both models as OVERLAPPING bars (user directive 2026-08-21). Only
the own-map slot overlaps: instruct wide and behind, base narrow and in front.
Base sits below instruct on every drawn own-map value, so the narrow bar always
nests visibly inside the wide one — `assert_base_nests_inside_instruct` pins
that precondition and fails loud if a future data refresh breaks it (the
overlap would hide a bar).

SIX GROUPS, on-policy condition, attributed-quote boundary for every story cell:
  assistant chat template  `<|im_start|>user\\n<Q><|im_end|>\\n
                           <|im_start|>assistant\\n<A><|im_end|>`.
  assistant in story       the assistant identity inside a narrative scaffold,
                           boundary `Assistant replied: "<A>"`.
  HELIOS / Wren / Dana / Vex   four fictional characters in the SAME story
                           frame, boundary `<Name> replied: "<A>"`. The x-label
                           descriptions condense `issue1310_common.PERSONAS`
                           verbatim (see SHORT_DESC).

BARS per group — three, not four: instruct contributes both, base only its own
map (its shared-map bar was dropped by user directive 2026-08-21; the values
stay in the sidecar).
  light  one shared map   — ONE ridge map fit jointly on all 56 #2054 cells
                            (both models, every framing/identity/condition),
                            scored on this cell's held-out folds. Displayed as
                            "one shared map" rather than "pooled" (vaguer, user
                            2026-08-21); it matches #1310's own name for this
                            leg, the SHARED-vs-SPECIFIC decomposition. The
                            sidecar KEYS keep `pooled_m0` so they still trace
                            to the digest field `r2_pooled_m0`.
  strong its own map      — this cell's banked within-cell ceiling (the same
                            estimator fit on this cell alone).

X ORDER: descending direct (rung-1) transfer of the assistant chat-template map
into each cell — how well the assistant's own map already works there with no
refit — read from eval_results/issue_2054/chat_closeness_ladder.json, instruct
ranking, source first. The per-group transfer NUMBERS were removed from the
canvas (user directive 2026-08-21); they survive in the sidecar, and the axis
label states only that the x order is a similarity ranking.

That ordering statistic is measured on the INSERTED cells while the bars are
ON-POLICY: the #2054 ladder's chat anchor is inserted-only by construction, so
an on-policy chat source has no transfer pair. Recorded in the sidecar's
`caveats.condition_mismatch`; no longer visible on the figure now that the
numbers are gone.

Numbers read ONLY from committed
  eval_results/issue_2054/pool_specialize/digest.json     (bars)
  eval_results/issue_2054/chat_closeness_ladder.json      (x order)
Never hand-typed.

Writes docs/posters/mats_2026/figures/plot6c_pooled_lattice.{png,pdf,meta.json}
+ plot6c_pooled_lattice_data.json. Supersedes the per-model
plot6c_pooled_lattice_{instruct,base}.* pair, deleted in the same change.
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

# Authored at the poster COLUMN's physical width (0.188 x ~34.6 in textwidth,
# placed at 0.99\linewidth) so beamerposter scales it ~1.0 and every in-figure
# point size survives to the printed poster. Authoring wider and letting LaTeX
# shrink it is what put the label lines at ~4 pt on a 36x24 in poster (measured
# 2026-08-21 off the built PDF); horizontal room per group, not vertical, is the
# binding constraint, hence the two-line descriptions below.
FIGSIZE = (8.0, 3.4)
CONDITION = "on_policy"
ORDER_MODEL = "instruct"

MODEL_SUFFIX = {"instruct": "qwen2.5-7b-instruct", "base": "qwen2.5-7b"}

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
# Base contributes ONLY its own-map bar (user directive 2026-08-21). Its
# shared-map values stay in the sidecar under `pooled_m0_r2`; dropping them
# from the canvas also removes the only negative quantity the figure drew
# (base chat template, -0.016), so the y floor returns to zero.
DRAWN_BASE_FIELDS = ("own_map_ceiling_r2",)
DROPPED = ["assistant bare text"]  # user directive 2026-08-21; kept in the sidecar

# Condensed from issue1310_common.PERSONAS (verbatim source strings recorded in
# the sidecar under `character_descriptions_verbatim`).
SHORT_DESC = {
    "assistant chat template": "the assistant\nitself",
    "assistant in story": "same assistant,\nas a character",
    "HELIOS": "a calm,\nprecise AI",
    "Wren": "warm, endlessly\nhelpful",
    "Dana": "an ordinary,\nunremarkable person",
    "Vex": "a theatrical,\nscheming villain",
}
VERBATIM_DESC = {
    "HELIOS": "a calm, precise artificial intelligence",
    "Wren": "a warm, endlessly helpful assistant who patiently helps anyone who asks",
    "Dana": "an ordinary, unremarkable everyday person",
    "Vex": "a theatrical, scheming villain who delights in menace",
}


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
        out.append(
            {
                "group": key,
                "cell": cell,
                "who": who,
                "frame": frame,
                "short_description": SHORT_DESC[key],
                "example": example,
                "is_transfer_source": key == SOURCE_KEY,
                "chat_map_direct_transfer_r2": close_rows[key].get("direct_transfer_r2"),
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


def assert_base_nests_inside_instruct(per_model: dict[str, list[dict]]) -> None:
    """The overlap only reads if the narrow (base) bar is the shorter one.

    Drawn base-in-front-of-instruct, a base value ABOVE its instruct twin would
    hide the instruct bar behind it. Fail loud rather than ship a figure whose
    occlusion silently drops a series.
    """
    for bi, bb in zip(per_model["instruct"], per_model["base"], strict=True):
        assert bi["group"] == bb["group"], (bi["group"], bb["group"])
        # Only the own-map slot overlaps now: base's shared-map bar is not drawn
        # (user directive 2026-08-21), so its value cannot occlude anything.
        for field in DRAWN_BASE_FIELDS:
            if bb[field] > bi[field]:
                raise AssertionError(
                    f"base {field} ({bb[field]:.4f}) exceeds instruct ({bi[field]:.4f}) for "
                    f"{bi['group']!r} — the narrow base bar would occlude the wide instruct "
                    f"bar; re-draw side by side instead of overlapping"
                )


def plot_combined(per_model: dict[str, list[dict]], ylim: tuple[float, float]) -> None:
    cells = per_model["instruct"]  # label source; both share the x order
    fig, ax = plt.subplots(figsize=FIGSIZE)
    xs = np.arange(len(cells))
    w_wide, w_narrow = 0.38, 0.19

    def shades(concept: str) -> tuple[tuple, str]:
        strong = paper_color(concept)
        light = tuple(0.35 * ch + 0.65 for ch in mcolors.to_rgb(strong))
        return light, strong

    i_light, i_strong = shades("instruct")
    _, b_strong = shades("base")

    # Wide instruct bars first, narrow base bars over them.
    ax.bar(
        xs - w_wide / 2,
        [c["pooled_m0_r2"] for c in per_model["instruct"]],
        width=w_wide,
        color=i_light,
        label="instruct — one shared map",
    )
    ax.bar(
        xs + w_wide / 2,
        [c["own_map_ceiling_r2"] for c in per_model["instruct"]],
        width=w_wide,
        color=i_strong,
        label="instruct — its own map",
    )
    ax.bar(
        xs + w_wide / 2,
        [c["own_map_ceiling_r2"] for c in per_model["base"]],
        width=w_narrow,
        color=b_strong,
        edgecolor="white",
        linewidth=0.4,
        label="base — its own map",
    )
    ax.axhline(0.0, color="0.6", lw=0.6, ls=":", zorder=1)

    # Tick label carries who/frame; the short description rides its own line so
    # it can take a smaller face. The verbatim boundary form is NOT drawn: at
    # this column width six of them collide, and shrinking them to fit puts them
    # near 4 pt on a 36x24 in poster — unreadable either way. They live in the
    # sidecar (`example`) and in the poster caption instead.
    ax.set_xticks(xs, [f"{c['who']}\n{c['frame']}" for c in cells])
    for x, c in zip(xs, cells, strict=True):
        ax.annotate(
            c["short_description"],
            xy=(x, 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, -34),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8.0,
            color="0.30",
        )

    ax.set_ylabel("held-out $R^2$")
    ax.set_ylim(*ylim)
    ax.set_xlabel("sorted by similarity to assistant in chat template", labelpad=30)
    ax.set_title("One shared map vs its own map, per character/framing — Qwen-2.5-7B")
    ax.legend(
        frameon=False,
        loc="upper right",
        ncols=2,
        handlelength=1.0,
        columnspacing=0.8,
        fontsize=7.0,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.32)
    savefig_paper(fig, "plot6c_pooled_lattice", dir=OUT_DIR)
    plt.close(fig)
    print(f"WROTE {OUT_DIR / 'plot6c_pooled_lattice.png'}")


def main() -> None:
    set_paper_style("iclr", font_scale=1.9)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    closeness = json.loads(CLOSENESS.read_text())
    order = x_order(closeness)
    per_model = {m: load_cells(m, order, closeness) for m in ("instruct", "base")}
    assert_base_nests_inside_instruct(per_model)

    # Only DRAWN quantities set the limits — base's undrawn shared-map values
    # include the lone negative, and letting it stretch the axis would leave
    # dead space under bars nobody can see.
    vals = [c["pooled_m0_r2"] for c in per_model["instruct"]]
    vals += [c["own_map_ceiling_r2"] for c in per_model["instruct"]]
    vals += [c[f] for c in per_model["base"] for f in DRAWN_BASE_FIELDS]
    ylim = (
        float(np.floor(min(0.0, min(vals) - 0.02) * 20) / 20),
        float(np.ceil((max(vals) * 1.30) * 20) / 20),  # headroom for the legend
    )
    plot_combined(per_model, ylim)

    (OUT_DIR / "plot6c_pooled_lattice_data.json").write_text(
        json.dumps(
            {
                "sources": [SRC_BARS, SRC_CLOSE],
                "issue": 2054,
                "layer": 19,
                "arm": "context",
                "bars_condition": CONDITION,
                "story_boundary_form": "attrib_quoted",
                "panel": "single panel, both models as overlapping bars "
                "(instruct wide behind, base narrow in front)",
                "ylim": list(ylim),
                "x_order": order,
                "x_order_derived_from": "instruct direct_transfer_r2 of the assistant "
                "chat-template map into each cell, descending; transfer source first",
                "x_order_values_not_drawn": "the per-group transfer numbers were removed "
                "from the canvas by user directive 2026-08-21; they remain per cell below",
                "fit": "GCV ridge (dof cap 0.9), shared conversation-grouped 5-fold CV, seed 137",
                "pooled_map_def": "ONE map fit jointly on all 56 #2054 cells (both models, "
                "every framing x identity x condition), scored on each cell's held-out folds",
                "own_map_def": "that cell's banked within-cell ceiling (same estimator, "
                "fit on that cell alone)",
                "character_descriptions_verbatim": VERBATIM_DESC,
                "character_descriptions_source": "issue1310_common.PERSONAS (condensed for "
                "the axis; verbatim strings above)",
                "bars_drawn": {
                    "instruct": ["pooled_m0_r2", "own_map_ceiling_r2"],
                    "base": ["own_map_ceiling_r2"],
                },
                "base_shared_map_not_drawn": "base's pooled_m0_r2 is retained per cell "
                "below but removed from the canvas by user directive 2026-08-21",
                "caveats": {
                    "condition_mismatch": "BARS are on-policy; the X ORDER is derived from a "
                    "transfer statistic measured on the INSERTED cells, because the #2054 "
                    "ladder's chat anchor is inserted-only by construction (an on-policy chat "
                    "source has no transfer pair). The numbers are no longer on the canvas, so "
                    "this caveat lives only here and in the module docstring.",
                    "bare_label_form_excluded": "the bare-label story boundary carries a "
                    "trailing-space tokenization artifact (23.2% digit-start onsets); the "
                    "attributed-quote form is drawn instead",
                    "overlap_precondition": "base < instruct on every DRAWN overlapping "
                    "quantity (the own-map slot); asserted at render time by "
                    "assert_base_nests_inside_instruct",
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
