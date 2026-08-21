"""MATS 2026 poster, section 6 ("Is it a persona mapping?"): the #2054 replacement.

Supersedes make_plot6_persona.py's plot6b for the poster's section-6 slot. Same
figure grammar (pooled map vs own map, held-out R^2, two bars per group), but
every group is drawn from ONE #2054 pooled lattice, so all seven are directly
comparable and the plot6b "*different pool + grain" asterisk disappears.

SEVEN GROUPS (x order fixed by user directive 2026-08-21), on-policy condition,
attributed-quote boundary for every story cell:
  Wren / HELIOS / Dana / Vex   four fictional characters inside a narrative
                               scaffold, boundary `<Name> replied: "<A>"`.
  assistant in story           the assistant identity in the SAME story frame,
                               boundary `Assistant replied: "<A>"`.
  assistant bare text          `User: <Q>\\n\\nAssistant: <A>`, no chat template.
  assistant chat template      `<|im_start|>user\\n<Q><|im_end|>\\n
                               <|im_start|>assistant\\n<A><|im_end|>`.

TWO BARS per group:
  light  pooled map (M0)  — ONE ridge map fit jointly on all 56 #2054 cells
                            (both models, every framing/identity/condition),
                            scored on this cell's held-out folds.
  dark   own map          — this cell's banked within-cell ceiling (the same
                            estimator fit on this cell alone).

TWO FIGURES, one per model (user directive 2026-08-21 "make both base and
instruct plots"), rendered on a SHARED y-limit so the panels are readable
side by side.

Numbers read ONLY from committed
  eval_results/issue_2054/pool_specialize/digest.json
(context arm; `r2_pooled_m0` and `ceiling_r2` per cell). Never hand-typed.

CAVEAT carried in the sidecar, NOT rendered on the canvas (poster figures stay
axes-only): the instruct assistant-bare-text on-policy cell is #2054's 42.5%
cap-hit cell — 42.5% of its generations ran to the token cap — so its low
pooled fraction is partly a truncation artifact, not only a map property.

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
OUT_DIR = REPO / "docs" / "posters" / "mats_2026" / "figures"
SRC = "eval_results/issue_2054/pool_specialize/digest.json"

# Wide enough that the seven three-line x labels never collide (~1.5 in per
# group at the longest example string, monospace 6 pt).
FIGSIZE = (10.6, 3.6)
CONDITION = "on_policy"

MODEL_SUFFIX = {"instruct": "qwen2.5-7b-instruct", "base": "qwen2.5-7b"}
MODEL_TITLE = {
    "instruct": "Qwen-2.5-7B-Instruct",
    "base": "Qwen-2.5-7B (base)",
}

# (cell-variant, form, who-line, frame-line, example-line). Order is the x order.
GROUPS = [
    ("char_wren", "attrib_quoted", "Wren", "in story", 'Wren replied: "…"'),
    ("char_helios", "attrib_quoted", "HELIOS", "in story", 'HELIOS replied: "…"'),
    ("char_dana", "attrib_quoted", "Dana", "in story", 'Dana replied: "…"'),
    ("char_vex", "attrib_quoted", "Vex", "in story", 'Vex replied: "…"'),
    (
        "conversation_paired_stories_assistant",
        "attrib_quoted",
        "assistant",
        "in story",
        'Assistant replied: "…"',
    ),
    (
        "conversation_paired_stories_assistant",
        "bare_text",
        "assistant",
        "bare text",
        "User: … / Assistant: …",
    ),
    (
        "conversation_paired_stories_assistant",
        "chat",
        "assistant",
        "chat template",
        "<|im_start|>assistant",
    ),
]


def load_cells(model: str) -> list[dict]:
    """Per-group pooled/own held-out R^2 for one model, in GROUPS order."""
    digest = json.loads(DIGEST.read_text())
    by_cell = {c["cell"]: c for c in digest["per_cell"] if c["arm"] == "context"}
    out = []
    for variant, form, who, frame, example in GROUPS:
        key = f"{variant}__{CONDITION}__{form}__{MODEL_SUFFIX[model]}"
        if key not in by_cell:
            raise KeyError(f"cell {key!r} absent from {SRC}")
        c = by_cell[key]
        out.append(
            {
                "cell": key,
                "who": who,
                "frame": frame,
                "example": example,
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

    m0 = [c["pooled_m0_r2"] for c in cells]
    own = [c["own_map_ceiling_r2"] for c in cells]

    ax.bar(xs - w / 2, m0, width=w, color=light, label="one map, all cells pooled")
    ax.bar(xs + w / 2, own, width=w, color=strong, label="that cell's own map")
    ax.axhline(0.0, color="0.6", lw=0.6, ls=":", zorder=1)

    # Two-line tick label (who / frame); the verbatim example rides a third
    # line placed separately so it can carry a smaller monospace face.
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

    ax.set_ylabel("held-out $R^2$")
    # Floor below zero on purpose: the base chat-template cell's pooled bar is
    # NEGATIVE (-0.016), and a 0.0 floor renders it as no bar at all — reading
    # as missing data rather than as a map that fails on that cell.
    ax.set_ylim(*ylim)
    ax.set_title(f"Pooled vs cell-specific fit, evaluated per cell — {MODEL_TITLE[model]}")
    ax.legend(frameon=False, loc="upper left", ncols=1, handlelength=1.4)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.30)
    savefig_paper(fig, f"plot6c_pooled_lattice_{model}", dir=OUT_DIR)
    plt.close(fig)
    print(f"WROTE {OUT_DIR / f'plot6c_pooled_lattice_{model}.png'}")


def main() -> None:
    set_paper_style("iclr", font_scale=1.5)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    per_model = {m: load_cells(m) for m in ("instruct", "base")}
    vals = [
        v
        for cells in per_model.values()
        for c in cells
        for v in (c["pooled_m0_r2"], c["own_map_ceiling_r2"])
    ]
    ylim_top = float(np.ceil((max(vals) * 1.28) * 20) / 20)  # headroom for the legend
    ylim_bot = float(np.floor(min(0.0, min(vals) - 0.02) * 20) / 20)
    ylim = (ylim_bot, ylim_top)

    for model in ("instruct", "base"):
        plot_model(model, per_model[model], ylim)

    (OUT_DIR / "plot6c_pooled_lattice_data.json").write_text(
        json.dumps(
            {
                "source": SRC,
                "issue": 2054,
                "layer": 19,
                "arm": "context",
                "condition": CONDITION,
                "story_boundary_form": "attrib_quoted",
                "shared_ylim": list(ylim),
                "fit": "GCV ridge (dof cap 0.9), shared conversation-grouped 5-fold CV, seed 137",
                "pooled_map_def": "ONE map fit jointly on all 56 #2054 cells (both models, "
                "every framing x identity x condition), scored on each cell's held-out folds",
                "own_map_def": "that cell's banked within-cell ceiling (same estimator, "
                "fit on that cell alone)",
                "bars_drawn": ["pooled_m0_r2", "own_map_ceiling_r2"],
                "pooled_bias_m1_not_drawn": "r2_m1_bias retained per cell below; the +bias "
                "rung is omitted from the figure to match the plot6b two-bar grammar",
                "caveats": {
                    "instruct_assistant_bare_text_caphit": "the instruct assistant bare-text "
                    "on-policy cell is #2054's 42.5% cap-hit cell (42.5% of generations ran "
                    "to the token cap); its low pooled fraction is partly a truncation "
                    "artifact, not only a map property",
                    "bare_label_form_excluded": "the bare-label story boundary carries a "
                    "trailing-space tokenization artifact (23.2% digit-start onsets); the "
                    "attributed-quote form is drawn instead",
                },
                "cells": per_model,
            },
            indent=1,
        )
    )
    print(f"WROTE {OUT_DIR / 'plot6c_pooled_lattice_data.json'}")


if __name__ == "__main__":
    main()
